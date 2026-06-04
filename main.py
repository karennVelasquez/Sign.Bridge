"""
main.py — Sign.Bridge  (versión completa con frontend web)

Comandos:
  python main.py --server          → servidor web http://localhost:8000
  python main.py --collect HOLA    → recolección OpenCV (modo terminal)
  python main.py --train           → entrenar CNN
  python main.py --list            → listar señas
  python main.py --delete HOLA     → borrar seña
  python main.py                   → predicción OpenCV (si hay modelo)
"""

import argparse
import sys
import json
import threading
import time
import collections
import os

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.storage import DataStorage
from utils.hands import HandExtractor

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Sign.Bridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────────────
storage         = DataStorage()
extractor       = HandExtractor(max_hands=2)
_extractor_lock = threading.Lock()
_model          = None
_labels         = []
_model_lock     = threading.Lock()

# Throttle para /api/collect: máx 20 fps de guardado por seña
_last_collect: dict = {}
MIN_COLLECT_INTERVAL = 0.05



def _load_model():
    global _model, _labels
    if not storage.model_exists():
        return
    try:
        from tensorflow import keras
        with _model_lock:
            _model  = keras.models.load_model(storage.get_model_path())
            _labels = storage.load_labels()
        print(f"✓ Modelo cargado — {len(_labels)} señas: {_labels}")
    except Exception as e:
        print(f"⚠ No se pudo cargar el modelo: {e}")


_load_model()

# ── Frontend estático ─────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")


# ─────────────────────────────────────────────────────────────────────────────
#  API REST
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/labels")
def get_labels():
    """Señas del modelo entrenado."""
    if storage.model_exists():
        try:
            return storage.load_labels()
        except Exception:
            pass
    return []


@app.get("/api/words")
def get_words():
    """Todas las señas recolectadas con conteo de muestras."""
    return [
        {"word": w, "samples": storage.count_samples(w)}
        for w in storage.list_words()
    ]


@app.post("/api/sample")
async def collect_sample(request: Request):
    word = request.headers.get("X-Word", "").strip().upper()
    if not word:
        return JSONResponse({"error": "Falta el header X-Word"}, status_code=400)

    try:
        target = int(request.headers.get("X-Target", "200"))
    except ValueError:
        target = 200

    # Verificar objetivo ANTES de procesar
    current = storage.count_samples(word)
    if current >= target:
        return JSONResponse({"saved": False, "reason": "target_reached",
                             "hand_detected": True, "word": word, "total": current})

    # Throttle (igual que collector.py MIN_INTERVAL)
    now  = time.time()
    last = _last_collect.get(word, 0.0)
    if now - last < MIN_COLLECT_INTERVAL:
        return JSONResponse({"saved": False, "reason": "throttle", "total": current})
    _last_collect[word] = now

    body  = await request.body()
    print(f"[/api/sample] {word}: recibido {len(body)} bytes")
    if len(body) < 100:
        print(f"[/api/sample] ⚠ Frame demasiado pequeño ({len(body)} bytes)")
        return JSONResponse({"error": f"Frame muy pequeño: {len(body)} bytes"}, status_code=400)

    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        print("[/api/sample] ⚠ cv2.imdecode retornó None — JPEG inválido")
        return JSONResponse({"error": "Frame inválido (imdecode falló)"}, status_code=400)

    print(f"[/api/sample] frame decodificado: {frame.shape}")
    frame = cv2.flip(frame, 1)

    with _extractor_lock:
        features, info = extractor.process(frame)

    print(f"[/api/sample] MediaPipe → count={info['count']} left={info['left']} right={info['right']}")

    if features is None or info["count"] == 0:
        return JSONResponse({"saved": False, "hand_detected": False,
                             "word": word, "total": storage.count_samples(word)})

    # Doble verificación antes de guardar
    total = storage.count_samples(word)
    if total >= target:
        return JSONResponse({"saved": False, "reason": "target_reached",
                             "hand_detected": True, "word": word, "total": total})

    storage.save_sample(word, features)
    total = storage.count_samples(word)

    return JSONResponse({"saved": True, "hand_detected": True, "word": word,
                         "total": total, "complete": total >= target,
                         "hands": info["count"]})


# Variable global para rastrear el estado del entrenamiento
_train_status = {"running": False, "epoch": 0, "total": 50, "done": False,
                 "val_acc": 0.0, "loss": 0.0, "log": [],
                 "history": {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}}

@app.post("/api/train")
def train_endpoint():
    global _train_status
    words = storage.list_words()
    if len(words) < 2:
        return JSONResponse(
            {"error": f"Necesitas al menos 2 señas. Tienes: {words}"},
            status_code=400,
        )
    if _train_status["running"]:
        return JSONResponse({"error": "Ya hay un entrenamiento en curso"}, status_code=400)

    _train_status = {"running": True, "epoch": 0, "total": 50,
                     "done": False, "val_acc": 0.0, "loss": 0.0, "log": [],
                     "history": {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}}

    def _run():
        global _train_status
        try:
            import tensorflow as tf
            from app.trainer import ModelTrainer

            # Callback personalizado para capturar el progreso real
            class FrontendCallback(tf.keras.callbacks.Callback):
                def _push(self, msg):
                    """Envía mensaje a todos los clientes WS conectados."""
                    _train_status["log"].append(msg)
                    # Broadcast a clientes WebSocket
                    dead = set()
                    for ws in list(_train_ws_clients):
                        try:
                            asyncio.run_coroutine_threadsafe(
                                ws.send_text(msg),
                                asyncio.get_event_loop()
                            )
                        except:
                            dead.add(ws)
                    _train_ws_clients.difference_update(dead)

                def on_epoch_end(self, epoch, logs=None):
                    logs    = logs or {}
                    val_acc = float(logs.get("val_accuracy", logs.get("val_acc", 0)))
                    loss    = float(logs.get("loss", 0))
                    val_loss = float(logs.get("val_loss", 0))
                    accuracy = float(logs.get("accuracy", logs.get("acc", 0)))
                    total   = self.params.get("epochs", 50)
                    line    = f"Época {epoch+1}/{total} — loss: {loss:.4f} — val_acc: {val_acc:.4f}"
                    _train_status.update({"epoch": epoch+1, "total": total,
                                        "val_acc": val_acc, "loss": loss})
                    # Acumular historial para la gráfica
                    _train_status["history"]["loss"].append(round(loss, 4))
                    _train_status["history"]["val_loss"].append(round(val_loss, 4))
                    _train_status["history"]["accuracy"].append(round(accuracy, 4))
                    _train_status["history"]["val_accuracy"].append(round(val_acc, 4))
                    self._push(line)

                def on_train_end(self, logs=None):
                    self._push("✓ Modelo guardado en models/sign_model.keras")

            trainer = ModelTrainer()
            trainer.train(epochs=50, extra_callbacks=[FrontendCallback()])
            _load_model()
            _train_status["done"]    = True
            _train_status["running"] = False
            _train_status["log"].append("✓ Modelo guardado en models/sign_model.keras")
            print("✓ Entrenamiento completado y modelo recargado.")
        except Exception as e:
            _train_status["running"] = False
            _train_status["done"]    = False
            _train_status["log"].append(f"✗ Error: {e}")
            print(f"✗ Error en entrenamiento: {e}")

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"message": f"Entrenamiento iniciado — señas: {words}", "words": words})


@app.get("/api/train/status")
def train_status():
    """El frontend hace polling a este endpoint para conocer el progreso real."""
    return JSONResponse(_train_status)


@app.get("/api/train/chart")
def train_chart():
    """Sirve la imagen PNG de la gráfica de entrenamiento."""
    from fastapi.responses import FileResponse, Response
    chart_path = "models/training_history.png"
    if not os.path.exists(chart_path):
        return Response(status_code=404)
    # Cache-busting: el archivo cambia cada entrenamiento
    return FileResponse(chart_path, media_type="image/png",
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.delete("/api/words/{word}")
def delete_word(word: str):
    ok = storage.delete_word(word.upper())
    if ok:
        return JSONResponse({"deleted": word.upper()})
    return JSONResponse({"error": f"'{word}' no encontrada"}, status_code=404)


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket — predicción en tiempo real
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW     = 10
MIN_STABLE_FRAMES    = 6

import asyncio
_train_queue: asyncio.Queue = None
_train_ws_clients = set()

@app.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    _train_ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        _train_ws_clients.discard(websocket)

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Cliente WS conectado")

    pred_buffer  = collections.deque(maxlen=SMOOTHING_WINDOW)
    stable_word  = None
    stable_count = 0

    try:
        while True:
            data  = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text(json.dumps(
                    {"letter": "—", "confidence": 0,
                     "hand_detected": False, "stable": False}
                ))
                continue

            frame = cv2.flip(frame, 1)

            with _extractor_lock:
                features, info = extractor.process(frame)

            if features is None or info["count"] == 0:
                pred_buffer.append(None)
                stable_count = 0
                await websocket.send_text(json.dumps(
                    {"letter": "—", "confidence": 0,
                     "hand_detected": False, "stable": False}
                ))
                continue

            with _model_lock:
                m      = _model
                labels = list(_labels)

            if m is None or not labels:
                await websocket.send_text(json.dumps(
                    {"letter": "Sin modelo", "confidence": 0,
                     "hand_detected": True, "stable": False}
                ))
                continue

            inp   = features[np.newaxis, :]
            probs = m.predict(inp, verbose=0)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            word  = labels[idx] if conf >= CONFIDENCE_THRESHOLD else None

            pred_buffer.append(word)
            valid = [w for w in pred_buffer if w is not None]

            if valid:
                from collections import Counter
                top_word, top_count = Counter(valid).most_common(1)[0]
                smooth_conf = (top_count / len(pred_buffer)
                               + float(probs[labels.index(top_word)])) / 2
            else:
                top_word    = None
                smooth_conf = 0.0

            if top_word and smooth_conf >= CONFIDENCE_THRESHOLD:
                if top_word == stable_word:
                    stable_count += 1
                else:
                    stable_word  = top_word
                    stable_count = 1
            else:
                stable_count = 0
                stable_word  = None

            is_stable = stable_count >= MIN_STABLE_FRAMES

            await websocket.send_text(json.dumps({
                "letter":        top_word or "—",
                "confidence":    round(smooth_conf * 100),
                "hand_detected": True,
                "stable":        is_stable,
            }))

    except WebSocketDisconnect:
        print("🔌 Cliente WS desconectado")
    except Exception as e:
        print(f"⚠ WS error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def cli():
    parser = argparse.ArgumentParser(description="Sign.Bridge")
    parser.add_argument("--server",  action="store_true")
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--collect", type=str, metavar="PALABRA")
    parser.add_argument("--list",    action="store_true")
    parser.add_argument("--delete",  type=str, metavar="PALABRA")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--port",    type=int, default=8000)
    args = parser.parse_args()

    if args.list:
        words = storage.list_words()
        if not words:
            print("\n  Sin señas registradas. Usa: python main.py --collect HOLA\n")
        else:
            print(f"\n  Señas ({len(words)}):")
            for w in words:
                print(f"   ✦ {w:<22} {storage.count_samples(w)} muestras")
            print()
        return

    if args.delete:
        w = args.delete.upper()
        print(f"  {'✓' if storage.delete_word(w) else '⚠'} '{w}' {'eliminada' if storage.delete_word(w) else 'no encontrada'}.")
        return

    if args.collect:
        from app.collector import DataCollector
        w = args.collect.upper()
        print(f"\n  Recolectando '{w}' — objetivo: {args.samples} muestras")
        print("  [E] grabar/pausar   [Q] salir\n")
        DataCollector(camera_index=args.camera).collect(word=w, target_samples=args.samples)
        return

    if args.train:
        words = storage.list_words()
        if len(words) < 2:
            print(f"\n  ⚠ Necesitas al menos 2 señas. Tienes: {words}\n")
            sys.exit(1)
        from app.trainer import ModelTrainer
        print(f"\n  Entrenando con {len(words)} señas: {words}\n")
        ModelTrainer().train(epochs=args.epochs)
        return

    # --server o sin argumentos → servidor web
    import uvicorn
    print(f"\n🌐 Sign.Bridge — http://localhost:{args.port}\n")
    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    cli()