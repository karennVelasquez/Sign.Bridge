"""
app/collector.py — Módulo de recolección de datos desde webcam

Flujo:
  1. Abre la cámara y muestra el feed con landmarks
  2. El usuario presiona [E] para iniciar/pausar la captura
  3. Por cada frame donde haya al menos una mano visible → guarda el vector
  4. Al llegar a target_samples muestra un resumen y cierra
"""

import cv2
import numpy as np
import time
from utils.hands import HandExtractor
from utils.storage import DataStorage


class DataCollector:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.extractor    = HandExtractor(max_hands=2)
        self.storage      = DataStorage()

    def collect(self, word: str, target_samples: int = 200):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        collecting    = False
        saved_count   = 0
        existing      = self.storage.count_samples(word)
        last_save_t   = 0.0
        MIN_INTERVAL  = 0.05   # máx 20 fps de guardado → evita duplicados idénticos

        print(f"\n  [{word}] Muestras existentes: {existing}")
        print(f"  Objetivo: {target_samples} muestras nuevas")
        print(f"  → Pon tu mano frente a la cámara y presiona [E]\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)          # espejo natural
            features, info = self.extractor.process(frame)
            frame = self.extractor.draw(frame, info)

            now = time.time()

            # Guardar si estamos capturando y hay manos detectadas
            if collecting and features is not None and info["count"] > 0:
                if now - last_save_t >= MIN_INTERVAL:
                    self.storage.save_sample(word, features)
                    saved_count  += 1
                    last_save_t   = now

            # UI ────────────────────────────────────────────────────────────
            h, w = frame.shape[:2]

            # Panel superior
            _draw_panel(frame, word, saved_count, target_samples,
                        collecting, info, existing)

            # Barra de progreso
            progress = min(saved_count / target_samples, 1.0)
            bar_w    = int((w - 40) * progress)
            cv2.rectangle(frame, (20, h - 30), (w - 20, h - 12), (40, 40, 60), -1)
            cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 12),
                          (0, 220, 120) if collecting else (80, 80, 140), -1)
            pct_text = f"{int(progress*100)}%  ({saved_count}/{target_samples})"
            cv2.putText(frame, pct_text, (25, h - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (220, 220, 220), 1)

            cv2.imshow(f"SignLens — Recolección: {word}", frame)

            # Fin automático
            if saved_count >= target_samples:
                _flash_done(frame, word, saved_count)
                cv2.imshow(f"SignLens — Recolección: {word}", frame)
                cv2.waitKey(1500)
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('e') or key == ord('E'):
                collecting = not collecting
                state = "▶ CAPTURANDO" if collecting else "⏸ PAUSADO"
                print(f"  {state} — Guardadas hasta ahora: {saved_count}")

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

        total = self.storage.count_samples(word)
        print(f"\n  ✓ Recolección finalizada.")
        print(f"    Nuevas muestras: {saved_count}")
        print(f"    Total para '{word}': {total}\n")

        words = self.storage.list_words()
        if len(words) >= 2:
            print("  ¿Quieres entrenar ahora?  →  python main.py --train\n")


# ─── Helpers UI ──────────────────────────────────────────────────────────────

def _draw_panel(frame, word, count, target, collecting, info, existing):
    h, w = frame.shape[:2]

    # Fondo semitransparente superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Título
    cv2.putText(frame, f"SEÑAL: {word}", (16, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 230, 255), 1)

    # Estado
    state_color = (0, 220, 80) if collecting else (180, 180, 60)
    state_txt   = "● REC" if collecting else "○ PAUSADO"
    cv2.putText(frame, state_txt, (16, 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, state_color, 1)

    # Manos detectadas
    hand_txt = ""
    if info["left"]:  hand_txt += "DER"
    if info["right"]: hand_txt += "IZQ"
    if not hand_txt:  hand_txt = "sin manos"
    hand_color = (80, 220, 80) if info["count"] else (80, 80, 200)
    cv2.putText(frame, f"Manos: {hand_txt}", (int(w * 0.45), 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, hand_color, 1)

    # Teclas
    cv2.putText(frame, "[E] capturar  [Q] salir", (int(w * 0.45), 55),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (150, 150, 180), 1)


def _flash_done(frame, word, count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 40, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    msg1 = f"¡Listo! {count} muestras de '{word}'"
    msg2 = "Entrena con:  python main.py --train"
    cv2.putText(frame, msg1, (w//2 - 220, h//2 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 120), 2)
    cv2.putText(frame, msg2, (w//2 - 200, h//2 + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (180, 255, 180), 1)