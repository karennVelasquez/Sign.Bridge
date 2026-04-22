"""
app/translator.py — Inferencia en tiempo real con CNN entrenada

Características:
  - Predicción frame a frame con suavizado temporal (ventana deslizante)
  - Umbral de confianza configurable
  - Detección separada mano izquierda / mano derecha
  - Historial de palabras detectadas
  - UI limpia con indicadores de confianza
"""

import cv2
import numpy as np
import time
import collections
import tensorflow as tf
from tensorflow import keras
from utils.hands import HandExtractor
from utils.storage import DataStorage


# ─── Configuración ────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.70    # mínima confianza para mostrar predicción
SMOOTHING_WINDOW     = 10      # frames para suavizado de predicciones
MIN_STABLE_FRAMES    = 6       # cuántos frames seguidos para confirmar palabra
HISTORY_MAX          = 8       # cuántas palabras en el historial


class RealTimeTranslator:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.storage      = DataStorage()
        self.extractor    = HandExtractor(max_hands=2)

        # Cargar modelo y etiquetas
        print("  Cargando modelo CNN…")
        self.model  = keras.models.load_model(self.storage.get_model_path())
        self.labels = self.storage.load_labels()
        print(f"  ✓ Modelo listo — {len(self.labels)} señas: {self.labels}\n")

        # Suavizado
        self._pred_buffer  = collections.deque(maxlen=SMOOTHING_WINDOW)
        self._stable_word  = None
        self._stable_count = 0
        self._history      = []
        self._last_added   = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_timer  = time.time()
        fps_count  = 0
        fps_val    = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            t0    = time.time()

            # ── Extracción de landmarks ────────────────────────────────────
            features, info = self.extractor.process(frame)
            frame = self.extractor.draw(frame, info)

            # ── Predicción ─────────────────────────────────────────────────
            pred_word   = None
            pred_conf   = 0.0
            all_probs   = None

            if features is not None and info["count"] > 0:
                inp        = features[np.newaxis, :]           # (1, 126)
                probs      = self.model.predict(inp, verbose=0)[0]  # (num_classes,)
                all_probs  = probs
                top_idx    = int(np.argmax(probs))
                pred_conf  = float(probs[top_idx])
                pred_word  = self.labels[top_idx] if pred_conf >= CONFIDENCE_THRESHOLD else None

            # ── Suavizado temporal ─────────────────────────────────────────
            self._pred_buffer.append(pred_word)
            smooth_word, smooth_conf = self._smooth(all_probs)

            # ── Confirmación estable ───────────────────────────────────────
            if smooth_word and smooth_conf >= CONFIDENCE_THRESHOLD:
                if smooth_word == self._stable_word:
                    self._stable_count += 1
                else:
                    self._stable_word  = smooth_word
                    self._stable_count = 1

                if self._stable_count >= MIN_STABLE_FRAMES:
                    if smooth_word != self._last_added:
                        self._history.append(smooth_word)
                        if len(self._history) > HISTORY_MAX:
                            self._history.pop(0)
                        self._last_added = smooth_word
            else:
                self._stable_count = 0
                self._stable_word  = None
                self._last_added   = None

            # ── FPS ────────────────────────────────────────────────────────
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps_val   = fps_count
                fps_count = 0
                fps_timer = time.time()

            # ── Dibujar UI ─────────────────────────────────────────────────
            _draw_ui(frame, smooth_word, smooth_conf, all_probs,
                     self.labels, self._history, info, fps_val,
                     self._stable_count, MIN_STABLE_FRAMES)

            cv2.imshow("SignLens — Traductor en Tiempo Real", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c') or key == ord('C'):
                self._history.clear()
                self._last_added = None

        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def _smooth(self, all_probs) -> tuple[str | None, float]:
        """Promedia las últimas N probabilidades para predicción estable."""
        if all_probs is None:
            return None, 0.0

        # Usar solo los últimos frames que tienen all_probs equivalente
        # (simplificación: promediar el buffer de palabras por votación)
        valid = [w for w in self._pred_buffer if w is not None]
        if not valid:
            return None, 0.0

        from collections import Counter
        counter    = Counter(valid)
        top_word   = counter.most_common(1)[0][0]
        top_count  = counter.most_common(1)[0][1]
        confidence = top_count / len(self._pred_buffer)

        # También considerar la confianza del modelo en tiempo real
        if all_probs is not None:
            idx = self.labels.index(top_word) if top_word in self.labels else -1
            if idx >= 0:
                model_conf = float(all_probs[idx])
                confidence = (confidence + model_conf) / 2

        return top_word, confidence


# ─── Dibujo de UI ─────────────────────────────────────────────────────────────

def _draw_ui(frame, word, conf, probs, labels, history,
             info, fps, stable_count, stable_needed):
    h, w = frame.shape[:2]

    # ── Panel izquierdo: predicción principal ──────────────────────────────
    panel_w = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (8, 8, 18), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.line(frame, (panel_w, 0), (panel_w, h), (40, 40, 70), 1)

    # Logo / título
    cv2.putText(frame, "SignLens", (14, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 210, 255), 1)
    cv2.putText(frame, "CNN TRANSLATOR", (14, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, (100, 100, 160), 1)

    # Separador
    cv2.line(frame, (14, 60), (panel_w - 14, 60), (40, 40, 70), 1)

    # Palabra detectada (grande)
    display_word = word if word else "---"
    conf_color   = _conf_color(conf)

    # Fondo de la palabra
    cv2.rectangle(frame, (10, 70), (panel_w - 10, 130), (20, 20, 40), -1)
    cv2.rectangle(frame, (10, 70), (panel_w - 10, 130), conf_color, 1)

    font_scale = max(0.6, min(1.2, 12 / max(len(display_word), 1)))
    text_size  = cv2.getTextSize(display_word, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0]
    tx = (panel_w - text_size[0]) // 2
    cv2.putText(frame, display_word, (tx, 112),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, conf_color, 2)

    # Barra de confianza
    _draw_confidence_bar(frame, conf, 14, 140, panel_w - 28, 14, conf_color)

    # Estabilidad
    cv2.putText(frame, f"Confianza: {conf*100:.0f}%", (14, 172),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (160, 160, 200), 1)
    stab_dots = "●" * min(stable_count, stable_needed) + "○" * max(0, stable_needed - stable_count)
    cv2.putText(frame, f"Estabilidad: {stab_dots}", (14, 190),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (100, 180, 100), 1)

    # Separador
    cv2.line(frame, (14, 205), (panel_w - 14, 205), (40, 40, 70), 1)

    # Top predicciones
    if probs is not None:
        cv2.putText(frame, "TOP SEÑAS", (14, 222),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, (100, 100, 160), 1)
        top_n  = min(5, len(labels))
        sorted_idx = np.argsort(probs)[::-1][:top_n]
        y_off  = 242
        for rank, idx in enumerate(sorted_idx):
            bar_conf = float(probs[idx])
            lbl      = labels[idx]
            is_top   = (rank == 0 and bar_conf >= 0.5)
            color    = (0, 210, 255) if is_top else (100, 100, 160)
            cv2.putText(frame, f"{lbl[:12]:<12}", (14, y_off),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
            bar_len = int((panel_w - 90) * bar_conf)
            cv2.rectangle(frame, (100, y_off - 10), (100 + bar_len, y_off - 2),
                          color, -1)
            cv2.putText(frame, f"{bar_conf*100:.0f}%", (panel_w - 44, y_off),
                        cv2.FONT_HERSHEY_DUPLEX, 0.36, color, 1)
            y_off += 22

    # Separador
    cv2.line(frame, (14, h - 145), (panel_w - 14, h - 145), (40, 40, 70), 1)

    # Historial
    cv2.putText(frame, "HISTORIAL", (14, h - 128),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, (100, 100, 160), 1)
    hist_text = "  ".join(history[-6:]) if history else "(vacío)"
    # Dividir si es muy largo
    words_in_hist = hist_text.split("  ")
    line1 = "  ".join(words_in_hist[:4])
    line2 = "  ".join(words_in_hist[4:]) if len(words_in_hist) > 4 else ""
    cv2.putText(frame, line1, (14, h - 108),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, (200, 220, 255), 1)
    if line2:
        cv2.putText(frame, line2, (14, h - 88),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (200, 220, 255), 1)

    # Info manos
    cv2.line(frame, (14, h - 68), (panel_w - 14, h - 68), (40, 40, 70), 1)
    left_c  = (140, 80, 255) if info["left"]  else (50, 50, 80)
    right_c = (80, 100, 255) if info["right"] else (50, 50, 80)
    cv2.circle(frame, (25,  h - 50), 8, left_c,  -1)
    cv2.putText(frame, "IZQ", (36, h - 44), cv2.FONT_HERSHEY_DUPLEX, 0.4, left_c, 1)
    cv2.circle(frame, (100, h - 50), 8, right_c, -1)
    cv2.putText(frame, "DER", (111, h - 44), cv2.FONT_HERSHEY_DUPLEX, 0.4, right_c, 1)

    # FPS
    cv2.putText(frame, f"{fps} FPS", (panel_w - 65, h - 44),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (70, 70, 100), 1)

    # ── Controles (esquina inferior derecha) ───────────────────────────────
    cv2.putText(frame, "[Q] Salir   [C] Limpiar historial",
                (panel_w + 10, h - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.4, (80, 80, 110), 1)


def _draw_confidence_bar(frame, conf, x, y, width, height, color):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 50), -1)
    bar_len = int(width * conf)
    if bar_len > 0:
        cv2.rectangle(frame, (x, y), (x + bar_len, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 90), 1)


def _conf_color(conf: float) -> tuple:
    """Verde → confianza alta, amarillo → media, rojo → baja."""
    if conf >= 0.80:  return (0, 220, 80)
    if conf >= 0.60:  return (0, 200, 255)
    if conf >= 0.40:  return (0, 180, 255)
    return (80, 80, 200)
