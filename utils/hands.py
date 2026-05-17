"""
utils/hands.py — Extracción de landmarks con MediaPipe Hands (nueva API Tasks)
           Detecta mano izquierda y derecha de forma independiente.
           Compatible con mediapipe >= 0.10.9 (Python 3.10 y 3.11)
"""

from cProfile import label

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
import urllib.request
import os

# ─── Constantes ─────────────────────────────────────────────────────────────

NUM_LANDMARKS   = 21
NUM_COORDS      = 3
ONE_HAND_FEATS  = NUM_LANDMARKS * NUM_COORDS   # 63
TWO_HANDS_FEATS = ONE_HAND_FEATS * 2           # 126

# Colores BGR
COLOR_LEFT  = (255, 100, 80)
COLOR_RIGHT = (80, 101, 255)
COLOR_CONN  = (200, 200, 200)

# Conexiones de la mano
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# Ruta del modelo .task de MediaPipe
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _ensure_model():
    """Descarga el modelo .task si no existe localmente."""
    if not os.path.exists(MODEL_PATH):
        print("  Descargando modelo MediaPipe Hands (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  ✓ Modelo descargado.")


# ─── Clase principal ─────────────────────────────────────────────────────────

class HandExtractor:
    """
    Usa la nueva API MediaPipe Tasks para detectar manos.

    Layout del vector resultante (126 valores):
        [0:63]   → landmarks mano izquierda  (zeros si no detectada)
        [63:126] → landmarks mano derecha    (zeros si no detectada)
    """

    def __init__(self, max_hands: int = 2, confidence: float = 0.7):
        _ensure_model()

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=0.6,
        )
        self.landmarker     = HandLandmarker.create_from_options(options)
        self.detected_left  = False
        self.detected_right = False

    def process(self, frame_bgr: np.ndarray) -> tuple:
        """
        Procesa un frame BGR y retorna:
            features : np.ndarray shape (126,) o None si no hay manos
            info     : dict con 'left', 'right', 'count', 'raw'
        """
        rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        info = {"left": False, "right": False, "count": 0, "raw": result,
                "shape": frame_bgr.shape}

        if not result.hand_landmarks:
            self.detected_left  = False
            self.detected_right = False
            return None, info

        left_lm  = None
        right_lm = None

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # El frame ya viene volteado con cv2.flip → etiqueta coincide con mano real
            label = handedness[0].category_name  # "Left" o "Right"
            if label == "Left":
                left_lm  = landmarks
            else:
                right_lm = landmarks    

        self.detected_left  = left_lm  is not None
        self.detected_right = right_lm is not None
        info["left"]  = self.detected_left
        info["right"] = self.detected_right
        info["count"] = int(self.detected_left) + int(self.detected_right)

        left_vec  = self._extract_landmarks(left_lm)
        right_vec = self._extract_landmarks(right_lm)
        features  = np.concatenate([left_vec, right_vec], axis=0).astype(np.float32)

        return features, info

    def _extract_landmarks(self, landmarks) -> np.ndarray:
        """
        Extrae y normaliza los landmarks de una mano.
        Si landmarks es None retorna zeros.
        Normalización: centrado en muñeca + escalado por extensión máxima.
        """
        if landmarks is None:
            return np.zeros(ONE_HAND_FEATS, dtype=np.float32)

        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float32
        )  # (21, 3)

        pts -= pts[0]                        # centrar en muñeca
        scale = np.max(np.abs(pts)) + 1e-6
        pts  /= scale                        # normalizar escala

        return pts.flatten()  # (63,)

    def draw(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """Dibuja los landmarks y conexiones sobre el frame."""
        result = info.get("raw")
        if not result or not result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            label = handedness[0].category_name
            color = COLOR_LEFT if label == "Right" else COLOR_RIGHT

            # Puntos como píxeles
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Dibujar conexiones
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts_px[a], pts_px[b], COLOR_CONN, 1)

            # Dibujar puntos
            for px, py in pts_px:
                cv2.circle(frame, (px, py), 3, color, -1)

            # Etiqueta en la muñeca
            cx, cy = pts_px[0]
            hand_label = "IZQ" if label == "Right" else "DER"
            cv2.rectangle(frame, (cx - 30, cy - 22), (cx + 30, cy - 4), color, -1)
            cv2.putText(frame, hand_label, (cx - 18, cy - 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 255, 255), 1)

        return frame

    def close(self):
        self.landmarker.close()