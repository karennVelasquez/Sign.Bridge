"""
utils/hands.py — Extracción de landmarks con MediaPipe Hands
           Detecta mano izquierda y derecha de forma independiente.
"""

import cv2
import numpy as np
import mediapipe as mp

# ─── Constantes ─────────────────────────────────────────────────────────────

NUM_LANDMARKS   = 21          # puntos por mano
NUM_COORDS      = 3           # x, y, z
ONE_HAND_FEATS  = NUM_LANDMARKS * NUM_COORDS        # 63
TWO_HANDS_FEATS = ONE_HAND_FEATS * 2                # 126  (izq + der concatenadas)

# Colores BGR
COLOR_LEFT  = (255, 100, 80)   # azul-morado  → mano izquierda
COLOR_RIGHT = (80, 101, 255)   # coral-rojo   → mano derecha
COLOR_CONN  = (200, 200, 200)

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles


# ─── Clase principal ─────────────────────────────────────────────────────────

class HandExtractor:
    """
    Envuelve MediaPipe Hands y extrae un vector de características
    normalizado de longitud fija (TWO_HANDS_FEATS = 126).

    Layout del vector resultante:
        [0:63]   → landmarks mano izquierda  (zeros si no detectada)
        [63:126] → landmarks mano derecha    (zeros si no detectada)
    """

    def __init__(self, max_hands: int = 2, confidence: float = 0.7):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.6,
        )
        self.detected_left  = False
        self.detected_right = False

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray | None, dict]:
        """
        Procesa un frame BGR y retorna:
            features : np.ndarray shape (126,) o None si no hay manos
            info     : dict con 'left', 'right', 'count'
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        info = {"left": False, "right": False, "count": 0}

        if not results.multi_hand_landmarks:
            self.detected_left  = False
            self.detected_right = False
            return None, info

        # Separar manos por etiqueta MediaPipe
        left_lm  = None
        right_lm = None

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = hand_info.classification[0].label  # "Left" / "Right"
            # El frame ya viene volteado con cv2.flip(frame, 1) antes de llamar
            # a este método, por lo que las etiquetas de MediaPipe coinciden
            # directamente con la mano real del usuario.
            if label == "Left":
                left_lm  = hand_lm
            else:
                right_lm = hand_lm

        self.detected_left  = left_lm  is not None
        self.detected_right = right_lm is not None
        info["left"]  = self.detected_left
        info["right"] = self.detected_right
        info["count"] = int(self.detected_left) + int(self.detected_right)
        info["raw"]   = results  # para dibujar

        # Construir vector de características
        left_vec  = self._extract_landmarks(left_lm,  frame_bgr.shape)
        right_vec = self._extract_landmarks(right_lm, frame_bgr.shape)

        features = np.concatenate([left_vec, right_vec], axis=0).astype(np.float32)
        return features, info

    def _extract_landmarks(
        self,
        hand_lm,
        frame_shape: tuple
    ) -> np.ndarray:
        """
        Extrae y normaliza los landmarks de una mano.
        Si hand_lm es None retorna zeros.

        Normalización:
          - Centrado en la muñeca (punto 0)
          - Escalado por la distancia máxima dentro de la mano
          - Independiente de posición y tamaño en el frame
        """
        if hand_lm is None:
            return np.zeros(ONE_HAND_FEATS, dtype=np.float32)

        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
            dtype=np.float32
        )  # (21, 3)

        # Centrado en muñeca (landmark 0)
        pts -= pts[0]

        # Normalización por escala
        scale = np.max(np.abs(pts)) + 1e-6
        pts /= scale

        return pts.flatten()  # (63,)

    def draw(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """Dibuja los landmarks y conexiones sobre el frame."""
        results = info.get("raw")
        if not results or not results.multi_hand_landmarks:
            return frame

        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = hand_info.classification[0].label
            # Frame ya está volteado → etiqueta coincide con mano real
            color = COLOR_LEFT if label == "Left" else COLOR_RIGHT

            # Dibuja conexiones
            mp_drawing.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=COLOR_CONN, thickness=1),
            )

            # Etiqueta de mano
            h, w = frame.shape[:2]
            wrist = hand_lm.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            hand_label = "IZQ" if label == "Left" else "DER"
            cv2.rectangle(frame, (cx - 30, cy - 22), (cx + 30, cy - 4), color, -1)
            cv2.putText(frame, hand_label, (cx - 18, cy - 7),
                        cv2.FONT_HERSHEY_DUPLEX, 0.38, (255, 255, 255), 1)

        return frame

    def close(self):
        self.hands.close()
