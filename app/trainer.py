"""
app/trainer.py — Entrenamiento de la CNN con los datos recolectados

Pasos:
  1. Carga el dataset completo desde disco
  2. Aumentación de datos (ruido + escala pequeña)
  3. Split train / validation
  4. Entrena la CNN
  5. Muestra métricas y guarda el modelo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils.storage import DataStorage
from utils.model import build_model, get_callbacks


class ModelTrainer:
    def __init__(self):
        self.storage = DataStorage()

    def train(self, epochs: int = 50):
        # ── 1. Cargar datos ────────────────────────────────────────────────
        print("  Cargando dataset…")
        X, y, labels = self.storage.load_dataset()

        print(f"  Total muestras : {len(X)}")
        print(f"  Clases ({len(labels)}): {labels}")
        for i, lbl in enumerate(labels):
            n = int((y == i).sum())
            print(f"    {lbl:<20} {n} muestras")

        if len(X) < 20:
            raise ValueError("Muy pocas muestras. Recolecta más datos.")

        # ── 2. Aumentación ─────────────────────────────────────────────────
        X_aug, y_aug = _augment(X, y, factor=3)
        print(f"\n  Muestras tras aumentación: {len(X_aug)}")

        # ── 3. Split ───────────────────────────────────────────────────────
        X_train, X_val, y_train, y_val = train_test_split(
            X_aug, y_aug,
            test_size=0.2,
            random_state=42,
            stratify=y_aug,
        )
        print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}\n")

        # ── 4. Pesos de clase (maneja desbalanceo) ─────────────────────────
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        cw_dict = dict(enumerate(class_weights))

        # ── 5. Modelo ──────────────────────────────────────────────────────
        model = build_model(num_classes=len(labels))
        model.summary()

        callbacks = get_callbacks(self.storage.get_model_path())

        print("\n  ⏳ Entrenando…\n")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=1,
        )

        # ── 6. Evaluación final ────────────────────────────────────────────
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\n  ✓ Entrenamiento completo")
        print(f"    Val Accuracy : {acc * 100:.1f}%")
        print(f"    Val Loss     : {loss:.4f}")

        # ── 7. Guardar etiquetas ───────────────────────────────────────────
        self.storage.save_labels(labels)
        print(f"    Modelo guardado en: {self.storage.get_model_path()}")
        print(f"    Etiquetas: {labels}")
        print(f"\n  ▶  Prueba: python main.py\n")

        return history, acc


# ─── Aumentación de datos ─────────────────────────────────────────────────────

def _augment(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple:
    """
    Genera versiones aumentadas de cada muestra:
      - Ruido gaussiano leve
      - Escala pequeña aleatoria
      - Volteo horizontal (intercambia mano izq/der)
    """
    X_out = [X]
    y_out = [y]

    rng = np.random.default_rng(42)

    for _ in range(factor - 1):
        noise = rng.normal(0, 0.01, size=X.shape).astype(np.float32)
        scale = rng.uniform(0.9, 1.1, size=(len(X), 1)).astype(np.float32)
        X_noisy = (X + noise) * scale
        X_out.append(X_noisy)
        y_out.append(y)

    # Volteo: intercambia los 63 features de izquierda y derecha
    X_flip = X.copy()
    X_flip[:, :63], X_flip[:, 63:] = X[:, 63:].copy(), X[:, :63].copy()
    X_out.append(X_flip)
    y_out.append(y)

    return np.concatenate(X_out, axis=0), np.concatenate(y_out, axis=0)
