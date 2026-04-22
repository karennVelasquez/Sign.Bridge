"""
utils/model.py — Arquitectura de la red neuronal CNN para señas estáticas

La entrada son landmarks normalizados (126 features = 2 manos × 21 puntos × 3 coords).
Se reformatean como "imagen" 1D para aplicar capas Conv1D que capturen
relaciones espaciales entre dedos y articulaciones.

Arquitectura:
    Input (126,)
        └─ Reshape (21, 6)          ← 21 landmarks, 6 valores (x,y,z izq + x,y,z der)
           └─ Conv1D 64, k=3        ← patrones locales entre articulaciones contiguas
              └─ Conv1D 128, k=3
                 └─ GlobalAvgPool
                    └─ Dense 256
                       └─ Dropout 0.4
                          └─ Dense 128
                             └─ Dropout 0.3
                                └─ Dense num_classes (softmax)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─── Construcción del modelo ─────────────────────────────────────────────────

def build_model(num_classes: int, input_dim: int = 126) -> keras.Model:
    """
    Construye y retorna la CNN para clasificación de señas.

    Args:
        num_classes: número de palabras/señas a clasificar
        input_dim:   dimensión del vector de entrada (default 126)

    Returns:
        Modelo Keras compilado
    """
    inp = keras.Input(shape=(input_dim,), name="landmarks")

    # ── Rama CNN  ──────────────────────────────────────────────────────────
    # Reformateamos: 21 landmarks × 6 (izq_xyz + der_xyz juntos por landmark)
    x = layers.Reshape((21, 6), name="reshape")(inp)

    x = layers.Conv1D(64,  kernel_size=3, padding="same", activation="relu",
                      name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu",
                      name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)

    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Rama densa ─────────────────────────────────────────────────────────
    x = layers.Dense(256, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.4, name="drop1")(x)

    x = layers.Dense(128, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="SignLens_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(model_path: str) -> list:
    """Callbacks de entrenamiento: checkpoint + early stopping + lr reduction."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
