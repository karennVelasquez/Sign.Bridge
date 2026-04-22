"""
utils/storage.py — Gestión de almacenamiento de datos y modelos
"""

import os
import json
import numpy as np
import shutil
from pathlib import Path


BASE_DIR    = Path("data")
MODEL_DIR   = Path("models")
MODEL_PATH  = MODEL_DIR / "sign_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
META_PATH   = BASE_DIR / "metadata.json"


class DataStorage:
    def __init__(self):
        BASE_DIR.mkdir(exist_ok=True)
        MODEL_DIR.mkdir(exist_ok=True)

    # ─── Palabras / clases ──────────────────────────────────────────────────

    def list_words(self) -> list[str]:
        words = []
        if BASE_DIR.exists():
            for p in sorted(BASE_DIR.iterdir()):
                if p.is_dir():
                    words.append(p.name)
        return words

    def count_samples(self, word: str) -> int:
        word_dir = BASE_DIR / word.upper()
        if not word_dir.exists():
            return 0
        return len(list(word_dir.glob("*.npy")))

    def delete_word(self, word: str) -> bool:
        word_dir = BASE_DIR / word.upper()
        if word_dir.exists():
            shutil.rmtree(word_dir)
            return True
        return False

    # ─── Guardar / cargar muestras ──────────────────────────────────────────

    def save_sample(self, word: str, landmarks: np.ndarray):
        """Guarda un array de landmarks como archivo .npy"""
        word_dir = BASE_DIR / word.upper()
        word_dir.mkdir(exist_ok=True)
        idx = self.count_samples(word)
        path = word_dir / f"{idx:05d}.npy"
        np.save(path, landmarks)

    def load_dataset(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Carga todo el dataset.
        Retorna: (X, y, labels)
          X: (N, features)  — landmarks normalizados
          y: (N,)           — índices de clase (enteros)
          labels: lista de palabras ordenadas
        """
        labels = self.list_words()
        X, y = [], []

        for idx, word in enumerate(labels):
            word_dir = BASE_DIR / word
            files = sorted(word_dir.glob("*.npy"))
            if not files:
                continue
            for f in files:
                arr = np.load(f)
                X.append(arr)
                y.append(idx)

        if not X:
            raise ValueError("Dataset vacío. Recolecta datos primero.")

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels

    # ─── Modelo ─────────────────────────────────────────────────────────────

    def model_exists(self) -> bool:
        return MODEL_PATH.exists() and LABELS_PATH.exists()

    def save_labels(self, labels: list[str]):
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

    def load_labels(self) -> list[str]:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_model_path(self) -> str:
        return str(MODEL_PATH)
