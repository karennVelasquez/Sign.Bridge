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


def plot_training_history(history, save_path=None):
    """
    Guarda la gráfica de curvas de entrenamiento como PNG.
    Usa siempre el backend 'Agg' (sin GUI) — el frontend mostrará el PNG.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # SIEMPRE sin GUI (evita warnings de thread)
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  ⚠ matplotlib no instalado. Instálalo con: pip install matplotlib")
        return

    h = history.history
    epochs = range(1, len(h['loss']) + 1)

    # Detectar si hay accuracy o acc (keras < 2.x usaba 'acc')
    acc_key     = 'accuracy'     if 'accuracy'     in h else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in h else 'val_acc'

    fig = plt.figure(figsize=(12, 5), facecolor='#0f1117')
    gs  = gridspec.GridSpec(1, 2, figure=fig, hspace=0.35, wspace=0.32)

    COLORS = {
        'train_acc':  '#38d9c0',
        'val_acc':    '#f0c040',
        'train_loss': '#888888',
        'val_loss':   '#e05a5a',
        'grid':       '#ffffff18',
        'text':       '#cccccc',
        'bg':         '#0f1117',
        'panel':      '#1a1d27',
    }

    def _style_ax(ax, title):
        ax.set_facecolor(COLORS['panel'])
        ax.set_title(title, color=COLORS['text'], fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(colors=COLORS['text'], labelsize=9)
        ax.spines[:].set_color('#333344')
        ax.grid(True, color=COLORS['grid'], linewidth=0.7)
        ax.set_xlabel('Época', color=COLORS['text'], fontsize=10)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # ── Gráfica 1: Accuracy ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, h[acc_key],     color=COLORS['train_acc'],  lw=2,   label='Train Accuracy')
    ax1.plot(epochs, h[val_acc_key], color=COLORS['val_acc'],    lw=2.5, label='Val Accuracy',   linestyle='--')

    # Marcar el mejor val_accuracy
    best_epoch = int(np.argmax(h[val_acc_key])) + 1
    best_val   = max(h[val_acc_key])
    ax1.axvline(best_epoch, color=COLORS['val_acc'], lw=1, linestyle=':', alpha=0.6)
    ax1.scatter([best_epoch], [best_val], color=COLORS['val_acc'], zorder=5, s=60)
    ax1.annotate(f'  Best: {best_val*100:.1f}%',
                 xy=(best_epoch, best_val),
                 color=COLORS['val_acc'], fontsize=9)

    _style_ax(ax1, 'Accuracy')
    ax1.set_ylabel('Accuracy', color=COLORS['text'], fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(facecolor='#1a1d27', edgecolor='#333344',
               labelcolor=COLORS['text'], fontsize=9)

    # ── Gráfica 2: Loss ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, h['loss'],     color=COLORS['train_loss'], lw=2,   label='Train Loss')
    ax2.plot(epochs, h['val_loss'], color=COLORS['val_loss'],   lw=2.5, label='Val Loss', linestyle='--')

    # Marcar mínimo val_loss
    best_loss_epoch = int(np.argmin(h['val_loss'])) + 1
    best_loss_val   = min(h['val_loss'])
    ax2.axvline(best_loss_epoch, color=COLORS['val_loss'], lw=1, linestyle=':', alpha=0.6)
    ax2.scatter([best_loss_epoch], [best_loss_val], color=COLORS['val_loss'], zorder=5, s=60)
    ax2.annotate(f'  Min: {best_loss_val:.4f}',
                 xy=(best_loss_epoch, best_loss_val),
                 color=COLORS['val_loss'], fontsize=9)

    _style_ax(ax2, 'Loss')
    ax2.set_ylabel('Loss', color=COLORS['text'], fontsize=10)
    ax2.legend(facecolor='#1a1d27', edgecolor='#333344',
               labelcolor=COLORS['text'], fontsize=9)

    fig.suptitle('Sign.Bridge — Curvas de entrenamiento',
                 color='white', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Guardar siempre como PNG junto al modelo
    out = save_path or 'models/training_history.png'
    try:
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"    Gráfica guardada en: {out}")
    except Exception as e:
        print(f"  ⚠ No se pudo guardar la gráfica: {e}")

    # NO mostramos ventana: el frontend muestra el PNG via /api/train/chart
    plt.close('all')


class ModelTrainer:
    def __init__(self):
        self.storage = DataStorage()

    def train(self, epochs: int = 50, extra_callbacks=None):
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
        all_callbacks = callbacks + (extra_callbacks or [])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            class_weight=cw_dict,
            callbacks=all_callbacks,
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

        # ── 8. Gráfica de entrenamiento ────────────────────────────────────
        plot_training_history(history, save_path='models/training_history.png')

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