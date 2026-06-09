"""
app/trainer.py — Entrenamiento de la CNN con los datos recolectados

Pasos:
  1. Carga el dataset completo desde disco
  2. Aumentación de datos (ruido + escala pequeña)
  3. Split train / validation
  4. Entrena la CNN
  5. Muestra métricas y guarda el modelo
"""

import os
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


def _compute_confusion(model, X_val, y_val, n_classes):
    """Helper interno: calcula la matriz de confusión completa."""
    from sklearn.metrics import confusion_matrix
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def plot_top_confusions_matrix(model, X_val, y_val, labels, top_n=6, save_path=None):
    """
    Submatriz de confusión con las N clases más conflictivas.
    Selecciona automáticamente las clases con más errores (sumando errores
    como clase real + como clase predicha) y muestra solo esas.

    Útil para presentaciones donde una matriz 24x24 sería ilegible.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib no instalado.")
        return

    cm = _compute_confusion(model, X_val, y_val, len(labels))

    # Errores por clase: fila + columna - 2*diagonal
    errors_per_class = (cm.sum(axis=1) + cm.sum(axis=0)) - 2 * np.diag(cm)
    # Top N clases más conflictivas. Si hay empate o todas son perfectas, rellena con las primeras.
    top_n = min(top_n, len(labels))
    if errors_per_class.sum() == 0:
        # Modelo perfecto: muestra las primeras N clases igualmente para tener algo
        top_idxs = list(range(top_n))
    else:
        top_idxs = np.argsort(errors_per_class)[::-1][:top_n].tolist()
        top_idxs.sort()  # ordenar por índice para que la matriz se lea consistente

    sub_cm     = cm[np.ix_(top_idxs, top_idxs)]
    sub_labels = [labels[i] for i in top_idxs]

    # Normalizar por fila
    with np.errstate(divide='ignore', invalid='ignore'):
        sub_norm = sub_cm.astype(float) / sub_cm.sum(axis=1, keepdims=True)
        sub_norm = np.nan_to_num(sub_norm)

    n = len(sub_labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.85)),
                            facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')

    im = ax.imshow(sub_norm, cmap='viridis', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sub_labels, rotation=45, ha='right', color='#cccccc', fontsize=11)
    ax.set_yticklabels(sub_labels, color='#cccccc', fontsize=11)
    ax.set_xlabel('Predicción', color='#cccccc', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real',       color='#cccccc', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n} clases con más confusiones',
                 color='white', fontsize=14, fontweight='bold', pad=12)

    # Anotaciones (siempre, porque la matriz es pequeña)
    for i in range(n):
        for j in range(n):
            val   = sub_norm[i, j]
            count = sub_cm[i, j]
            if count == 0:
                continue
            txt_color = 'black' if val > 0.5 else 'white'
            ax.text(j, i, f'{val*100:.0f}%\n({count})',
                    ha='center', va='center', color=txt_color,
                    fontsize=10, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#cccccc', labelsize=9)
    plt.setp(cbar.ax.get_yticklabels(), color='#cccccc')
    cbar.outline.set_edgecolor('#333344')

    for spine in ax.spines.values():
        spine.set_color('#333344')

    fig.text(0.5, -0.02,
             f'Estas {n} clases concentran la mayoría de los errores del modelo. '
             f'Las {len(labels) - n} restantes se clasifican casi perfectamente.',
             ha='center', color='#888', fontsize=9, style='italic')

    plt.tight_layout()
    out = save_path or 'models/confusion_matrix.png'
    try:
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"    Top confusiones guardadas en: {out}")
    except Exception as e:
        print(f"  ⚠ No se pudo guardar top confusiones: {e}")
    plt.close('all')


def plot_top_errors_bar(model, X_val, y_val, labels, top_k=10, save_path=None):
    """
    Gráfico de barras horizontales con los K pares (Real → Predicción) más confundidos.
    Es una lectura narrativa de la matriz de confusión, mucho más legible
    en presentaciones que la matriz completa.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib no instalado.")
        return

    cm = _compute_confusion(model, X_val, y_val, len(labels))

    # Extraer todos los pares (i, j) con i != j y su conteo de errores
    errors = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                # cálculo del porcentaje sobre la clase real i
                total_i = cm[i, :].sum()
                pct = (cm[i, j] / total_i * 100) if total_i > 0 else 0
                errors.append((labels[i], labels[j], int(cm[i, j]), pct))

    if not errors:
        print("    ✓ El modelo no tiene errores en validación — gráfico vacío.")
        # Generar imagen con mensaje
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f1117')
        ax.set_facecolor('#1a1d27')
        ax.text(0.5, 0.5, '✓ El modelo clasificó correctamente todas las muestras\nde validación. Sin errores que mostrar.',
                ha='center', va='center', color='#38d9c0', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        out = save_path or 'models/top_errors.png'
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close('all')
        return

    # Ordenar por número de errores descendente y tomar top K
    errors.sort(key=lambda x: x[2], reverse=True)
    errors = errors[:top_k]

    real_labels = [f"{e[0]} → {e[1]}" for e in errors]
    counts      = [e[2] for e in errors]
    percents    = [e[3] for e in errors]

    # Invertir para que el de más errores quede arriba
    real_labels.reverse()
    counts.reverse()
    percents.reverse()

    n = len(errors)
    fig, ax = plt.subplots(figsize=(11, max(4, n * 0.5)), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')

    # Colorear según severidad (cantidad relativa al máximo)
    max_c   = max(counts)
    colors  = []
    for c in counts:
        ratio = c / max_c
        if ratio > 0.66:
            colors.append('#e05a5a')   # rojo: errores graves
        elif ratio > 0.33:
            colors.append('#f0c040')   # ámbar: medios
        else:
            colors.append('#a875e0')   # morado: leves

    bars = ax.barh(real_labels, counts, color=colors,
                   edgecolor='#0f1117', linewidth=0.8)

    # Anotar con conteo y porcentaje
    for bar, count, pct in zip(bars, counts, percents):
        w = bar.get_width()
        ax.text(w + max_c * 0.015, bar.get_y() + bar.get_height() / 2,
                f'{count} ({pct:.1f}%)',
                va='center', color='#cccccc', fontsize=10, fontweight='bold')

    ax.set_xlabel('Cantidad de errores', color='#cccccc', fontsize=11, fontweight='bold')
    ax.set_title(f'Top {n} confusiones más frecuentes (Real → Predicción)',
                 color='white', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors='#cccccc', labelsize=11)
    ax.grid(True, axis='x', color='#ffffff15', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_c * 1.18)
    for spine in ax.spines.values():
        spine.set_color('#333344')

    fig.text(0.5, -0.02,
             'Pares ordenados por frecuencia. El porcentaje indica qué fracción de la clase real fue mal clasificada.',
             ha='center', color='#888', fontsize=9, style='italic')

    plt.tight_layout()
    out = save_path or 'models/top_errors.png'
    try:
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"    Top errores guardado en: {out}")
    except Exception as e:
        print(f"  ⚠ No se pudo guardar top errores: {e}")
    plt.close('all')



def plot_class_metrics(model, X_val, y_val, labels, save_path=None):
    """
    Genera un gráfico de barras agrupadas con Precision / Recall / F1-Score por clase.
    Útil para identificar qué señas son débiles aunque la accuracy global sea alta.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        print("  ⚠ Falta matplotlib o sklearn.")
        return

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1) if y_val.ndim > 1 else y_val

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )

    n = len(labels)
    x = np.arange(n)
    w = 0.27

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 5.5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')

    bars1 = ax.bar(x - w, p, w, label='Precision', color='#38d9c0', edgecolor='#0f1117', linewidth=0.5)
    bars2 = ax.bar(x,     r, w, label='Recall',    color='#f0c040', edgecolor='#0f1117', linewidth=0.5)
    bars3 = ax.bar(x + w, f, w, label='F1-Score',  color='#a875e0', edgecolor='#0f1117', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#cccccc', fontsize=10, rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score', color='#cccccc', fontsize=11, fontweight='bold')
    ax.set_title('Métricas por clase: Precision · Recall · F1',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#cccccc', labelsize=9)
    ax.grid(True, axis='y', color='#ffffff15', linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#333344')

    leg = ax.legend(facecolor='#1a1d27', edgecolor='#333344', labelcolor='#cccccc',
                    fontsize=10, loc='lower right', ncol=3)

    # Línea horizontal en F1 promedio para referencia
    avg_f1 = float(np.mean(f))
    ax.axhline(avg_f1, color='#888', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.text(n - 0.5, avg_f1 + 0.015, f'F1 promedio: {avg_f1:.2f}',
            color='#888', fontsize=9, ha='right')

    plt.tight_layout()
    out = save_path or 'models/class_metrics.png'
    try:
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"    Métricas por clase guardadas en: {out}")
    except Exception as e:
        print(f"  ⚠ No se pudo guardar métricas por clase: {e}")
    plt.close('all')


def plot_class_distribution(X, y, labels, save_path=None):
    """
    Gráfico de barras con la cantidad de muestras por clase.
    Sirve para defender el balance del dataset.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib no instalado.")
        return

    y_idx  = np.argmax(y, axis=1) if y.ndim > 1 else y
    counts = np.bincount(y_idx, minlength=len(labels))
    n      = len(labels)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 5), facecolor='#0f1117')
    ax.set_facecolor('#1a1d27')

    mean   = counts.mean()
    colors = ['#e05a5a' if c < mean * 0.8 else '#38d9c0' for c in counts]
    bars   = ax.bar(range(n), counts, color=colors, edgecolor='#0f1117', linewidth=0.5)

    # Valor encima de cada barra
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.015, str(int(c)),
                ha='center', va='bottom', color='#cccccc', fontsize=9)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, color='#cccccc', fontsize=10)
    ax.set_ylabel('Cantidad de muestras', color='#cccccc', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribución de muestras por clase (Total: {int(counts.sum())})',
                 color='white', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors='#cccccc', labelsize=9)
    ax.grid(True, axis='y', color='#ffffff15', linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#333344')

    # Línea de promedio
    ax.axhline(mean, color='#f0c040', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(n - 0.5, mean + max(counts) * 0.02, f'Promedio: {mean:.0f}',
            color='#f0c040', fontsize=9, ha='right')

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='#38d9c0', label='Balanceado'),
        Patch(facecolor='#e05a5a', label='Sub-representado (<80% del promedio)'),
    ]
    ax.legend(handles=legend_items, facecolor='#1a1d27', edgecolor='#333344',
              labelcolor='#cccccc', fontsize=9, loc='upper right')

    plt.tight_layout()
    out = save_path or 'models/class_distribution.png'
    try:
        plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"    Distribución de clases guardada en: {out}")
    except Exception as e:
        print(f"  ⚠ No se pudo guardar distribución: {e}")
    plt.close('all')



class ModelTrainer:
    def __init__(self):
        self.storage = DataStorage()

    def _dataset_signature(self, labels, y):
        """Firma del dataset: cantidad de muestras por clase. Si cambia, hay que reentrenar."""
        return {lbl: int((y == i).sum()) for i, lbl in enumerate(labels)}

    def _load_signature(self):
        """Carga la firma del último entrenamiento. None si no existe."""
        import json
        path = "models/dataset_signature.json"
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_signature(self, sig):
        import json
        os.makedirs("models", exist_ok=True)
        with open("models/dataset_signature.json", "w", encoding="utf-8") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)

    def train(self, epochs: int = 50, extra_callbacks=None, force: bool = False):
        # ── 0. Verificar GPU ───────────────────────────────────────────────
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                print(f"  ✓ GPU detectada: {len(gpus)} dispositivo(s) → {[g.name for g in gpus]}")
                # Permitir crecimiento dinámico de memoria (evita reservar toda la VRAM)
                for g in gpus:
                    try: tf.config.experimental.set_memory_growth(g, True)
                    except Exception: pass
            else:
                print("  ⚠ No se detectó GPU. Entrenamiento en CPU (más lento).")
                print("    Para usar GPU instala: tensorflow-gpu + drivers CUDA/cuDNN.")
        except Exception as e:
            print(f"  ⚠ No se pudo verificar GPU: {e}")

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

        # ── 1.5. Detectar cambios y decidir estrategia ─────────────────────
        current_sig  = self._dataset_signature(labels, y)
        previous_sig = self._load_signature()
        model_path   = self.storage.get_model_path()
        model_exists = os.path.exists(model_path)

        same_classes  = previous_sig is not None and set(previous_sig.keys()) == set(current_sig.keys())
        identical_ds  = same_classes and previous_sig == current_sig
        only_more     = same_classes and not identical_ds and all(
            current_sig[k] >= previous_sig[k] for k in current_sig
        )

        strategy = "full"  # por defecto
        if not force:
            if identical_ds and model_exists:
                print("\n  ✓ El dataset no ha cambiado desde el último entrenamiento.")
                print("    No es necesario reentrenar. Usa force=True para forzar.")
                # Retorno mínimo: cargar modelo y devolver acc previa estimada
                from tensorflow import keras
                model = keras.models.load_model(model_path)
                _, acc = model.evaluate(X, y, verbose=0)
                # Crea un history "vacío" para que las gráficas existentes sigan funcionando
                class _FakeHistory:
                    def __init__(self): self.history = {"loss": [0.0], "val_loss": [0.0],
                                                         "accuracy": [acc], "val_accuracy": [acc]}
                print(f"    Accuracy actual: {acc*100:.1f}%\n")
                return _FakeHistory(), acc
            elif only_more and model_exists:
                strategy = "finetune"
                print("\n  ⚡ Mismo conjunto de clases con más muestras → fine-tuning")
                print(f"    Se reanuda desde el modelo anterior (entrenamiento mucho más rápido).")
            elif same_classes and model_exists:
                strategy = "full"
                print("\n  Las clases son las mismas pero el conteo cambió → entrenamiento completo.")
            else:
                strategy = "full"
                print("\n  Clases distintas al modelo anterior → entrenamiento desde cero.")

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
        if strategy == "finetune":
            from tensorflow import keras
            print("  Cargando modelo anterior para fine-tuning…")
            model = keras.models.load_model(model_path)
            # Recompilar con learning rate menor para fine-tuning
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            effective_epochs = max(8, min(15, epochs // 4))
            print(f"  Fine-tuning con learning rate reducido por {effective_epochs} épocas.")
        else:
            model = build_model(num_classes=len(labels))
            effective_epochs = epochs

        model.summary()

        callbacks = get_callbacks(model_path)

        print(f"\n  ⏳ Entrenando ({strategy})…\n")
        all_callbacks = callbacks + (extra_callbacks or [])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=effective_epochs,
            batch_size=64,    # ↑ de 32 a 64 → ~2x más rápido por época
            class_weight=cw_dict,
            callbacks=all_callbacks,
            verbose=1,
        )

        # Guardar firma para la próxima
        self._save_signature(current_sig)

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

        # ── 9. Top confusiones (matriz reducida) + Top errores (barras) ────
        plot_top_confusions_matrix(model, X_val, y_val, labels, top_n=6,
                                   save_path='models/confusion_matrix.png')
        plot_top_errors_bar(model, X_val, y_val, labels, top_k=10,
                            save_path='models/top_errors.png')

        # ── 10. Métricas por clase (precision/recall/F1) ───────────────────
        plot_class_metrics(model, X_val, y_val, labels,
                           save_path='models/class_metrics.png')

        # ── 11. Distribución de muestras por clase ─────────────────────────
        plot_class_distribution(X, y, labels,
                                save_path='models/class_distribution.png')

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