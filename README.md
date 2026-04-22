# 🖐️ SignLens — Traductor de Lenguaje de Señas en Tiempo Real

Sistema de reconocimiento de señas estáticas en tiempo real usando una **CNN (Conv1D)**
entrenada con tus propios gestos mediante webcam. Detecta **mano izquierda y mano derecha**
de forma independiente gracias a **MediaPipe Hands**.

---

## 📋 Arquitectura

```
Webcam → MediaPipe Hands → Landmarks (126 features)
                                  ↓
                         CNN (Conv1D + Dense)
                                  ↓
                      Predicción con suavizado temporal
                                  ↓
                         Texto en pantalla en tiempo real
```

**Vector de entrada (126 valores):**
- `[0:63]`   → 21 landmarks × (x, y, z) — mano **izquierda**
- `[63:126]` → 21 landmarks × (x, y, z) — mano **derecha**
- Normalizados: centrado en muñeca + escala por extensión máxima

**Red neuronal:**
```
Input(126) → Reshape(21,6) → Conv1D(64) → Conv1D(128) → Conv1D(256)
           → GlobalAvgPool → Dense(256) → Dropout → Dense(128) → Dropout
           → Dense(num_clases, softmax)
```

---

## ⚙️ Instalación

```bash
pip install -r requirements.txt
```

> **Nota Python:** Requiere Python 3.9–3.11. TensorFlow no soporta Python 3.12+ aún.

---

## 🚀 Uso paso a paso

### 1. Recolectar datos para cada seña

```bash
python main.py --collect HOLA
python main.py --collect GRACIAS
python main.py --collect SI
python main.py --collect NO
python main.py --collect POR_FAVOR
```

**Controles durante recolección:**
| Tecla | Acción |
|-------|--------|
| `E`   | Empezar / pausar captura |
| `Q`   | Salir |

**Consejos:**
- Captura **200+ muestras** por seña para mejores resultados
- Varía ligeramente el ángulo y posición de la mano
- Iluminación uniforme
- Puedes usar **una sola mano** o **ambas** según la seña

### 2. Ver las señas registradas

```bash
python main.py --list
```

Salida ejemplo:
```
📚 Palabras registradas (4):
   ✦ GRACIAS             215 muestras
   ✦ HOLA                200 muestras
   ✦ NO                  198 muestras
   ✦ SI                  202 muestras
```

### 3. Entrenar el modelo

```bash
python main.py --train
# o con más épocas:
python main.py --train --epochs 80
```

El entrenamiento incluye:
- **Aumentación de datos** (ruido + escala + volteo)
- **Early stopping** automático
- **Reducción de learning rate** al estancarse
- Guardado del **mejor modelo** por val_accuracy

### 4. Traducción en tiempo real

```bash
python main.py
```

**Controles durante traducción:**
| Tecla | Acción |
|-------|--------|
| `C`   | Limpiar historial de palabras |
| `Q`   | Salir |

### 5. Agregar nuevas señas (sin borrar las anteriores)

```bash
python main.py --collect NUEVA_SEÑA
python main.py --train    # re-entrena con todas las señas
```

### 6. Eliminar una seña

```bash
python main.py --delete HOLA
python main.py --train    # re-entrena sin esa seña
```

---

## 📊 Parámetros configurables

```bash
python main.py --collect HOLA --samples 300  # más muestras
python main.py --train --epochs 100           # más épocas
python main.py --camera 1                     # segunda cámara
```

---

## 📁 Estructura del proyecto

```
sign_language_translator/
│
├── main.py                  # Punto de entrada (CLI)
│
├── app/
│   ├── collector.py         # Recolección de muestras por webcam
│   ├── trainer.py           # Entrenamiento CNN
│   └── translator.py        # Inferencia en tiempo real
│
├── utils/
│   ├── hands.py             # MediaPipe: extracción y normalización de landmarks
│   ├── model.py             # Arquitectura CNN (Conv1D)
│   └── storage.py           # Gestión de archivos (datos, modelo, etiquetas)
│
├── data/                    # (auto-generado) Muestras .npy por seña
│   ├── HOLA/
│   │   ├── 00000.npy
│   │   └── ...
│   └── GRACIAS/
│
├── models/                  # (auto-generado)
│   ├── sign_model.keras     # Modelo entrenado
│   └── labels.json          # Lista de señas
│
└── requirements.txt
```

---

## 🔧 Ajustes avanzados (en translator.py)

```python
CONFIDENCE_THRESHOLD = 0.70   # mínimo para mostrar predicción (0.0–1.0)
SMOOTHING_WINDOW     = 10     # frames para suavizado (más = más lento pero estable)
MIN_STABLE_FRAMES    = 6      # frames seguidos para confirmar una palabra
HISTORY_MAX          = 8      # palabras visibles en el historial
```

---

## ❓ Problemas frecuentes

| Problema | Solución |
|----------|----------|
| `No se pudo abrir la cámara 0` | Prueba `--camera 1` o `--camera 2` |
| Precisión baja | Recolecta más muestras (300+), mejora iluminación |
| "Necesitas al menos 2 palabras" | Recolecta datos para 2+ señas antes de entrenar |
| Predicciones inestables | Aumenta `SMOOTHING_WINDOW` y `MIN_STABLE_FRAMES` |
| TensorFlow no instala | Usa Python 3.10 o 3.11, no 3.12 |
