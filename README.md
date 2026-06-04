# Sign.Bridge
Traductor de Lengua de Señas Colombiana (LSC) en tiempo real usando CNN y MediaPipe.

---

## Requisitos del sistema

- Windows 10 / 11
- Miniconda (ver instalación abajo)
- Cámara web

---

## Instalación

### 1. Instalar Miniconda

Descargar el instalador desde:
```
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
```

Durante la instalación marcar la opción:
```
Add Miniconda3 to my PATH environment variable
```

Reiniciar el equipo al terminar.

### 2. Habilitar scripts en PowerShell

Abrir una terminal y ejecutar:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Confirmar con `S` cuando pregunte.

### 3. Inicializar conda en PowerShell

```bash
conda init powershell
```

Cerrar la terminal y abrir una nueva. Debe aparecer `(base)` al inicio del prompt.

### 4. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

### 5. Crear el entorno

```bash
conda create -n signbridge python=3.10.11 -y
```

### 6. Activar el entorno

```bash
conda activate signbridge
```

Debe aparecer `(signbridge)` al inicio del prompt.

### 7. Instalar dependencias

```bash
pip install -r requirements.txt
```

Este proceso puede tardar entre 5 y 15 minutos dependiendo de la conexión.

---

## Uso

Cada vez que abras una terminal nueva debes activar el entorno primero:
```bash
conda activate signbridge
```

### Recolectar muestras de una seña

```bash
python main.py --collect A
```

Controles durante la recolección:
- `E` — iniciar / pausar captura
- `Q` — salir

Se recomienda un mínimo de 200 muestras por seña. Variar ligeramente el ángulo y posición de la mano entre capturas.

### Ver señas registradas

```bash
python main.py --list
```

### Entrenar el modelo

Se requieren mínimo 2 señas recolectadas antes de entrenar.

```bash
python main.py --train
```

Opciones adicionales:
```bash
python main.py --train --epochs 80
```

### Ejecutar el traductor en tiempo real

```bash
python main.py
```

Si tienes más de una cámara:
```bash
python main.py --camera 1
```

### Eliminar una seña

```bash
python main.py --delete A
```

---

## Estructura del proyecto

```
Sign.Bridge/
├── main.py                  # Punto de entrada y CLI
├── requirements.txt         # Dependencias Python
│
├── app/
│   ├── collector.py         # Recolección de muestras por webcam
│   ├── trainer.py           # Entrenamiento del modelo CNN
│   └── translator.py        # Traducción en tiempo real
│
├── utils/
│   ├── hands.py             # Detección de manos con MediaPipe Tasks
│   ├── model.py             # Arquitectura CNN (Conv1D)
│   ├── storage.py           # Gestión de archivos y modelo
│   └── hand_landmarker.task # Modelo MediaPipe (se descarga automáticamente)
│
├── data/                    # Muestras recolectadas por seña (no se sube al repo)
└── models/                  # Modelo entrenado (no se sube al repo)
    ├── sign_model.h5
    └── labels.json
```

---

## Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| TensorFlow | 2.13.0 | Entrenamiento e inferencia |
| MediaPipe | >= 0.10.9 | Detección de landmarks de manos |
| OpenCV | 4.8.1.78 | Captura de video y visualización |
| scikit-learn | 1.3.2 | Aumentación y métricas |
| FastAPI | 0.103.2 | Servidor web (opcional) |
| pyttsx3 | 2.90 | Síntesis de voz offline |

---

## Notas importantes

**Modelo MediaPipe:** La primera vez que se ejecuta el proyecto se descarga automáticamente el archivo `hand_landmarker.task` (~8 MB) en la carpeta `utils/`. Este archivo no se sube al repositorio.

**Datos y modelo entrenado:** Las carpetas `data/` y `models/` están excluidas del repositorio. Cada equipo debe recolectar sus propias muestras y entrenar, o compartir estas carpetas por otro medio (OneDrive, Google Drive, etc.).

**Mensajes en consola:** Los mensajes `W0000`, `E0000` e `INFO` que aparecen al ejecutar son advertencias internas de MediaPipe y TensorFlow, no afectan el funcionamiento del sistema.

**Indicador de estabilidad:** En la ventana del traductor, la barra de estabilidad muestra `###---` donde `#` representa los frames consecutivos acumulados y `-` los que faltan para confirmar la seña. Al llegar a `######` la seña se confirma.

**Compatibilidad:** El proyecto funciona con Python 3.10.x y 3.11.x. Se recomienda usar el entorno Conda especificado para garantizar compatibilidad entre equipos.

---

## Flujo recomendado para un equipo nuevo

```bash
# 1. Instalar Miniconda y reiniciar el equipo
# 2. Abrir terminal

conda init powershell
# Cerrar y abrir terminal nueva

conda create -n signbridge python=3.10.11 -y
conda activate signbridge
pip install -r requirements.txt

# 3. Recolectar datos (repetir para cada seña)
python main.py --collect A
python main.py --collect B

# 4. Entrenar
python main.py --train

# 5. Traducir
python main.py
```
