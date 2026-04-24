# LSM Training Pipeline — 1D CNN

Pipeline completo para reconocimiento de **Lengua de Señas Mexicana (LSM)** usando una red neuronal convolucional 1D entrenada sobre landmarks de manos y pose extraídos con Vision Framework (iOS/macOS).

```
videos .mp4  →  organize  →  skeletization (iOS)  →  train  →  Core ML .mlpackage
```

---

## Contenido

- [Requisitos](#requisitos)
- [Setup](#setup)
- [Pipeline completo](#pipeline-completo)
  - [1. Organizar videos](#1-organizar-videos)
  - [2. Extraer landmarks](#2-extraer-landmarks)
  - [3. Entrenamiento](#3-entrenamiento)
  - [4. Conversión a Core ML](#4-conversión-a-core-ml)
- [Scripts de utilidad](#scripts-de-utilidad)
- [Formato de datos](#formato-de-datos)
- [Arquitectura del modelo](#arquitectura-del-modelo)
- [Augmentation](#augmentation)
- [Parámetros de entrenamiento](#parámetros-de-entrenamiento)
- [Outputs del entrenamiento](#outputs-del-entrenamiento)

---

## Requisitos

- Python 3.9+
- CUDA (opcional, también soporta Apple MPS y CPU)
- `ffprobe` / `ffmpeg` (para validación de videos en `organize_from_json.py`)
- Xcode con Vision Framework (para la etapa de extracción de landmarks, fuera de este repo)

---

## Setup

```bash
python -m venv ~/lsm_env
source ~/lsm_env/bin/activate
pip install -r requirements.txt

# Con soporte CUDA (opcional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**`requirements.txt`:**
```
torch>=2.0.0
torchvision
numpy
scipy
matplotlib
coremltools
```

---

## Pipeline completo

### 1. Organizar videos

Antes de extraer landmarks, los videos crudos deben organizarse en la estructura de carpetas que espera la app iOS de skeletization.

**Script:** `organize_from_json.py`

Lee `themes.json` como fuente de verdad, valida cada `.mp4` con `ffprobe` y los mueve a:

```
output/
└── Categoria/
    └── Nombre Palabra/
        └── video.mp4
```

Archivos problemáticos se separan automáticamente:
- `_invalidos/` — GIFs renombrados, encoding corrupto, duración < 0.5 s
- `_sin_match/` — archivos sin correspondencia en el JSON

```bash
# Ver qué haría sin mover nada
python3 organize_from_json.py \
    --json   themes.json \
    --input  ./videos \
    --output ./dataset \
    --dry-run

# Ejecución real
python3 organize_from_json.py \
    --json   themes.json \
    --input  ./videos \
    --output ./dataset \
    --verbose
```

| Parámetro | Descripción |
|---|---|
| `--json` | Ruta al `themes.json` |
| `--input` | Carpeta con los `.mp4` planos |
| `--output` | Carpeta destino del dataset organizado |
| `--dry-run` | Simula sin mover archivos |
| `--verbose` | Muestra cada archivo procesado |

---

### 2. Extraer landmarks

Esta etapa se realiza en la app iOS/macOS usando Vision Framework (fuera de este repo). La app procesa cada video y genera un archivo `*_landmarks.json` por video con los keypoints de manos y pose por frame.

El pipeline de entrenamiento soporta dos estructuras de salida:

**Estructura plana** (un nivel de carpetas):
```
SalidasLSMSkeletization/
└── palabra/
    └── video_landmarks.json
```

**Estructura anidada** (dos niveles):
```
SalidasLSMSkeletization/
└── Categoria/
    └── palabra/
        └── video_landmarks.json
```

La detección es automática — `dataset.py` comprueba si las subcarpetas de primer nivel contienen JSON directamente o subcarpetas.

---

### 3. Entrenamiento

```bash
# Entrenamiento básico
python train.py \
    --dataset ./SalidasLSMSkeletization \
    --output  ./runs

# Con todos los parámetros
python train.py \
    --dataset    ./SalidasLSMSkeletization \
    --output     ./runs \
    --epochs     150 \
    --batch      64 \
    --lr         1e-3 \
    --dropout    0.3 \
    --val_split  0.15 \
    --augment    10 \
    --min_videos 2

# Continuar desde checkpoint
python train.py \
    --dataset ./SalidasLSMSkeletization \
    --output  ./runs \
    --resume  ./runs/checkpoint_epoch50.pt
```

**Flujo interno:**
1. Carga todos los `*_landmarks.json` con `load_raw_dataset()`
2. Excluye clases con menos de `--min_videos` videos
3. Hace el split train/val **antes** del augmentation (evita data leakage)
4. Aplica `augment_factor` versiones aumentadas solo al conjunto de train
5. Entrena con AdamW + cosine annealing, cross-entropy con label smoothing 0.1
6. Guarda `best_model.pt` en cada mejora de `val_acc`
7. Guarda checkpoint periódico cada 10 épocas
8. Al finalizar, calcula Top-1 y Top-5 accuracy y genera curvas de entrenamiento

---

### 4. Conversión a Core ML

```bash
pip install coremltools

python convert_to_coreml.py \
    --checkpoint ./runs/best_model.pt \
    --output     ./lsm_model
```

Genera `lsm_model.mlpackage` listo para arrastrar al proyecto Xcode.

**Metadata embebida en el modelo:**

| Campo | Descripción |
|---|---|
| `classes` | JSON con la lista de clases (mismo orden que los logits) |
| `n_frames` | Longitud de secuencia esperada (85) |
| `feature_dim` | Dimensión del vector por frame (135) |
| `n_classes` | Número de clases |
| `val_acc` | Accuracy de validación del checkpoint |

**Input/output del modelo Core ML:**

| Nombre | Shape | Tipo |
|---|---|---|
| `landmarks` | `(1, 85, 135)` | `float32` |
| `logits` | `(1, n_classes)` | `float32` |

Los logits son sin normalizar — aplicar softmax en el lado iOS para obtener probabilidades.

**Verificación rápida del modelo guardado:**
```bash
python testCoreml.py
```

---

## Scripts de utilidad

### `list_words.py` — Explorar el catálogo de palabras

```bash
# Listar todas las palabras
python3 list_words.py --json themes.json

# Filtrar por categoría
python3 list_words.py --json themes.json --categoria "Animales e Insectos"

# Exportar a archivo de texto
python3 list_words.py --json themes.json --exportar palabras.txt
```

### `testCoreml.py` — Verificar modelo Core ML

```bash
python testCoreml.py
```

Carga `./coreml_models.mlpackage` e imprime los metadatos embebidos.

---

## Formato de datos

### `themes.json`

Catálogo de palabras organizado por categorías. Estructura esperada:

```json
[
  {
    "name": "Categoria",
    "words": [
      {
        "name":     "Nombre para mostrar",
        "videoUrl": "https://.../.../carpeta_destino"
      }
    ]
  }
]
```

El último segmento de `videoUrl` es el nombre de la carpeta destino en el dataset.

### `*_landmarks.json` (salida de Vision Framework)

Cada archivo representa un video. Puede ser:
- Un array JSON de frames directamente: `[frame, frame, ...]`
- Un objeto con clave `"frames"`: `{"frames": [frame, frame, ...]}`

Cada frame tiene la estructura:

```json
{
  "leftHand":  [ {"x": 0.5, "y": 0.3}, ... ],
  "rightHand": [ {"x": 0.6, "y": 0.4}, ... ],
  "pose": [
    {"id": 0, "x": 0.5, "y": 0.2, "visibility": 0.99},
    ...
  ]
}
```

- `leftHand` / `rightHand`: 21 puntos `(x, y)` en coordenadas normalizadas [0, 1]
- `pose`: hasta 17 keypoints con `id`, `x`, `y`, `visibility`
- Puntos no detectados → ausentes del JSON (se rellenan con 0.0)
- Nombres alternativos soportados: `left_hand`, `right_hand`

### Vector de features (135 valores / frame)

```
[ 0: 42]  left_hand  — 21 puntos × (x, y)
[42: 84]  right_hand — 21 puntos × (x, y)
[84:135]  pose       — 17 puntos × (x, y, visibility)
```

Los videos se normalizan a **85 frames** mediante interpolación lineal (`N_FRAMES = 85`, percentil 10 del dataset).

---

## Arquitectura del modelo

**`LSM_CNN`** — Red convolucional 1D para clasificación de secuencias de landmarks.

```
Input  : (batch, 85, 135)
         └── permute ──→ (batch, 135, 85)

Block 1: Conv1d(135→128, k=3) + BatchNorm + ReLU + Dropout + MaxPool(2)
         └── (batch, 128, 42)

Block 2: Conv1d(128→256, k=3) + BatchNorm + ReLU + Dropout + MaxPool(2)
         └── (batch, 256, 21)

Block 3: Conv1d(256→512, k=3) + BatchNorm + ReLU + Dropout
         └── (batch, 512, 21)

Global Average Pooling → (batch, 512, 1)

FC(512 → 256) + ReLU + Dropout
FC(256 → n_classes)

Output : logits (batch, n_classes)
```

- **Optimizador:** AdamW, weight decay 1e-4
- **LR scheduler:** CosineAnnealingLR (eta_min = 1e-6)
- **Loss:** CrossEntropyLoss con label_smoothing = 0.1
- **Gradient clipping:** max_norm = 1.0

---

## Augmentation

Se aplica **solo al conjunto de train**, después del split, para evitar data leakage.

Cada video original genera `--augment` (default: 10) versiones adicionales con las siguientes transformaciones aleatorias:

| Transformación | Probabilidad | Rango |
|---|---|---|
| Escala alrededor del centroide | 80% | 0.9× – 1.1× |
| Traslación (x, y) | 80% | ±0.05 |
| Time warp (cambio de velocidad) | 70% | 0.8× – 1.2× |
| Ruido gaussiano | 60% | σ = 0.01 |

Las transformaciones solo modifican coordenadas `x, y`. Los valores de `visibility` de pose no se alteran. Los puntos no detectados (valor 0.0) no reciben augmentation.

---

## Parámetros de entrenamiento

| Parámetro | Default | Descripción |
|---|---|---|
| `--dataset` | _(requerido)_ | Carpeta raíz del dataset con los JSON de landmarks |
| `--output` | `./runs` | Carpeta de salida para checkpoints y curvas |
| `--epochs` | `100` | Número de épocas |
| `--batch` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate inicial (decae con cosine annealing) |
| `--dropout` | `0.3` | Dropout en capas Conv y FC |
| `--val_split` | `0.15` | Fracción de videos originales para validación |
| `--augment` | `10` | Versiones augmentadas por video de train |
| `--min_videos` | `2` | Mínimo de videos por clase para incluirla |
| `--resume` | `None` | Checkpoint `.pt` desde el que continuar |

---

## Outputs del entrenamiento

```
runs/
├── best_model.pt          # Mejor checkpoint según val_acc (Top-1)
├── checkpoint_epoch10.pt  # Checkpoints periódicos cada 10 épocas
├── checkpoint_epoch20.pt
├── ...
├── classes.json           # Lista de clases ordenada alfabéticamente
└── training_curves.png    # Curvas de loss y accuracy (train vs val)
```

### Contenido de un checkpoint `.pt`

```python
{
    "epoch":        int,       # Epoch en que se guardó (0-indexed)
    "model":        dict,      # state_dict del modelo
    "optimizer":    dict,      # state_dict del optimizador
    "scheduler":    dict,      # state_dict del scheduler
    "best_val_acc": float,     # Mejor val_acc hasta ese momento
    "history":      dict,      # train_loss, train_acc, val_loss, val_acc por época
    "classes":      list[str], # Lista de clases (orden = índice de logits)
    "n_frames":     int,       # 85
    "feature_dim":  int,       # 135
}
```

### `classes.json`

Lista JSON con los nombres de clase en orden alfabético. El índice en la lista corresponde al índice de logit que produce el modelo:

```json
["Abuela", "Agua", "Amor", ...]
```

Este archivo es necesario para interpretar las predicciones del modelo Core ML en la app iOS.
