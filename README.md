# LSM Training Pipeline — Multi-Model (CNN, TCN, 3D CNN)

Pipeline completo para reconocimiento de **Lengua de Señas Mexicana (LSM)** con soporte para tres arquitecturas de redes neuronales entrenadas sobre landmarks de manos y pose extraídos con Vision Framework (iOS/macOS):
- **1D CNN** (baseline rápido)
- **TCN** (Temporal Convolutional Network — temporal receptivo más amplio)
- **3D CNN** (convolución espaciotemporal directa)

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
# Entrenamiento con 1D CNN (default)
python train.py \
    --dataset ./SalidasLSMSkeletization \
    --output  ./runs

# Entrenar TCN (Temporal Convolutional Network)
python train.py \
    --dataset ./SalidasLSMSkeletization \
    --output  ./runs_tcn \
    --model   tcn

# Entrenar 3D CNN
python train.py \
    --dataset ./SalidasLSMSkeletization \
    --output  ./runs_3dcnn \
    --model   3dcnn

# Con todos los parámetros
python train.py \
    --dataset    ./SalidasLSMSkeletization \
    --output     ./runs \
    --model      cnn \
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
    --model   tcn \
    --resume  ./runs_tcn/checkpoint_epoch50.pt
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

# Convertir modelo 1D CNN
python convert_to_coreml.py \
    --checkpoint ./runs/best_model.pt \
    --output     ./lsm_model_cnn

# Convertir modelo TCN
python convert_to_coreml.py \
    --checkpoint ./runs_tcn/best_model.pt \
    --output     ./lsm_model_tcn

# Convertir modelo 3D CNN
python convert_to_coreml.py \
    --checkpoint ./runs_3dcnn/best_model.pt \
    --output     ./lsm_model_3dcnn
```

Genera `lsm_model*.mlpackage` listo para arrastrar al proyecto Xcode.

**Nota:** El script detecta automáticamente la arquitectura (`model_type`) desde el checkpoint, por lo que no necesita parámetro adicional.

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

### 1. CNN 1D (Baseline)

**`LSM_CNN`** — Red convolucional 1D para clasificación de secuencias de landmarks.

```
Input  : (batch, 85, 135)
         └── permute ──→ (batch, 135, 85)

Block 1: Conv1d(135→128, k=3) + BatchNorm + ReLU + Dropout + MaxPool(2)
Block 2: Conv1d(128→256, k=3) + BatchNorm + ReLU + Dropout + MaxPool(2)
Block 3: Conv1d(256→512, k=3) + BatchNorm + ReLU + Dropout

Global Average Pooling → (batch, 512)
FC(512 → 256) + ReLU + Dropout
FC(256 → n_classes)

Output : logits (batch, n_classes)
```

**Características:**
- Parámetros: ~690K
- Receptive field: N/A (standard Conv1d)
- Velocidad: muy rápida (~15-20ms/sample)
- Mejor para: inferencia en tiempo real, dispositivos con recursos limitados

### 2. TCN (Temporal Convolutional Network)

**`LSM_TCN`** — Red convolucional temporal con dilación exponencial para mayor receptive field.

```
Input  : (batch, 85, 135)
         └── permute ──→ (batch, 135, 85)

Proyección: Conv1d(135→64, k=1)

Block 1: ResidualBlock(64→64, dilation=1, kernel=5)
Block 2: ResidualBlock(64→128, dilation=2, kernel=5)
Block 3: ResidualBlock(128→256, dilation=4, kernel=5)
Block 4: ResidualBlock(256→256, dilation=8, kernel=5)
         └─ Convoluciones causales (padding izquierdo)

Global Average Pooling → (batch, 256)
FC(256 → 128) + BatchNorm + ReLU + Dropout
FC(128 → n_classes)

Output : logits (batch, n_classes)
```

**Características:**
- Parámetros: ~1.5M
- Receptive field: 61 frames (71.8% de 85)
- Velocidad: moderada (~100-150ms/sample)
- Mejor para: capturar dependencias temporales largas, patrones complejos

### 3. 3D CNN

**`LSM_3DCNN`** — Red convolucional 3D con atención temporal para procesamiento espaciotemporal directo.

```
Input  : (batch, 85, 135)
         └── view ──→ (batch, 1, 85, 135, 1)

Block 1: Conv3d(1→32, k=3×3×1, stride=1×1×1) + BatchNorm3d + ReLU + Dropout3d
Block 2: Conv3d(32→64, k=3×3×1, stride=2×1×1) + BatchNorm3d + ReLU + Dropout3d  (downsample tiempo)
Block 3: Conv3d(64→128, k=3×3×1, stride=2×1×1) + BatchNorm3d + ReLU + Dropout3d (downsample tiempo)

Temporal Attention: Conv3d(128→32→1) + Sigmoid
                   └─ Escala cada posición temporal

Global Average Pooling → (batch, 128)
FC(128 → 256) + BatchNorm + ReLU + Dropout
FC(256 → n_classes)

Output : logits (batch, n_classes)
```

**Características:**
- Parámetros: ~144K (el más ligero)
- Receptive field: ~11 frames (temporal), todas las features (spatial)
- Velocidad: moderada (~80-90ms/sample)
- Mejor para: modelos compactos, despliegue en mobile, balance eficiencia-precisión

### Configuración de entrenamiento

- **Optimizador:** AdamW, weight decay 1e-4
- **LR scheduler:** CosineAnnealingLR (eta_min = 1e-6)
- **Loss:** CrossEntropyLoss con label_smoothing = 0.1
- **Gradient clipping:** max_norm = 1.0
- **Dispositivo:** GPU (CUDA/MPS) o CPU (fallback automático)

### Comparativa rápida

| Modelo | Parámetros | Velocidad | Receptive Field | Mejor para |
|---|---|---|---|---|
| **CNN 1D** | ~690K | ⚡⚡⚡ Muy rápido | Standard | Producción mobile |
| **TCN** | ~1.5M | ⚡⚡ Rápido | 61 frames | Patrones temporales |
| **3D CNN** | ~144K | ⚡⚡ Rápido | ~11 frames | Modelos compactos |

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
| `--model` | `cnn` | Arquitectura: `cnn` \| `tcn` \| `3dcnn` |
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
    "epoch":        int,         # Epoch en que se guardó (0-indexed)
    "model":        dict,        # state_dict del modelo
    "optimizer":    dict,        # state_dict del optimizador
    "scheduler":    dict,        # state_dict del scheduler
    "best_val_acc": float,       # Mejor val_acc hasta ese momento
    "history":      dict,        # train_loss, train_acc, val_loss, val_acc por época
    "classes":      list[str],   # Lista de clases (orden = índice de logits)
    "n_frames":     int,         # 85
    "feature_dim":  int,         # 135
    "model_type":   str,         # "cnn" | "tcn" | "3dcnn"
}
```

El campo `model_type` permite a `convert_to_coreml.py` reconstruir automáticamente la arquitectura correcta.

### `classes.json`

Lista JSON con los nombres de clase en orden alfabético. El índice en la lista corresponde al índice de logit que produce el modelo:

```json
["Abuela", "Agua", "Amor", ...]
```

Este archivo es necesario para interpretar las predicciones del modelo Core ML en la app iOS.
