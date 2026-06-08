# LSM Training Pipeline

Pipeline de entrenamiento para **reconocimiento de Lengua de Señas Mexicana (LSM)**. El proyecto clasifica gestos a partir de landmarks de manos y pose corporal extraídos de videos, entrena modelos de deep learning en PyTorch y los exporta a Core ML para despliegue en iOS/macOS.

## Resumen

El flujo completo tiene dos fases:

**Fase 1 (sin entrenamiento):** huella estadística + DTW — ver [lsm_fingerprints](./lsm_fingerprints/)

**Fase 2 (red neuronal):**

1. **Organizar videos** según categorías y palabras definidas en `themes.json`
2. **Extraer landmarks** con LSMExtractorGUI → archivos `*_landmarks.json`
3. **Preprocesar y cargar** secuencias temporales normalizadas
4. **Entrenar** uno de tres modelos (1D CNN, TCN o 3D CNN)
5. **Exportar** el mejor checkpoint a Core ML (`.mlpackage`)

El TCN de Fase 2 debe superar el baseline de Fase 1 (~60% top-1 LOO) para validar el entrenamiento.

## Vocabulario y datos

`themes.json` define el vocabulario del proyecto:

| Categoría | Palabras |
|---|---|
| Alimentos y Bebidas | 40 |
| Animales e Insectos | 40 |
| Colores | 2 |
| Familia | 21 |
| Festividades | 20 |
| Frutas, Verduras y Plantas | 29 |
| Locaciones | 2 |
| Objetos | 17 |
| Personajes Históricos | 10 |
| Personas | 11 |
| Profesiones | 29 |

**Total: 221 palabras en 11 categorías.**

Cada entrada incluye un nombre de visualización y una ruta de video (`videoUrl`) que determina la carpeta destino al organizar el dataset.

### Formato de entrada (landmarks)

Los modelos consumen JSON generados por Vision Framework. Cada frame se convierte en un vector de **135 dimensiones**:

| Componente | Dimensiones | Descripción |
|---|---|---|
| Mano izquierda | 42 | 21 puntos × (x, y) |
| Mano derecha | 42 | 21 puntos × (x, y) |
| Pose corporal | 51 | 17 puntos × (x, y, visibility) |

Las secuencias se interpolan linealmente a **85 frames** (percentil 10 del dataset). Puntos no detectados se rellenan con `0.0`.

### Estructura del dataset

El loader (`dataset.py`) soporta dos layouts:

```
# Anidada (recomendada)
dataset/
└── Categoria/
    └── Palabra/
        └── video_landmarks.json

# Plana
dataset/
└── Palabra/
    └── video_landmarks.json
```

Solo se incluyen clases con al menos `--min-videos` secuencias válidas (por defecto: 2).

## Modelos disponibles

| Modelo | Flag en `train.py` | Descripción |
|---|---|---|
| **1D CNN** | `cnn` (default) | Convoluciones 1D sobre la dimensión temporal. Baseline rápido (~0.76M parámetros). |
| **TCN** | `tcn` | Red convolucional temporal con bloques residuales causales y dilatación exponencial. Campo receptivo: 61 frames. |
| **3D CNN** | `3dcnn` | Convoluciones 3D con atención temporal aprendida. Más compacto (~0.2M parámetros). |

Los tres modelos reciben tensores de forma `(batch, 85, 135)` y producen logits de clasificación.

También existe un paquete modular en `models/` con un registro de fábrica (`create_model`) usado por los smoke tests.

## Características actuales

### Preprocesamiento y augmentación

- Interpolación temporal a longitud fija
- Split train/val **antes** de augmentar (evita data leakage)
- Augmentación configurable (por defecto 10×) sobre el conjunto de entrenamiento:
  - Escala alrededor del centroide (0.9–1.1)
  - Traslación (±0.05)
  - Time warp / cambio de velocidad (0.8–1.2×)
  - Ruido gaussiano (σ = 0.01)

### Entrenamiento

- Optimizador AdamW con weight decay
- Scheduler coseno annealing
- Cross-entropy con label smoothing (0.1)
- Gradient clipping (max norm 1.0)
- Soporte para CUDA, Apple MPS y CPU
- Checkpoints: mejor modelo por `val_acc` + checkpoint cada 10 epochs
- Resume desde checkpoint interrumpido
- Métricas Top-1 y Top-5 al finalizar
- Gráficas de loss/accuracy (`training_curves.png`)

### Exportación móvil

- Conversión PyTorch → Core ML vía TorchScript
- Metadata embebida: lista de clases, dimensiones, accuracy de validación
- Target: iOS 16+

### Utilidades de datos

- **`organize_from_json.py`**: organiza videos `.mp4` en la estructura de carpetas correcta usando `themes.json`, con validación `ffprobe` (rechaza GIFs, codecs de imagen, archivos corruptos)
- **`list_words.py`**: lista o exporta el vocabulario completo

## Requisitos

- Python 3.11+
- PyTorch 2.1+
- `ffprobe` (FFmpeg) para organizar videos
- GPU recomendada para entrenamiento (CUDA o Apple Silicon MPS)

## Instalación

```bash
# Opción 1: script de setup (crea venv + instala PyTorch CUDA 12.4)
./setup_env.sh
source venv/bin/activate

# Opción 2: manual
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Con [direnv](https://direnv.net/), el proyecto usa `layout python3` automáticamente.

## Uso

### 1. Organizar videos

```bash
python organize_from_json.py \
  --json themes.json \
  --input ./videos \
  --output ./dataset

# Vista previa sin mover archivos
python organize_from_json.py --json themes.json --input ./videos --output ./dataset --dry-run
```

Archivos inválidos van a `_invalidos/`; archivos sin match a `_sin_match/`.

### 2. Listar vocabulario

```bash
python list_words.py --json themes.json
python list_words.py --json themes.json --categoria "Animales e Insectos"
python list_words.py --json themes.json --exportar palabras.txt
```

### 3. Verificar dataset

```bash
python dataset.py ./dataset
```

### 4. Entrenar

```bash
# 1D CNN (baseline)
python train.py --dataset ./dataset --epochs 100 --batch 64

# TCN
python train.py --model tcn --dataset ./dataset --epochs 120 --batch 16 --output ./runs_tcn

# 3D CNN
python train.py --model 3dcnn --dataset ./dataset --epochs 100 --batch 12 --output ./runs_3dcnn

# Reanudar entrenamiento
python train.py --dataset ./dataset --resume ./runs/checkpoint_epoch50.pt
```

#### Parámetros principales de `train.py`

| Parámetro | Default | Descripción |
|---|---|---|
| `--model` | `cnn` | Arquitectura: `cnn`, `tcn`, `3dcnn` |
| `--dataset` | — | Carpeta raíz del dataset (requerido) |
| `--output` | `./runs` | Carpeta de salida para checkpoints |
| `--epochs` | 100 | Épocas de entrenamiento |
| `--batch` | 64 | Tamaño de batch |
| `--lr` | 1e-3 | Learning rate |
| `--dropout` | 0.3 | Tasa de dropout |
| `--val_split` | 0.15 | Fracción de validación |
| `--augment` | 10 | Versiones augmentadas por video |
| `--min_videos` | 2 | Mínimo de videos por clase |
| `--resume` | — | Checkpoint para continuar |

Hiperparámetros de referencia adicionales en `configs/` (`cnn1d_config.yaml`, `tcn_config.yaml`, `cnn3d_config.yaml`).

### 5. Exportar a Core ML

```bash
python convert_to_coreml.py \
  --checkpoint ./runs/best_model.pt \
  --output ./lsm_model
```

Genera `lsm_model.mlpackage` listo para integrar en un proyecto Xcode.

### 6. Smoke tests de modelos

```bash
python test_models_smoke.py
```

Verifica que las tres arquitecturas del paquete `models/` producen salidas con la forma correcta.

## Estructura del proyecto

```
TrainingPipeline/
├── train.py                 # Script principal de entrenamiento
├── dataset.py               # Carga, interpolación y augmentación
├── model.py                 # 1D CNN (usado por train.py)
├── model_tcn.py             # TCN (usado por train.py)
├── model_3dcnn.py           # 3D CNN (usado por train.py)
├── convert_to_coreml.py     # Exportación PyTorch → Core ML
├── organize_from_json.py    # Organización de videos
├── list_words.py            # Listado del vocabulario
├── test_models_smoke.py     # Tests de forma de los modelos
├── themes.json              # Vocabulario y categorías LSM
├── palabras.txt             # Exportación del vocabulario
├── requirements.txt         # Dependencias Python
├── setup_env.sh             # Script de instalación
├── configs/                 # Hiperparámetros de referencia (YAML)
│   ├── cnn1d_config.yaml
│   ├── tcn_config.yaml
│   └── cnn3d_config.yaml
├── models/                  # Implementaciones modulares + factory
│   ├── __init__.py
│   ├── cnn1d.py
│   ├── tcn.py
│   └── cnn3d.py
├── runs/                    # Salida de entrenamiento (checkpoints, classes.json)
├── runs_3dcnn_test/         # Runs de prueba 3D CNN
├── coreml_models.mlpackage/ # Modelo Core ML exportado
├── LSM_Fase1_Architecture.md  # Especificación Fase 1
└── lsm_fingerprints/        # Fase 1: huella estadística + DTW (sin entrenamiento)
    ├── build_db.py
    ├── evaluate.py
    ├── fingerprint.py
    ├── matcher.py
    └── tests/
```

## Salidas de entrenamiento

Cada run genera en la carpeta `--output`:

| Archivo | Descripción |
|---|---|
| `best_model.pt` | Mejor checkpoint por accuracy de validación |
| `checkpoint_epoch{N}.pt` | Checkpoint periódico cada 10 epochs |
| `classes.json` | Lista de clases entrenadas (necesaria para inferencia) |
| `training_curves.png` | Gráficas de loss y accuracy |

El checkpoint incluye: pesos del modelo, optimizer, scheduler, historial, tipo de modelo, `n_frames`, `feature_dim` y `classes`.

## Estado actual

- Vocabulario definido: **221 palabras** en 11 categorías temáticas
- Último entrenamiento documentado en `runs/`: **187 clases** (clases con suficientes videos de entrenamiento)
- Tres arquitecturas implementadas y seleccionables desde `train.py`
- Exportación a Core ML funcional para despliegue en dispositivos Apple
- Extracción de landmarks: [AppLSMTests/LSMExtractorGUI](../AppLSMTests/LSMExtractorGUI/)
- Baseline Fase 1: [lsm_fingerprints](./lsm_fingerprints/) — `python evaluate.py --db fingerprints.npz`
- Inferencia móvil: [AppLSMTests/LSMMobileModelTesting](../AppLSMTests/LSMMobileModelTesting/)

## Dependencias principales

```
torch, torchvision, numpy, scipy, matplotlib,
coremltools, pyyaml, tensorboard, scikit-learn, pandas
```
