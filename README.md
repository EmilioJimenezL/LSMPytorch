# LSM Training Pipeline — 1D CNN

## Archivos

| Archivo | Descripción |
|---|---|
| `dataset.py` | Carga JSON de Vision Framework, interpolación temporal, augmentation |
| `model.py` | Arquitectura 1D CNN |
| `train.py` | Loop de entrenamiento, checkpoints, curvas |
| `requirements.txt` | Dependencias Python |

## Setup

```bash
python -m venv ~/lsm_env
source ~/lsm_env/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Entrenamiento

```bash
# Primer entrenamiento
python train.py --dataset ./SalidasLSMSkeletization --output ./runs

# Con parámetros personalizados
python train.py \
    --dataset ./SalidasLSMSkeletization \
    --output  ./runs \
    --epochs  150 \
    --batch   64 \
    --lr      1e-3 \
    --augment 10

# Continuar desde checkpoint
python train.py --dataset ./SalidasLSMSkeletization --resume ./runs/checkpoint_epoch50.pt
```

## Parámetros clave

| Parámetro | Default | Descripción |
|---|---|---|
| `--epochs` | 100 | Épocas de entrenamiento |
| `--batch` | 64 | Batch size |
| `--lr` | 1e-3 | Learning rate inicial (cosine decay) |
| `--dropout` | 0.3 | Dropout en capas Conv y FC |
| `--val_split` | 0.15 | Fracción de validación |
| `--augment` | 10 | Versiones augmentadas por video original |

## Vector de features (135 valores / frame)

```
[  0: 42] left_hand  — 21 puntos × (x, y)
[ 42: 84] right_hand — 21 puntos × (x, y)
[ 84:135] pose       — 17 puntos × (x, y, visibility)
```

## Outputs del entrenamiento

```
runs/
├── best_model.pt          # Mejor checkpoint por val_acc
├── checkpoint_epoch10.pt  # Checkpoints periódicos
├── classes.json           # Lista de clases (necesaria para Core ML)
└── training_curves.png    # Curvas de loss y accuracy
```
