"""
LSM — Dataset loader y preprocesamiento
========================================
Carga los JSON de Vision Framework, aplica interpolación temporal a N_FRAMES
y construye tensores listos para entrenamiento.
"""

import json
import os
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# ── Constantes del vector de features ────────────────────────────────────────
N_LEFT_HAND  = 21 * 2   # 42
N_RIGHT_HAND = 21 * 2   # 42
N_POSE       = 17 * 3   # 51  (x, y, visibility)
FEATURE_DIM  = N_LEFT_HAND + N_RIGHT_HAND + N_POSE  # 135

N_FRAMES = 85  # Percentil 10 del dataset


# ── Extracción de vector por frame ────────────────────────────────────────────

def frame_to_vector(frame: dict) -> np.ndarray:
    """
    Convierte un frame del JSON de Vision Framework a un vector de 135 valores.
    Orden: [left_hand (42), right_hand (42), pose (51)]
    Puntos no detectados → 0.0
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Left hand [0:42]
    lh = frame.get("leftHand") or frame.get("left_hand")
    if lh:
        for i, pt in enumerate(lh[:21]):
            vec[i*2]     = pt["x"]
            vec[i*2 + 1] = pt["y"]

    # Right hand [42:84]
    rh = frame.get("rightHand") or frame.get("right_hand")
    if rh:
        offset = N_LEFT_HAND
        for i, pt in enumerate(rh[:21]):
            vec[offset + i*2]     = pt["x"]
            vec[offset + i*2 + 1] = pt["y"]

    # Pose [84:135]
    pose = frame.get("pose")
    if pose:
        offset = N_LEFT_HAND + N_RIGHT_HAND
        for pt in pose:
            idx = pt["id"]
            if idx < 17:
                base = offset + idx * 3
                vec[base]     = pt["x"]
                vec[base + 1] = pt["y"]
                vec[base + 2] = pt.get("visibility", 1.0)

    return vec


def json_to_sequence(json_path: str) -> np.ndarray:
    """
    Carga un JSON y retorna una secuencia (n_frames_original, 135).
    """
    with open(json_path) as f:
        data = json.load(f)

    frames = data if isinstance(data, list) else data.get("frames", [])
    return np.stack([frame_to_vector(f) for f in frames], axis=0)


# ── Interpolación temporal ────────────────────────────────────────────────────

def interpolate_sequence(seq: np.ndarray, target_len: int = N_FRAMES) -> np.ndarray:
    """
    Resamplea una secuencia (T, 135) a (target_len, 135) via interpolación lineal.
    """
    T = len(seq)
    if T == target_len:
        return seq

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    interp = interp1d(x_old, seq, axis=0, kind="linear")
    return interp(x_new).astype(np.float32)


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Aplica augmentation aleatoria a una secuencia (N_FRAMES, 135).
    Modifica solo coordenadas x,y — no toca visibility de pose.

    Augmentations:
      - Escala           : factor 0.9–1.1 alrededor del centroide
      - Traslación       : ±0.05 en x e y
      - Cambio velocidad : resamplea a 0.8x–1.2x y vuelve a N_FRAMES
      - Ruido gaussiano  : σ = 0.01
    """
    seq = seq.copy()

    # Índices de coordenadas x,y (excluye visibility de pose)
    xy_indices = list(range(N_LEFT_HAND + N_RIGHT_HAND))  # hands: todos son x,y
    for i in range(17):                                    # pose: solo x,y, skip visibility
        base = N_LEFT_HAND + N_RIGHT_HAND + i * 3
        xy_indices += [base, base + 1]

    # 1. Escala alrededor del centroide
    if rng.random() < 0.8:
        scale = rng.uniform(0.9, 1.1)
        # Calcula centroide solo con puntos no cero
        nonzero = seq[:, xy_indices]
        mask = nonzero != 0
        if mask.any():
            centroid = nonzero[mask].mean()
            seq[:, xy_indices] = np.where(
                seq[:, xy_indices] != 0,
                centroid + (seq[:, xy_indices] - centroid) * scale,
                0.0
            )

    # 2. Traslación
    if rng.random() < 0.8:
        tx = rng.uniform(-0.05, 0.05)
        ty = rng.uniform(-0.05, 0.05)
        x_indices = [i for i in xy_indices if i % 2 == 0]
        y_indices = [i for i in xy_indices if i % 2 == 1]
        # Solo trasladar puntos no cero
        seq[:, x_indices] = np.where(seq[:, x_indices] != 0,
                                     seq[:, x_indices] + tx, 0.0)
        seq[:, y_indices] = np.where(seq[:, y_indices] != 0,
                                     seq[:, y_indices] + ty, 0.0)

    # 3. Cambio de velocidad (time warp)
    if rng.random() < 0.7:
        speed = rng.uniform(0.8, 1.2)
        new_len = max(10, int(N_FRAMES * speed))
        seq = interpolate_sequence(seq, new_len)
        seq = interpolate_sequence(seq, N_FRAMES)  # volver a N_FRAMES

    # 4. Ruido gaussiano
    if rng.random() < 0.6:
        noise = rng.normal(0, 0.01, seq.shape).astype(np.float32)
        # Solo añadir ruido a coordenadas no cero
        mask = seq != 0
        seq = np.where(mask, seq + noise, 0.0)

    return seq.astype(np.float32)


# ── Carga del dataset completo ────────────────────────────────────────────────

def load_raw_dataset(dataset_dir: str, min_videos: int = 2):
    """
    Carga todos los _landmarks.json SIN augmentation.
    Soporta dos estructuras:
        - Plana   : dataset/palabra/video_landmarks.json
        - Anidada : dataset/categoria/palabra/video_landmarks.json

    Solo incluye clases con al menos min_videos secuencias válidas.

    Returns:
        X       : np.ndarray (N_videos, N_FRAMES, 135)
        y       : np.ndarray (N_videos,) — índices de clase
        classes : list[str] — nombres de clase ordenados alfabéticamente
        meta    : list[dict] — metadata de cada muestra (clase, categoria, archivo)
    """
    dataset_dir = Path(dataset_dir)

    # ── Detectar estructura ───────────────────────────────────────────────────
    # Si las subcarpetas de primer nivel contienen directamente JSON → estructura plana
    # Si contienen más subcarpetas → estructura anidada categoria/palabra
    subdirs = [d for d in dataset_dir.iterdir()
               if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")]

    # Comprobar si alguna subdir de primer nivel tiene JSON directo
    has_direct_json = any(
        list(d.glob("*_landmarks.json"))
        for d in subdirs[:5]  # solo checar las primeras 5
    )

    # Recopilar (categoria, palabra, json_path) para cada archivo
    entries = []  # (categoria, palabra, json_path)

    if has_direct_json:
        # Estructura plana: dataset/palabra/
        for word_dir in sorted(subdirs):
            for jf in sorted(word_dir.glob("*_landmarks.json")):
                entries.append(("", word_dir.name, jf))
    else:
        # Estructura anidada: dataset/categoria/palabra/
        for cat_dir in sorted(subdirs):
            if not cat_dir.is_dir():
                continue
            for word_dir in sorted(cat_dir.iterdir()):
                if not word_dir.is_dir() or word_dir.name.startswith("."):
                    continue
                for jf in sorted(word_dir.glob("*_landmarks.json")):
                    entries.append((cat_dir.name, word_dir.name, jf))

    # ── Agrupar por palabra ───────────────────────────────────────────────────
    from collections import defaultdict
    word_entries = defaultdict(list)  # palabra → [(categoria, jf), ...]
    for cat, word, jf in entries:
        word_entries[word].append((cat, jf))

    # Filtrar clases con menos de min_videos
    valid_words   = sorted(k for k, v in word_entries.items() if len(v) >= min_videos)
    skipped_words = sorted(k for k, v in word_entries.items() if len(v) < min_videos)

    if skipped_words:
        print(f"\n  ⚠ Clases excluidas por tener < {min_videos} videos ({len(skipped_words)}):")
        for w in skipped_words:
            count = len(word_entries[w])
            print(f"     - {w} ({count} video{'s' if count != 1 else ''})")

    classes       = valid_words
    class_to_idx  = {cls: i for i, cls in enumerate(classes)}

    X_list, y_list, meta_list = [], [], []
    skipped_files = 0
    total_files   = 0

    for word in classes:
        for cat, jf in word_entries[word]:
            total_files += 1
            try:
                seq = json_to_sequence(str(jf))
                if len(seq) < 10:
                    skipped_files += 1
                    continue
                X_list.append(interpolate_sequence(seq, N_FRAMES))
                y_list.append(class_to_idx[word])
                meta_list.append({
                    "clase":     word,
                    "categoria": cat,
                    "archivo":   str(jf),
                })
            except Exception as e:
                print(f"  ⚠ Error en {jf}: {e}")
                skipped_files += 1

    estructura = "plana" if has_direct_json else "anidada (categoria/palabra)"
    print(f"\n  Dataset cargado ({estructura}):")
    print(f"    Clases válidas      : {len(classes)}")
    print(f"    Clases excluidas    : {len(skipped_words)}")
    print(f"    Videos procesados   : {total_files - skipped_files}/{total_files}")
    print(f"    Videos en memoria   : {len(X_list)}")
    if skipped_files:
        print(f"    Archivos saltados   : {skipped_files}")

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=np.int64),
        classes,
        meta_list,
    )


def apply_augmentation(X: np.ndarray, y: np.ndarray,
                        augment_factor: int = 10, seed: int = 42):
    """
    Aplica augmentation a un conjunto ya separado (train only).
    Retorna originales + versiones augmentadas mezcladas.
    """
    rng = np.random.default_rng(seed)
    X_aug, y_aug = [X], [y]

    for _ in range(augment_factor):
        batch = np.stack([augment_sequence(x, rng) for x in X], axis=0)
        X_aug.append(batch)
        y_aug.append(y)

    X_out = np.concatenate(X_aug, axis=0)
    y_out = np.concatenate(y_aug, axis=0)

    # Shuffle
    idx = rng.permutation(len(X_out))
    return X_out[idx], y_out[idx]


# Alias para compatibilidad con código anterior
def load_dataset(dataset_dir: str, augment_factor: int = 10):
    """
    DEPRECADO — el augmentation debe aplicarse solo después del split.
    Usar load_raw_dataset() + apply_augmentation() en train.py.
    """
    import warnings
    warnings.warn(
        "load_dataset() aplica augmentation antes del split. "
        "Usa load_raw_dataset() + apply_augmentation() para un split correcto.",
        DeprecationWarning, stacklevel=2
    )
    X, y, classes, _ = load_raw_dataset(dataset_dir)
    X_aug, y_aug = apply_augmentation(X, y, augment_factor)
    return X_aug, y_aug, classes


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python dataset.py <dataset_dir>")
        sys.exit(1)

    X, y, classes, meta = load_raw_dataset(sys.argv[1])
    print(f"\n  X shape : {X.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  Clases  : {classes[:5]} ...")
    print(f"  X range : [{X.min():.3f}, {X.max():.3f}]")
