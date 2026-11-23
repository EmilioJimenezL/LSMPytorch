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
    Acepta schema v1 (array) y v2 (documento con frames).
    """
    from features import frame_to_vector_v1, load_landmark_json

    frames, _meta = load_landmark_json(json_path)
    if not frames:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([frame_to_vector_v1(f) for f in frames], axis=0)


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

def _load_sequence_from_json(
    json_path: str,
    shoulder_norm: bool = True,
    layout: str = "v1_135",
) -> np.ndarray | None:
    """Load one video as (N_FRAMES, feature_dim) for the requested layout."""
    from features import LAYOUT_V1, LAYOUT_V2, prepare_sequence_v1, prepare_sequence_v2

    if layout == LAYOUT_V2:
        return prepare_sequence_v2(json_path=str(json_path))
    # v1 / legacy
    if shoulder_norm:
        return prepare_sequence_v1(json_path=str(json_path))
    seq = json_to_sequence(str(json_path))
    if len(seq) < 10:
        return None
    return interpolate_sequence(seq, N_FRAMES)


def _discover_entries(dataset_dir: Path) -> tuple[list[tuple[str, str, Path]], bool]:
    """Return ([(categoria, palabra, json_path), ...], has_direct_json)."""
    subdirs = [d for d in dataset_dir.iterdir()
               if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")]

    has_direct_json = any(
        list(d.glob("*_landmarks.json"))
        for d in subdirs[:5]
    ) if subdirs else False

    entries: list[tuple[str, str, Path]] = []
    if has_direct_json:
        for word_dir in sorted(subdirs):
            for jf in sorted(word_dir.glob("*_landmarks.json")):
                entries.append(("", word_dir.name, jf))
    else:
        for cat_dir in sorted(subdirs):
            if not cat_dir.is_dir():
                continue
            for word_dir in sorted(cat_dir.iterdir()):
                if not word_dir.is_dir() or word_dir.name.startswith("."):
                    continue
                for jf in sorted(word_dir.glob("*_landmarks.json")):
                    entries.append((cat_dir.name, word_dir.name, jf))
    return entries, has_direct_json


def _class_key(word: str, category: str, ambiguous_words: set[str]) -> str:
    """Disambiguate words that appear in more than one category."""
    if word in ambiguous_words and category:
        return f"{word} ({category})"
    return word


def load_raw_dataset(
    dataset_dir: str,
    min_videos: int = 2,
    shoulder_norm: bool = True,
    layout: str = "v1_135",
    max_videos_per_class: int | None = None,
):
    """
    Carga todos los _landmarks.json SIN augmentation.

    - Class key is the word folder; if the same word appears in two categories,
      the key becomes ``word (category)``.
    - ``min_videos`` counts *valid* sequences after preprocessing, not files.
    - ``max_videos_per_class`` is applied later by split_utils.cap_and_split_indices
      (all valid sequences are returned; the cap only affects the split).

    Returns:
        X, y, classes, meta
    """
    from collections import defaultdict
    from features import LAYOUT_V1, LAYOUT_V2, feature_dim_for_layout

    if layout not in (LAYOUT_V1, LAYOUT_V2):
        raise ValueError(f"Unknown layout: {layout}")

    dataset_dir = Path(dataset_dir)
    entries, has_direct_json = _discover_entries(dataset_dir)

    # Words that appear in more than one category → disambiguate.
    word_cats: dict[str, set[str]] = defaultdict(set)
    for cat, word, _jf in entries:
        if cat:
            word_cats[word].add(cat)
    ambiguous = {w for w, cats in word_cats.items() if len(cats) > 1}

    # Group by disambiguated class key.
    class_entries: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for cat, word, jf in entries:
        key = _class_key(word, cat, ambiguous)
        class_entries[key].append((cat, jf))

    # Load all sequences first, then filter by valid-count >= min_videos.
    prepared: dict[str, list[tuple[np.ndarray, str, str]]] = defaultdict(list)
    # class → [(seq, categoria, archivo), ...]
    load_errors = 0
    for key in sorted(class_entries.keys()):
        for cat, jf in class_entries[key]:
            try:
                seq = _load_sequence_from_json(
                    str(jf), shoulder_norm=shoulder_norm, layout=layout,
                )
                if seq is None:
                    continue
                prepared[key].append((seq, cat, str(jf)))
            except Exception as e:
                print(f"  ⚠ Error en {jf}: {e}")
                load_errors += 1

    valid_classes = sorted(k for k, v in prepared.items() if len(v) >= min_videos)
    skipped = sorted(
        (k, len(class_entries[k]), len(prepared.get(k, [])))
        for k in class_entries
        if len(prepared.get(k, [])) < min_videos
    )

    if skipped:
        print(f"\n  ⚠ Clases excluidas por tener < {min_videos} secuencias válidas ({len(skipped)}):")
        for name, n_files, n_valid in skipped[:40]:
            print(f"     - {name} ({n_valid} válidas / {n_files} archivos)")
        if len(skipped) > 40:
            print(f"     ... y {len(skipped) - 40} más")

    classes = valid_classes
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    X_list, y_list, meta_list = [], [], []
    for word in classes:
        for seq, cat, path in prepared[word]:
            X_list.append(seq)
            y_list.append(class_to_idx[word])
            meta_list.append({
                "clase": word,
                "categoria": cat,
                "archivo": path,
            })

    dim = feature_dim_for_layout(layout)
    estructura = "plana" if has_direct_json else "anidada (categoria/palabra)"
    print(f"\n  Dataset cargado ({estructura}, layout={layout}):")
    print(f"    Clases válidas      : {len(classes)}")
    print(f"    Clases excluidas    : {len(skipped)}")
    print(f"    Videos en memoria   : {len(X_list)}")
    print(f"    Feature dim         : {dim}")
    if ambiguous:
        print(f"    Palabras desambiguadas: {sorted(ambiguous)}")
    if load_errors:
        print(f"    Errores de carga    : {load_errors}")
    if max_videos_per_class is not None:
        print(f"    Cap por clase       : {max_videos_per_class} (aplicado en el split)")

    if not X_list:
        raise RuntimeError("No valid sequences loaded")

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
