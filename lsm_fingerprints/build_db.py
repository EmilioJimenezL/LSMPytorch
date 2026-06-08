"""
LSM Phase 1 — Build fingerprint database from landmark JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from constants import FEATURE_DIM, FINGERPRINT_DIM, MIN_VALID_FRAMES, N_FRAMES
from export_bin import export_bin
from fingerprint import FingerprintExtractor
from preprocess import prepare_sequence

# Dataset discovery from TrainingPipeline (parent package)
_TP = Path(__file__).resolve().parent.parent
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))


def discover_entries(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    """Return list of (category, word, json_path)."""
    subdirs = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    ]

    has_direct_json = any(
        list(d.glob("*_landmarks.json")) for d in subdirs[:5]
    )

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
    return entries


def build_database(
    dataset_dir: str,
    output_path: str,
    min_videos: int = 2,
    export_bin_path: str | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    extractor = FingerprintExtractor()

    entries = discover_entries(dataset_dir)
    word_entries: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for cat, word, jf in entries:
        word_entries[word].append((cat, jf))

    valid_words = sorted(k for k, v in word_entries.items() if len(v) >= min_videos)
    classes = valid_words
    class_to_idx = {c: i for i, c in enumerate(classes)}

    fp_list, label_list, seq_list, cat_list, paths = [], [], [], [], []
    skipped = 0

    for word in valid_words:
        for cat, jf in word_entries[word]:
            try:
                seq = prepare_sequence(json_path=str(jf))
                if seq is None:
                    skipped += 1
                    continue
                fp = extractor.extract(seq)
                fp_list.append(fp)
                label_list.append(class_to_idx[word])
                seq_list.append(seq)
                cat_list.append(cat)
                paths.append(str(jf))
            except Exception as e:
                print(f"  ⚠ Error en {jf}: {e}")
                skipped += 1

    if not fp_list:
        raise RuntimeError("No valid videos found in dataset")

    fingerprints = np.stack(fp_list, axis=0)
    labels = np.array(label_list, dtype=np.int64)
    sequences = np.stack(seq_list, axis=0)
    categories = np.array(cat_list, dtype=object)

    metadata = {
        "n_classes": len(classes),
        "n_videos": len(fp_list),
        "build_date": datetime.now(timezone.utc).isoformat(),
        "n_frames": N_FRAMES,
        "feature_dim": FEATURE_DIM,
        "fingerprint_dim": FINGERPRINT_DIM,
        "skipped": skipped,
        "source_dir": str(dataset_dir),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fingerprints=fingerprints,
        labels=labels,
        sequences=sequences,
        classes=np.array(classes, dtype=object),
        categories=categories,
        paths=np.array(paths, dtype=object),
        metadata=json.dumps(metadata),
    )

    if export_bin_path:
        export_bin(
            fingerprints, labels.astype(np.uint32), sequences, classes, export_bin_path
        )

    print(f"\n  Base de datos construida:")
    print(f"    Clases    : {len(classes)}")
    print(f"    Videos    : {len(fp_list)}")
    print(f"    Saltados  : {skipped}")
    print(f"    Salida    : {output_path}")
    if export_bin_path:
        print(f"    Binario   : {export_bin_path}")

    return metadata


def load_database(db_path: str | Path) -> dict:
    data = np.load(db_path, allow_pickle=True)
    classes = list(data["classes"])
    metadata_raw = data.get("metadata")
    metadata = json.loads(str(metadata_raw)) if metadata_raw is not None else {}

    return {
        "fingerprints": data["fingerprints"],
        "labels": data["labels"],
        "sequences": data["sequences"],
        "classes": classes,
        "categories": list(data["categories"]) if "categories" in data else None,
        "paths": list(data["paths"]) if "paths" in data else None,
        "metadata": metadata,
    }


def add_class(
    db_path: str,
    class_name: str,
    sequences: list[np.ndarray],
    category: str = "",
    export_bin_path: str | None = None,
) -> dict:
    """Append new class videos to existing database."""
    db = load_database(db_path)
    extractor = FingerprintExtractor()
    classes = list(db["classes"])

    if class_name in classes:
        label = classes.index(class_name)
    else:
        label = len(classes)
        classes.append(class_name)

    new_fps, new_labels, new_seqs, new_cats = [], [], [], []
    for seq in sequences:
        if seq.shape != (N_FRAMES, FEATURE_DIM):
            raise ValueError(f"Sequence must be ({N_FRAMES}, {FEATURE_DIM})")
        new_fps.append(extractor.extract(seq))
        new_labels.append(label)
        new_seqs.append(seq)
        new_cats.append(category)

    fingerprints = np.concatenate([db["fingerprints"], np.stack(new_fps)], axis=0)
    labels = np.concatenate([db["labels"], np.array(new_labels, dtype=np.int64)])
    sequences_arr = np.concatenate([db["sequences"], np.stack(new_seqs)], axis=0)

    old_cats = db.get("categories") or [""] * len(db["labels"])
    categories = np.array(list(old_cats) + new_cats, dtype=object)

    metadata = dict(db["metadata"])
    metadata["n_classes"] = len(classes)
    metadata["n_videos"] = len(labels)
    metadata["build_date"] = datetime.now(timezone.utc).isoformat()

    np.savez_compressed(
        db_path,
        fingerprints=fingerprints,
        labels=labels,
        sequences=sequences_arr,
        classes=np.array(classes, dtype=object),
        categories=categories,
        metadata=json.dumps(metadata),
    )

    if export_bin_path:
        export_bin(
            fingerprints, labels.astype(np.uint32), sequences_arr, classes, export_bin_path
        )

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Construye fingerprints.npz")
    parser.add_argument("--dataset", required=True, help="Carpeta con *_landmarks.json")
    parser.add_argument("--output", default="./fingerprints.npz")
    parser.add_argument("--export-bin", default=None, help="Ruta fingerprints.bin")
    parser.add_argument("--min-videos", type=int, default=2)
    args = parser.parse_args()
    build_database(
        args.dataset,
        args.output,
        min_videos=args.min_videos,
        export_bin_path=args.export_bin,
    )


if __name__ == "__main__":
    main()
