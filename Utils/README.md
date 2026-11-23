# Utils

Small standalone tools and shared data files. Scripts import `TrainingPipeline` through `_paths.py` (adds it to `sys.path`; also defines `REPO_ROOT`, `TRAINING_PIPELINE`, `LANDMARK_EXTRACTION`, `CLI_ENGINE`, `GUI_ENGINE`). Run them from anywhere with `python Utils/<script>.py`.

| File | Purpose |
|---|---|
| `check_extractor_sync.py` | Fails if the CLI and GUI extractor engines drifted. Compares `Models.swift`, `DatasetScanner.swift`, `LandmarkExtractor.swift` between `LandmarkExtraction/LSMExtractorCLI` and `.../LSMExtractorGUI`, ignoring `import` lines. Exit `0` in sync, `1` otherwise. No arguments |
| `dataset_report.py` | Inventory of a landmark dataset: classes, videos per class, per-category counts, and optional overlap with a model's class list |
| `list_words.py` | Lists or exports the vocabulary in `themes.json` |
| `themes.json` | Vocabulary and categories: 14 categories, 501 words, 2,134 videos (`name`, `words[{name, videoUrl, videos[]}]`). Used by `organize_from_json.py`; the extractors do not use it |
| `palabras.txt` | Text export of the vocabulary (output of `list_words.py --exportar`) |
| `live_test_log_template.csv` | Header for manual live-test logs: `session_id,sign,category,backend,top1,confidence,inference_ms,correct_yes_no,notes`. The iOS app writes its own richer CSV, see [`TestingTools`](../TestingTools/README.md) |

## dataset_report.py

```bash
python Utils/dataset_report.py --dataset <landmarks dir> [--output ./reports] [--min-videos 2] [--fase2-classes Runs/<run>/classes.json]
```

Inputs: landmark dataset root (nested or flat), `--min-videos` default 2, optional `classes.json`. Outputs in `--output` (default `./reports`): `dataset_inventory.json`, `class_overlap.json`. It loads data through the v1 path (`dataset.load_raw_dataset`), so counts reflect v1 validity rules.

## list_words.py

```bash
python Utils/list_words.py --json Utils/themes.json [--categoria "Animales e Insectos"] [--exportar palabras.txt]
```

Prints each word with its video folder, grouped by category (case-insensitive substring filter on the category), with totals; `--exportar` writes the same text to a file.
