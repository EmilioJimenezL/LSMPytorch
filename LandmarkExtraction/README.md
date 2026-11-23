# LandmarkExtraction

**macOS only.** Turns sign-language videos into landmark JSON files (hands + body pose) using Apple Vision. Everything downstream (`TrainingPipeline`) consumes these JSON files, so this is the first stage of the pipeline. On Linux/Windows, get the landmarks from a Mac instead.

```
videos  ──►  LSMExtractorCLI / LSMExtractorGUI  ──►  *_landmarks.json  ──►  TrainingPipeline (prepare/train)
```

## Contents

| Path | Purpose |
|---|---|
| [`LSMExtractorCLI/`](./LSMExtractorCLI/README.md) | Headless command-line extractor (used by `lsm extract`) |
| [`LSMExtractorGUI/`](./LSMExtractorGUI/README.md) | SwiftUI desktop app, same engine, folder pickers + live log |
| `organize_from_json.py` | Optional: sorts a flat folder of `.mp4` files into `Category/Word/` using `Utils/themes.json` |

The two Xcode projects share an identical extraction engine (`Models.swift`, `DatasetScanner.swift`, `LandmarkExtractor.swift`). `Utils/check_extractor_sync.py` fails if they drift (it ignores `import` lines). Change both together.

## Input layout

Discovery is **path-based**; `themes.json` is not used by the extractors. Only this depth is scanned:

```
{input}/{category}/{word}/*.{mp4,mov,m4v}
```

Category and word come from folder names. Category and word folders that are symlinks are resolved. Videos at any other depth are skipped and logged.

## Output layout

The output mirrors the input under a separate root:

```
{output}/
├── dataset_manifest.json            # catalog (version 1); merged with an existing one if given
├── _batch_report.json               # totals, elapsed seconds, failed videos
└── Familia/Abuela/
    ├── Abuela1_landmarks.json       # schema v2 (input to TrainingPipeline)
    └── Abuela1_overlay.mp4          # only with overlay enabled
```

`TrainingPipeline` accepts either this nested layout or a flat `word/*_landmarks.json` layout. Pass `{output}` (or a `landmarks/` subfolder of it) as the dataset root.

## Landmark JSON, schema v2

```json
{
  "schema_version": 2,
  "fps": 30.0, "width": 1080, "height": 1920,
  "orientation": "up | right | left | down",
  "frames": [
    {
      "frame": 0, "timestamp_ms": 0.0,
      "leftHand":  [{"x": 0.45, "y": 0.62, "c": 0.98}, null, "... 21 entries"],
      "rightHand": [ "... 21 entries, null = joint not detected" ],
      "pose": [{"id": 6, "name": "left_shoulder", "x": 0.4, "y": 0.5, "visibility": 0.9}, "... 12 joints"]
    }
  ]
}
```

- **Hands:** 21 Vision joints per hand, each `{x, y, c}` or `null`. Joints below confidence 0.1 are `null`. Left/right comes from Vision chirality; up to 2 hands per frame.
- **Pose:** 12 distinct joints, ids `0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear, 5 neck, 6 left_shoulder, 7 right_shoulder, 8 left_elbow, 9 right_elbow, 10 left_wrist, 11 right_wrist`. Low-confidence joints get `visibility 0`.
- **Coordinates:** normalized `[0, 1]`, **Y=0 at the top** (Vision's bottom-origin Y is flipped to `1 - y` on write).
- **Orientation:** derived from the video's `preferredTransform` and passed to Vision, so rotated phone clips are handled.
- **Schema v1** (older datasets) is a bare array of frames with a 17-slot MediaPipe-style pose (`id`, `x`, `y`, `visibility`) and no `c` on hands. The Python loaders (`TrainingPipeline/features.py`) read both schemas and can build either tensor layout (`v1_135` or `v2_162`) from v2 JSON. See [`TrainingPipeline/README.md`](../TrainingPipeline/README.md).

## Flags

Both tools process the same way; the CLI exposes them as flags and the GUI as controls.

| CLI flag | GUI control | Meaning | Default |
|---|---|---|---|
| `-i, --input DIR` | Dataset | Root with `category/word/*.mp4` (required) | – |
| `-o, --output DIR` | Salida | Output root, created if missing (required) | – |
| `-m, --manifest FILE` | Manifiesto | Existing `dataset_manifest.json` to load and update in place | `{output}/dataset_manifest.json` |
| `--overlay` | Generar video overlay | Also write `{base}_overlay.mp4` with the skeleton drawn | off (JSON only, faster) |
| `--skip-existing` | – | Skip videos whose `*_landmarks.json` exists; marked `skipped` | off |

CLI exit codes: `0` success, `1` bad arguments/paths or manifest/report could not be saved, `2` at least one video failed.

## organize_from_json.py

Optional pre-step when videos arrive as one flat folder.

```bash
python LandmarkExtraction/organize_from_json.py \
  --json Utils/themes.json --input ./videos --output ./dataset [--dry-run] [--verbose]
```

Validates each file with `ffprobe` (rejects GIFs renamed to `.mp4`, image codecs, corrupt files) and moves it to `output/Category/Word/`. Invalid files go to `_invalidos/`, files with no match in the JSON go to `_sin_match/`. Requires `ffprobe` (FFmpeg). Output is the `input` for the extractors.

## Requirements

- macOS with Xcode. Deployment targets: CLI macOS 26.5, GUI macOS 15.7 (Swift 5).
- No entitlements. The GUI uses `NSOpenPanel`; the CLI inherits the terminal's file permissions.
- `build/` folders inside each project are local build output and gitignored. `LSMExtractorCLI/build/sample_v2_out/` is a small sample some `TrainingPipeline` tests read; those tests skip when it is absent.
