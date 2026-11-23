# LSMExtractorGUI

SwiftUI macOS app for batch landmark extraction. Same engine and output as the [CLI](../LSMExtractorCLI/README.md); use it when you want folder pickers, a live log, and per-video status. Format details: [`../README.md`](../README.md).

## Use

1. Open `LSMExtractorGUI.xcodeproj` in Xcode and run (⌘R). Deployment target macOS 15.7.
2. Pick the **Dataset** folder (`category/word/*.mp4`, `.mov`, `.m4v`).
3. Pick the **Salida** (output) folder.
4. Optional: pick an existing **Manifiesto** (`dataset_manifest.json`); it is updated in place when the batch ends. Without one, `{output}/dataset_manifest.json` is written.
5. Optional: turn off **Generar video overlay** for JSON-only extraction (faster).
6. **Procesar**. **Cancelar** stops the batch. **Abrir salida** opens the output folder when done.

The window shows overall progress, a log (videos outside `category/word/` are listed as skipped), and a per-video list with status (pending, processing, done, error, skipped) and frame counts. At the end it writes `{output}/_batch_report.json`.

## Files (`LSMExtractorGUI/`)

| File | Role |
|---|---|
| `LSMExtractorApp.swift` | App entry point |
| `ContentView.swift` | Control panel, progress, log, video list |
| `BatchProcessor.swift` | `ObservableObject` driving the batch: manifest load/merge, processing loop, cancellation, report |
| `LandmarkExtractor.swift` | Vision detection, optional overlay, JSON writer (shared engine) |
| `DatasetScanner.swift` | `category/word` discovery and manifest merge (shared engine) |
| `Models.swift` | Schema v2 types and manifest/report models (shared engine) |
| `LSMExtractorGUI-Bridging-Header.h` | Bridging header |

`Models`, `DatasetScanner` and `LandmarkExtractor` must match the CLI's copies (`python Utils/check_extractor_sync.py`).

## Inputs and outputs

Identical to the CLI: `{category}/{word}/{base}_landmarks.json` (schema v2), optional `{base}_overlay.mp4`, `dataset_manifest.json`, `_batch_report.json`. The GUI has no `--skip-existing` equivalent; re-running overwrites files for the selected videos. For unattended or repeat runs use the CLI.
