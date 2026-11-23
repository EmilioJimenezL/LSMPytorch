# LSMExtractorCLI

Command-line landmark extractor (macOS). Same engine and JSON schema as the [GUI](../LSMExtractorGUI/README.md); no UI, suitable for scripts and `lsm extract`. Format details: [`../README.md`](../README.md).

## Files

`LSMExtractorCLI/` (the Swift target) contains:

| File | Role |
|---|---|
| `main.swift` | Argument parsing, path validation (realpath-canonicalized), exit codes |
| `BatchRunner.swift` | Batch loop, `--skip-existing`, writes manifest and `_batch_report.json`, one log line per video |
| `LandmarkExtractor.swift` | Vision hand + body pose detection, orientation handling, optional overlay video, JSON writer |
| `DatasetScanner.swift` | `category/word` discovery (video extensions `mp4`, `mov`, `m4v`), manifest build/merge |
| `Models.swift` | Schema v2 types, `DatasetManifest`, `BatchReport` |

`LSMExtractorCLI.xcodeproj` is a command-line-tool project, deployment target macOS 26.5, Swift 5. The engine files `Models`, `DatasetScanner`, `LandmarkExtractor` must stay identical to the GUI's (`python Utils/check_extractor_sync.py`).

## Build

```bash
cd LandmarkExtraction/LSMExtractorCLI
xcodebuild -project LSMExtractorCLI.xcodeproj -scheme LSMExtractorCLI \
  -configuration Release -derivedDataPath build build
# binary: build/Build/Products/Release/LSMExtractorCLI
```

Or open the project in Xcode and run. `./lsm extract` builds it on demand.

## Run

```bash
LSMExtractorCLI --input ./LSMVideos --output ./LSMVideosOutput --skip-existing
LSMExtractorCLI -i ./LSMVideos -o ./out -m ./out/dataset_manifest.json --overlay
LSMExtractorCLI --help
```

| Flag | Meaning | Default |
|---|---|---|
| `-i, --input <dir>` | Dataset root `category/word/*.{mp4,mov,m4v}` (required) | – |
| `-o, --output <dir>` | Output root (required, created if missing) | – |
| `-m, --manifest <file>` | Existing manifest to load and update | `{output}/dataset_manifest.json` |
| `--overlay` | Also write `{base}_overlay.mp4` | off |
| `--skip-existing` | Skip videos that already have `*_landmarks.json` | off |
| `-h, --help` | Usage | – |

## Inputs and outputs

- **In:** `{input}/{category}/{word}/*.mp4|mov|m4v`.
- **Out:** `{output}/{category}/{word}/{base}_landmarks.json` (schema v2), optional `{base}_overlay.mp4`, plus `{output}/dataset_manifest.json` (or the `--manifest` path) and `{output}/_batch_report.json`.
- **Log:** stdout, one line per video, then a summary (processed/total, skipped, errors, elapsed). Errors go to stderr.
- **Exit codes:** `0` ok, `1` invalid arguments/paths or manifest/report save failure, `2` at least one video failed.

The output folder is what you pass as `--landmarks` to `lsm prepare` / `lsm pipeline`.
