# LSMPytorch

End-to-end pipeline for **Lengua de Señas Mexicana (LSM)** recognition: extract Vision landmarks from videos, train PyTorch sequence models, convert to Core ML, and test on iOS.

## Tech stack

| Stage | Tech |
|---|---|
| Landmark extraction | Swift, Apple Vision (macOS only) |
| Training | Python 3.11+, PyTorch (MPS / CUDA / CPU) |
| Conversion | coremltools |
| Mobile testing | SwiftUI iOS app |

## Current state

Lab **cap-5** cycle (≤5 videos/class, layout `v2_162`, 518 classes):

| Model | Val top-1 | Extra-test top-1 | Stopped |
|---|---|---|---|
| CNN + attn pool | **36.7%** | 26.6% | epoch 87 |
| TCN non-causal + attn | **33.2%** | 23.8% | epoch 114 |

Earlier uncapped LSMOutput cycle reached ~48% val. Full notes: [Runs/lab_cap5/](Runs/lab_cap5/).

**macOS is the recommended platform** (extraction + Core ML predict + iOS). Linux and Windows can train and convert; Windows conversion needs WSL2.

---

## Clone

```bash
git clone <repo-url> LSMPytorch
cd LSMPytorch
git lfs install
git lfs pull
```

Large weights (`.pt`, `.mlpackage`) use Git LFS.

---

## Requirements by stage

1. **Videos** — `category/word/*.mp4|mov` (or already-extracted landmark JSON trees).
2. **Extract** — macOS + Xcode (Apple Vision).
3. **Train** — Python 3.11+, optional GPU (Apple Silicon MPS, NVIDIA CUDA, or CPU).
4. **Convert** — coremltools (predict/parity needs macOS; convert works on Linux; Windows → WSL2).
5. **Test on device** — Xcode + iPhone (or simulator for Phase-2 parity).

---

## Quickstart (macOS)

```bash
./lsm setup          # creates .venv, installs torch + deps
./lsm doctor         # device, Core ML, Xcode, git-lfs

# End-to-end from videos:
./lsm pipeline --videos /path/to/videos --models cnn,tcn --name my_run

# Or step by step:
./lsm extract  --videos /path/to/videos --output Workspace/landmarks
./lsm prepare  --landmarks Workspace/landmarks --layout v2_162 --cap 5 --name my_run
./lsm train    --model cnn --cache Workspace/prepared/my_run.npz --name my_run_cnn --pool attn
./lsm convert  --run my_run_cnn --fixture
./lsm deploy   --run my_run_cnn --sign-set labCap5
./lsm test
```

Every command prints a resolved plan first. Use `--dry-run` to preview, `-y` to skip overwrite prompts, `-v`/`-q` for verbosity. Full flag reference: [Orchestrator/README.md](Orchestrator/README.md).

Open the iOS app: `TestingTools/LSMMobileModelTesting/LSMMobileModelTesting.xcodeproj`.

---

## Linux

Training and conversion only. Produce landmarks on a Mac (or share a landmarks tree).

```bash
./lsm setup --cuda 12.4   # or: ./lsm setup --cpu
./lsm doctor
./lsm prepare --landmarks /data/landmarks --layout v2_162 --cap 5 --name lab
./lsm train   --model cnn --cache Workspace/prepared/lab.npz --name lab_cnn --pool attn
./lsm convert --run lab_cnn --fixture   # parity skipped off-macOS
```

---

## Windows

Native PowerShell for setup/train. Core ML conversion: use **WSL2** (coremltools has no native Windows support).

```powershell
.\lsm.ps1 setup --cuda 12.4   # or --cpu
.\lsm.ps1 doctor
.\lsm.ps1 prepare --landmarks D:\landmarks --layout v2_162 --cap 5 --name lab
.\lsm.ps1 train --model cnn --cache Workspace\prepared\lab.npz --name lab_cnn --pool attn
```

In WSL2, run `./lsm convert --run lab_cnn` against the same repo checkout.

---

## Repo map

| Directory | Role |
|---|---|
| [Orchestrator/](Orchestrator/) | `lsm` CLI |
| [LandmarkExtraction/](LandmarkExtraction/) | macOS Vision extractors (CLI + GUI) |
| [TrainingPipeline/](TrainingPipeline/) | prepare, train, models, tests |
| [CoreMLConversion/](CoreMLConversion/) | PyTorch → Core ML + app export |
| [TestingTools/](TestingTools/) | iOS inference / grading app |
| [Runs/](Runs/) | Training runs (metrics, curves, READMEs) |
| [Models/](Models/) | `Pytorch/` checkpoints + `CoreML/` packages |
| [Legacy/](Legacy/) | Older cycles, Fase 1 fingerprints, deprecated scripts |
| [Utils/](Utils/) | Reports, extractor sync check, word lists |
| [Workspace/](Workspace/) | Local caches, logs (gitignored contents) |

Agents: see [AGENTS.md](AGENTS.md) and [`.cursor/skills/`](.cursor/skills/).
