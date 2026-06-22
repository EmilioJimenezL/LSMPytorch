# AppLSMTests

> **For AI-assisted development:** see [AGENTS.md](./AGENTS.md) for architecture invariants, file index, commands, and doc-update protocol.

Aplicaciones Swift para el proyecto **LSM (Lengua de Señas Mexicana)**. Este repositorio complementa el [TrainingPipeline](../TrainingPipeline/) de Python y cubre las dos etapas que faltaban en el flujo completo: **extracción de landmarks desde video** y **prueba de modelos en dispositivo móvil**.

## Resumen del ecosistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Flujo completo LSM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LSMVideos/           LSMExtractorGUI          lsm_fingerprints (Fase 1) │
│  category/word/*.mp4  (macOS, Vision)    ──►   fingerprints.npz/.bin      │
│  + manifest (opt.)    *_landmarks.json          evaluate.py (LOO)         │
│                       *_overlay.mp4 (opt.)      │                         │
│                                                   ├──► LSMMobileModelTesting│
│                                                   │    (Fingerprint+DTW)  │
│  TrainingPipeline (Fase 2)                        │                         │
│  train.py ──► Core ML ────────────────────────────┘                         │
│  (iOS, cámara en vivo — Fase 1 o Core ML)                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Repositorio / app | Plataforma | Rol |
|---|---|---|
| **LSMExtractorGUI** | macOS | Extrae landmarks de manos y pose desde videos con Vision Framework |
| **TrainingPipeline** | Python | Fase 2: entrena modelos y exporta a Core ML; incluye **lsm_fingerprints** (Fase 1) |
| **LSMMobileModelTesting** | iOS | Inferencia en vivo (Fase 1 fingerprint o Fase 2 Core ML) |

## Proyectos incluidos

### LSMExtractorGUI — Extractor de landmarks (macOS)

App de escritorio en SwiftUI que procesa videos en lote y genera los archivos JSON que consume el pipeline de entrenamiento.

**Qué hace:**

- Lee videos `.mp4`, `.mov` o `.m4v` desde una estructura `category/word/`
- Detecta por frame manos (`VNDetectHumanHandPoseRequest`) y pose corporal (`VNDetectHumanBodyPoseRequest`) con Apple Vision
- Guarda secuencias de landmarks en JSON
- Opcionalmente genera un video con overlay visual de esqueleto para inspección
- Descubre categoría y palabra desde la ruta de carpetas (sin `themes.json`)
- Mantiene un catálogo `dataset_manifest.json` (opcional al inicio, siempre actualizado al final)
- Produce un reporte de batch (`_batch_report.json`)

**Requisitos:**

- macOS 15.7+
- Xcode con SDK macOS
- Apple Silicon o Intel Mac

**Uso:**

1. Abrir `LSMExtractorGUI/LSMExtractorGUI.xcodeproj` en Xcode
2. Compilar y ejecutar (⌘R)
3. Seleccionar la carpeta raíz **Dataset** (`LSMVideos/`) con estructura `category/word/*.mp4`
4. Seleccionar la carpeta de **Salida** (`LSMVideosOutput/`)
5. Opcional: cargar un `dataset_manifest.json` existente (se actualiza en el mismo archivo al terminar)
6. Opcional: desactivar **Generar video overlay** para modo rápido (solo JSON)
7. Pulsar **Procesar**

**Estructura de entrada esperada:**

El extractor recorre solo `inputRoot/{category}/{word}/*.mp4`. La categoría y la palabra se infieren de las carpetas, no del nombre del archivo.

```
LSMVideos/
├── Familia/
│   └── Abuela/
│       ├── Abuela1.mp4
│       └── Abuela2.mp4
└── ...
```

Videos fuera de esa profundidad (p. ej. directamente bajo la raíz o bajo `category/`) se registran en el log como omitidos.

**Estructura de salida:**

La salida replica la estructura de entrada bajo una raíz separada:

```
LSMVideosOutput/
├── dataset_manifest.json              ← catálogo del dataset (generado o actualizado)
├── Familia/
│   └── Abuela/
│       ├── Abuela1_landmarks.json     ← input para train.py
│       └── Abuela1_overlay.mp4        ← opcional (toggle en UI)
└── _batch_report.json                 ← resumen del procesamiento
```

**Archivos principales:**

| Archivo | Responsabilidad |
|---|---|
| `LandmarkExtractor.swift` | Detección Vision, overlay opcional, lectura/escritura de video |
| `BatchProcessor.swift` | Procesamiento en lote, manifiesto, estado de la UI |
| `DatasetScanner.swift` | Descubrimiento por ruta, build/merge de `dataset_manifest.json` |
| `Models.swift` | Tipos JSON, `DatasetManifest`, modelos de estado (`VideoItem`, `BatchReport`) |
| `ContentView.swift` | Interfaz SwiftUI con panel de control, log y lista de videos |

---

### LSMMobileModelTesting — Inferencia en iOS

App de prueba en SwiftUI que ejecuta modelos Core ML en tiempo real usando la cámara del iPhone/iPad.

**Qué hace:**

- Captura video de la cámara frontal o trasera (720p)
- Extrae landmarks por frame con Vision (misma lógica que el extractor)
- Acumula un buffer temporal de hasta 85 frames
- Interpola la secuencia y ejecuta inferencia Core ML cada 15 frames
- Muestra las **Top-5 predicciones** con barras de confianza
- Dibuja esqueleto de manos y pose sobre la vista de cámara
- Panel de debug con FPS, latencia de inferencia, estado de detección y progreso del buffer

**Requisitos:**

- iOS 26.2+ (según deployment target del proyecto)
- Dispositivo físico con cámara (recomendado; el simulador no tiene cámara real)
- Xcode con SDK iOS

**Uso:**

1. Abrir `LSMMobileModelTesting/LSMMobileModelTesting.xcodeproj` en Xcode
2. Conectar un iPhone/iPad y seleccionarlo como destino
3. Compilar y ejecutar (⌘R)
4. Conceder permiso de cámara cuando se solicite
5. Apuntar la cámara hacia alguien realizando una seña
6. Esperar a que el anillo de buffer llegue al 100% (85 frames) para ver predicciones

**Controles de la UI:**

| Control | Función |
|---|---|
| Selector de modelo | Cambia entre **Fase 1 (Fingerprint+DTW)** y **Fase 2 (Core ML)** |
| Esqueleto (figura) | Muestra/oculta overlay de manos y pose |
| Info (ⓘ) | Muestra panel de debug |
| Rotar cámara | Alterna cámara frontal / trasera (reinicia el buffer) |
| Anillo de progreso | Indica cuántos frames hay en el buffer (0–85) |
| Reset (↺) | Limpia buffer y predicciones |

**Modelos incluidos (2026-06):**

| Modo | Archivo | Clases | Val Top-1 | Notas |
|---|---|---|---|---|
| Fase 1 (default) | `fingerprints.bin` | 307 | 6.57% LOO | Huella + DTW |
| Fase 2 CNN | `runs_lsmoutput_cnn_lsmoutput.mlpackage` | 307 | **47.16%** | Recomendado para pruebas |
| Fase 2 TCN | `runs_lsmoutput_tcn_lsmoutput.mlpackage` | 307 | 44.78% | Mejor val Top-5 |
| Fase 2 3D CNN | `runs_lsmoutput_3dcnn_lsmoutput.mlpackage` | 307 | 15.52% | |
| Fase 2 legacy | `coreml_models.mlpackage` | 187 | 9.68% | Pre-LSMOutput |

Todos los artefactos son **locales y gitignored** (~97 MB). Copiar desde TrainingPipeline (ver abajo). El selector de modelos permite alternar entre los 5 backends. **Fase 1 y Fase 2 LSMOutput comparten `SequencePreprocessor`** (normalización hombro).

## Integrar modelos Fase 2 entrenados en LSMOutput

Tras `./train_all_lsmoutput.sh` en TrainingPipeline, copiar todos los backends para A/B:

```bash
IOS=LSMMobileModelTesting/LSMMobileModelTesting
TP=../TrainingPipeline

cp "$TP/lsm_fingerprints/fingerprints.bin" "$IOS/"
cp -r "$TP/exports/runs_lsmoutput_cnn_lsmoutput.mlpackage" "$IOS/"
cp -r "$TP/exports/runs_lsmoutput_tcn_lsmoutput.mlpackage" "$IOS/"
cp -r "$TP/exports/runs_lsmoutput_3dcnn_lsmoutput.mlpackage" "$IOS/"
cp "$TP/runs_lsmoutput_cnn/classes.json" "$IOS/runs_lsmoutput_cnn_classes.json"
cp "$TP/runs_lsmoutput_tcn/classes.json" "$IOS/runs_lsmoutput_tcn_classes.json"
cp "$TP/runs_lsmoutput_3dcnn/classes.json" "$IOS/runs_lsmoutput_3dcnn_classes.json"
```

`AVAILABLE_MODELS` en `LandmarkPipeline.swift` ya registra los 5 backends con métricas val actualizadas.

Lista de señas compartidas para A/B: `TrainingPipeline/reports/class_overlap.json` (307 clases).

## Comparar Fase 1 vs Fase 2 en dispositivo

**Requisito:** Mac + Xcode + dispositivo físico.

1. Usar señas de `TrainingPipeline/reports/class_overlap.json` (307 clases compartidas)
2. Misma sesión: alternar selector entre los 5 backends sin reiniciar la app
3. Registrar resultados en CSV — plantilla: `TrainingPipeline/review/live_test_log_template.csv`

| session_id | sign | category | backend | top1 | confidence | inference_ms | correct_yes_no | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-06-22-1 | Navidad | Festividades | fase2_cnn | Navidad | 0.82 | 45 | yes | |

**Matriz mínima (10 señas):** mezcla de Festividades, Personajes, Colores, Familia, Abecedario.

4. Reportar resultados en [TrainingPipeline/AGENTS.md — iOS validation](../TrainingPipeline/AGENTS.md#ios-validation-results)

Checklist detallado: [AGENTS.md — Phase 2 iOS validation](./AGENTS.md#phase-2-ios-validation-checklist)

**Actualizar base de datos Fase 1:**

```bash
cp ../TrainingPipeline/lsm_fingerprints/fingerprints.bin \
   LSMMobileModelTesting/LSMMobileModelTesting/fingerprints.bin
```

Regenerar desde LSMOutput si hace falta:

```bash
cd ../TrainingPipeline/lsm_fingerprints
python build_db.py \
  --dataset ../LSMOutput \
  --output ./fingerprints.npz \
  --export-bin ./fingerprints.bin \
  --min-videos 2
```

El bin (~95 MB, 307 clases / 1934 videos) **no se commitea** — está en `.gitignore`. El proyecto Xcode usa `PBXFileSystemSynchronizedRootGroup`: basta copiar el archivo a `LSMMobileModelTesting/LSMMobileModelTesting/` y recompilar.

## Validar Fase 1 en dispositivo (después del baseline Python)

Requisitos previos (generados en TrainingPipeline, no commiteados):

- `TrainingPipeline/lsm_fingerprints/fingerprints.bin` — base de producción (~307 clases)
- Baseline LOO registrado en [TrainingPipeline/AGENTS.md](../TrainingPipeline/AGENTS.md#phase-1-baseline-lsmoutput)

### Estado de integración (2026-06-22)

| Paso | Estado | Notas |
|---|---|---|
| Bin en bundle local | Hecho | `fingerprints.bin` (~95 MB) copiado al target; gitignored |
| CNN/TCN/3DCNN mlpackages | Hecho | Copiados desde `TrainingPipeline/exports/`; gitignored |
| `AVAILABLE_MODELS` (5 backends) | Hecho | Fase 1, CNN, TCN, 3DCNN, legacy |
| Header / carga | Verificado offline | Magic `LSMF`, 307 clases, 1934 videos; parse ~0.1 s (Linux) |
| Paridad huella | Verificado offline | `golden_sequence.json` max diff **0.0** (Python); Debug build debe imprimir `✓ Parity check passed` |
| Carga async + UI | Hecho | `loadModel` parsea en background; spinner «Cargando fingerprints.bin…» |
| DTW radius + inferencia async | Hecho | `dtwRadius=10` (Sakoe-Chiba); `match()` en cola de fondo |
| Fase 2 preprocessing parity | Hecho | Core ML path uses `SequencePreprocessor` |
| Phase 1 checklist B2–B5 | **Pending device** | Mac required |
| Phase 2 checklist C2–C5 | **Pending device** | Mac required; use `live_test_log_template.csv` |

Referencia de latencia (Linux desktop, Python FastDTW radius=10): **~930 ms/ciclo** — por encima del objetivo de 200 ms. iOS con Accelerate + banda DTW debería ser más rápido; medir en el panel debug. Si sigue >200 ms, el código ya evita bloquear el hilo principal.

### 1. Copiar bin al target iOS

```bash
cp ../TrainingPipeline/lsm_fingerprints/fingerprints.bin \
   LSMMobileModelTesting/LSMMobileModelTesting/fingerprints.bin
```

Abrir `LSMMobileModelTesting.xcodeproj` y compilar (el archivo sincronizado se incluye automáticamente en el bundle).

### 2. Comprobar paridad Debug

1. Compilar en **Debug** (dispositivo recomendado)
2. Consola Xcode: `✓ Parity check passed (max diff …)` y `✓ Fingerprint DB: 307 clases, 1934 videos`
3. Si falla paridad: regenerar fixture con `export_fixtures.py` y alinear `LSMConstants.swift` con `constants.py`

### 3. Matriz de prueba en vivo (Fase 1)

Priorizar categorías con mejor LOO. Registrar Top-1, confianza %, Inferencia/Cosine/DTW ms del panel debug.

| # | Categoría | Señas de ejemplo | LOO Python |
|---|---|---|---|
| 1 | Festividades | Cualquier palabra indexada | 24.4% |
| 2 | Personajes_Historicos | Gobernador, etc. | 17.3% |
| 3 | Colores | 2 palabras | 8.3% |
| 4 | Familia | Abuela, Madre | 4.0% |
| 5 | Abecedario | Letras | 2.1% |

Criterio por seña: Top-1 visible; confianza >15% → punto verde «Detectando»; pills L/R/P activas.

### 4. Medir latencia

Panel debug (botón ⓘ): **Inferencia**, **Cosine**, **DTW**. Objetivo total **<200 ms** (meta <100 ms).

### 5. Release smoke

Compilar **Release** una vez: paridad omitida (`#if DEBUG`); Fase 1 debe cargar y responder igual.

### 6. Decisión según resultados

| LOO Python | Comportamiento en vivo | Acción |
|---|---|---|
| <40% (actual: 6.57%) | Coincidencias pobres esperadas | Mejorar datos/preprocessing en TrainingPipeline antes de tuning iOS |
| 40–60% | Coincidencias parciales | Iterar categorías débiles |
| ≥60% | Coincidencias estables | Baseline Fase 1 cumplido; comparar con Fase 2 |

Checklist para agentes: [AGENTS.md — Phase 1 iOS validation](./AGENTS.md#phase-1-ios-validation-checklist)

**Agregar palabra nueva (Fase 1, sin reentrenar):**

```python
from build_db import add_class
from preprocess import prepare_sequence
add_class("./fingerprints.npz", "NuevaPalabra", [prepare_sequence("video_landmarks.json")],
          export_bin_path="./fingerprints.bin")
```

**Agregar un nuevo modelo Core ML (Fase 2):**

1. Exportar desde Python:
   ```bash
   python convert_to_coreml.py --checkpoint ./runs/best_model.pt --output ./mi_modelo
   ```
2. Arrastrar `mi_modelo.mlpackage` al target en Xcode (copiar al bundle)
3. Copiar `classes.json` del run de entrenamiento al bundle
4. Registrar el modelo en `LandmarkPipeline.swift`:
   ```swift
   let AVAILABLE_MODELS: [LSMModel] = [
       LSMModel(name: "v1 — baseline", bundleName: "coreml_models", description: "Primer entrenamiento"),
       LSMModel(name: "v2 — TCN",      bundleName: "mi_modelo",     description: "Modelo TCN"),
   ]
   ```

**Archivos principales:**

| Archivo | Responsabilidad |
|---|---|
| `LandmarkPipeline.swift` | Extracción Vision, buffer, inferencia (Core ML o fingerprint) |
| `SequencePreprocessor.swift` | Normalización hombro + filtro valid (Fase 1) |
| `FingerprintExtractor.swift` | Huella 810-dim (Fase 1) |
| `FingerprintMatcher.swift` | Carga `fingerprints.bin`, cosine + DTW + voting |
| `CameraManager.swift` | Sesión AVCapture, permisos, cambio de cámara |
| `ContentView.swift` | UI: cámara, esqueleto, predicciones, debug |
| `fingerprints.bin` | Base de datos Fase 1 para inferencia |
| `classes.json` | Lista de clases Core ML (fallback) |
| `coreml_models.mlpackage` | Modelo Core ML Fase 2 |

---

## Formato de datos compartido

Ambas apps y el pipeline de Python usan el mismo esquema de features:

| Componente | Índices | Dimensiones | Descripción |
|---|---|---|---|
| Mano izquierda | 0–41 | 21 × (x, y) | 21 joints MediaPipe-style |
| Mano derecha | 42–83 | 21 × (x, y) | 21 joints |
| Pose corporal | 84–134 | 17 × (x, y, visibility) | Cabeza, hombros, codos, muñecas |

**Total: 135 valores por frame, 85 frames por secuencia.**

Coordenadas normalizadas en rango `[0, 1]`, con **Y=0 en la parte superior** (se aplica flip `1.0 - y` al guardar en JSON, consistente con `dataset.py`).

### Formato JSON de landmarks

Cada archivo `*_landmarks.json` es un array de frames:

```json
[
  {
    "frame": 0,
    "timestamp_ms": 0.0,
    "leftHand": [{"x": 0.45, "y": 0.62}, ...],
    "rightHand": [{"x": 0.55, "y": 0.58}, ...],
    "pose": [{"id": 0, "x": 0.50, "y": 0.30, "visibility": 0.95}, ...]
  }
]
```

Puntos no detectados se omiten (`null`) o se rellenan con `0.0` en el vector de inferencia.

---

## Flujo de trabajo recomendado

### 1. Preparar videos

Organizar videos con el script de Python (opcional, si vienen desordenados):

```bash
cd ../TrainingPipeline
python organize_from_json.py --json themes.json --input ./videos --output ./dataset
```

### 2. Extraer landmarks (macOS)

```bash
# Abrir LSMExtractorGUI en Xcode
# Dataset: ./LSMVideos   Salida: ./LSMVideosOutput
```

La carpeta `./LSMVideosOutput` resultante es el `--dataset` de `train.py`.

### 3. Entrenar modelo (Python)

```bash
cd ../TrainingPipeline
python train.py --dataset ./output --epochs 100 --batch 64 --output ./runs
python convert_to_coreml.py --checkpoint ./runs/best_model.pt --output ./lsm_model
```

### 4. Probar en dispositivo (iOS)

1. Copiar `lsm_model.mlpackage` y `runs/classes.json` al proyecto iOS
2. Registrar el modelo en `AVAILABLE_MODELS`
3. Compilar en dispositivo y probar señas en vivo

---

## Estructura del repositorio

```
AppLSMTests/
├── README.md
│
├── LSMExtractorGUI/                    # App macOS — extracción de landmarks
│   ├── LSMExtractorGUI.xcodeproj
│   └── LSMExtractorGUI/
│       ├── LSMExtractorApp.swift       # Entry point
│       ├── ContentView.swift           # UI principal
│       ├── LandmarkExtractor.swift     # Vision + AVFoundation
│       ├── BatchProcessor.swift        # Procesamiento en lote
│       ├── DatasetScanner.swift        # Descubrimiento y manifiesto
│       └── Models.swift                # Tipos y DatasetManifest
│
└── LSMMobileModelTesting/              # App iOS — inferencia en vivo
    ├── LSMMobileModelTesting.xcodeproj
    └── LSMMobileModelTesting/
        ├── LSMInferenceApp.swift       # Entry point
        ├── ContentView.swift           # UI: cámara + predicciones
        ├── CameraManager.swift         # AVCaptureSession
        ├── LandmarkPipeline.swift      # Vision + Core ML
        ├── classes.json                # 187 clases del modelo baseline
        └── coreml_models.mlpackage/      # Modelo Core ML incluido
```

---

## Detalles técnicos

### Vision Framework

Ambas apps usan las mismas APIs de Apple Vision:

- `VNDetectHumanHandPoseRequest` — hasta 2 manos, 21 joints cada una
- `VNDetectHumanBodyPoseRequest` — 17 puntos de pose mapeados a índices fijos

El mapeo de pose replica el de `dataset.py`: nariz, ojos, orejas, cuello, hombros, codos y muñecas. Vision no expone todos los puntos faciales de MediaPipe; los índices de ojo se rellenan con el joint más cercano disponible.

### Pipeline de inferencia móvil

```
Frame de cámara
    → Vision (manos + pose)
    → Vector 135D
    → Buffer circular (hasta 170 frames, usa últimos 85)
    → Interpolación lineal a 85 frames
    → MLMultiArray (1, 85, 135)
    → Core ML prediction
    → Softmax → Top-5 clases
```

La inferencia se ejecuta cada **15 frames** cuando el buffer tiene al menos 85 frames. Una seña se considera detectada cuando la confianza del top-1 supera **15%**.

### dataset_manifest.json

El extractor puede cargar un manifiesto existente y lo actualiza al finalizar (mismo archivo si se seleccionó uno, o `{outputRoot}/dataset_manifest.json` si no). El escaneo de disco agrega categorías/palabras/videos nuevos sin borrar entradas históricas que ya no estén en disco.

Esquema simplificado:

```json
{
  "version": 1,
  "source_root": "/path/LSMVideos",
  "output_root": "/path/LSMVideosOutput",
  "categories": [
    {
      "name": "Familia",
      "words": [
        {
          "name": "Abuela",
          "relative_path": "Familia/Abuela",
          "videos": [
            {
              "filename": "Abuela1.mp4",
              "landmarks_file": "Abuela1_landmarks.json",
              "overlay_file": "Abuela1_overlay.mp4",
              "status": "done",
              "frames": 120
            }
          ]
        }
      ]
    }
  ],
  "totals": { "categories": 11, "words": 221, "videos": 500 }
}
```

`themes.json` sigue usándose en **TrainingPipeline** para organizar videos y entrenar; el extractor ya no lo consume.

---

## Requisitos de desarrollo

| Herramienta | Versión |
|---|---|
| Xcode | Reciente (con SDK macOS 15.7+ e iOS 26.2+) |
| Swift | 5.0 |
| Apple Vision Framework | Incluido en SDK |
| Core ML | Incluido en SDK |

**Permisos:**

- **LSMExtractorGUI**: acceso a carpetas del sistema vía `NSOpenPanel` (sin entitlements especiales)
- **LSMMobileModelTesting**: permiso de cámara (`NSCameraUsageDescription`: *"Used to detect and extract sign language landmarks"*)

**Code signing:** ambos proyectos usan firma automática con team `6Q5574XLD8`. Cambiar al team propio antes de compilar en dispositivo.

---

## Estado actual

| Componente | Estado |
|---|---|
| Extracción batch macOS | Funcional — procesa videos, genera JSON + overlay opcional |
| Catálogo dataset_manifest.json | Funcional — descubrimiento por ruta, merge y persistencia |
| Inferencia iOS en vivo | Funcional — Top-5, esqueleto, debug |
| Fase 1 production bin | Copiado localmente (~95 MB, 307 clases); gitignored |
| Fase 2 LSMOutput models | CNN + TCN + 3DCNN copiados localmente; gitignored |
| Multi-modelo | 5 backends en selector (Fase 1 default, CNN recomendado NN) |
| Validación dispositivo | Pendiente Mac — B2–B5, C2–C5 |

## Relación con TrainingPipeline

Este repositorio cierra el ciclo que antes quedaba fuera del repo de Python:

| Antes (TrainingPipeline README) | Ahora (AppLSMTests) |
|---|---|
| "Extracción de landmarks no incluida" | **LSMExtractorGUI** genera `*_landmarks.json` |
| "Listo para arrastrar al proyecto Xcode" | **LSMMobileModelTesting** ya incluye el modelo y la app de prueba |

Para documentación del entrenamiento, vocabulario y exportación Core ML, ver [TrainingPipeline/README.md](../TrainingPipeline/README.md).
