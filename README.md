# AppLSMTests

Aplicaciones Swift para el proyecto **LSM (Lengua de Señas Mexicana)**. Este repositorio complementa el [TrainingPipeline](../TrainingPipeline/) de Python y cubre las dos etapas que faltaban en el flujo completo: **extracción de landmarks desde video** y **prueba de modelos en dispositivo móvil**.

## Resumen del ecosistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Flujo completo LSM                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Videos .mp4          LSMExtractorGUI          TrainingPipeline         │
│  + themes.json   ──►  (macOS, Vision)    ──►   (PyTorch)                │
│                       *_landmarks.json          train.py                │
│                       *_overlay.mp4             │                       │
│                                                   ▼                       │
│                                            best_model.pt                │
│                                                   │                       │
│                                            convert_to_coreml.py         │
│                                                   │                       │
│                                                   ▼                       │
│  LSMMobileModelTesting  ◄──  coreml_models.mlpackage + classes.json    │
│  (iOS, cámara en vivo)                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| Repositorio / app | Plataforma | Rol |
|---|---|---|
| **LSMExtractorGUI** | macOS | Extrae landmarks de manos y pose desde videos con Vision Framework |
| **TrainingPipeline** | Python | Entrena modelos y exporta a Core ML |
| **LSMMobileModelTesting** | iOS | Inferencia en tiempo real con la cámara del dispositivo |

## Proyectos incluidos

### LSMExtractorGUI — Extractor de landmarks (macOS)

App de escritorio en SwiftUI que procesa videos en lote y genera los archivos JSON que consume el pipeline de entrenamiento.

**Qué hace:**

- Lee videos `.mp4`, `.mov` o `.m4v` de una carpeta de dataset
- Detecta por frame manos (`VNDetectHumanHandPoseRequest`) y pose corporal (`VNDetectHumanBodyPoseRequest`) con Apple Vision
- Guarda secuencias de landmarks en JSON
- Genera un video con overlay visual de esqueleto para inspección
- Clasifica videos por categoría usando `themes.json` (opcional)
- Produce un reporte de batch (`_batch_report.json`)

**Requisitos:**

- macOS 15.7+
- Xcode con SDK macOS
- Apple Silicon o Intel Mac

**Uso:**

1. Abrir `LSMExtractorGUI/LSMExtractorGUI.xcodeproj` en Xcode
2. Compilar y ejecutar (⌘R)
3. Seleccionar opcionalmente `themes.json` (puede copiarse desde `TrainingPipeline/themes.json`)
4. Seleccionar la carpeta de **Dataset** con los videos
5. Seleccionar la carpeta de **Salida**
6. Pulsar **Procesar**

**Estructura de entrada esperada:**

El extractor recorre recursivamente la carpeta de dataset y acepta videos con nombres del tipo `{palabra}[N].mp4` (letras, espacios, guiones y número opcional al final).

```
dataset/
├── Categoria/
│   └── Palabra/
│       ├── Palabra1.mp4
│       └── Palabra2.mp4
└── ...
```

Si se carga `themes.json`, solo se procesan videos cuyo nombre coincide con una palabra del vocabulario. Los demás se omiten y aparecen en el log.

**Estructura de salida:**

```
output/
├── Categoria/
│   └── Palabra/
│       ├── Palabra1_landmarks.json    ← input para train.py
│       └── Palabra1_overlay.mp4       ← video con esqueleto dibujado
└── _batch_report.json                 ← resumen del procesamiento
```

**Archivos principales:**

| Archivo | Responsabilidad |
|---|---|
| `LandmarkExtractor.swift` | Detección Vision, dibujo de overlay, lectura/escritura de video |
| `BatchProcessor.swift` | Descubrimiento de videos, procesamiento en lote, estado de la UI |
| `Models.swift` | Tipos JSON, `ThemeMap`, modelos de estado (`VideoItem`, `BatchReport`) |
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
| Selector de modelo | Cambia entre modelos Core ML disponibles |
| Esqueleto (figura) | Muestra/oculta overlay de manos y pose |
| Info (ⓘ) | Muestra panel de debug |
| Rotar cámara | Alterna cámara frontal / trasera (reinicia el buffer) |
| Anillo de progreso | Indica cuántos frames hay en el buffer (0–85) |
| Reset (↺) | Limpia buffer y predicciones |

**Modelo incluido:**

El bundle incluye `coreml_models.mlpackage` (187 clases, baseline v1) y `classes.json` como respaldo. El modelo espera entrada `landmarks` con forma `(1, 85, 135)` y produce logits de salida.

**Agregar un nuevo modelo:**

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
| `LandmarkPipeline.swift` | Extracción Vision, buffer temporal, interpolación, inferencia Core ML |
| `CameraManager.swift` | Sesión AVCapture, permisos, cambio de cámara |
| `ContentView.swift` | UI: cámara, esqueleto, predicciones, debug |
| `classes.json` | Lista de clases (fallback si no están en metadata del modelo) |
| `coreml_models.mlpackage` | Modelo Core ML exportado desde PyTorch |

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
# Dataset: ./dataset   Salida: ./output
```

La carpeta `./output` resultante es el `--dataset` de `train.py`.

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
│       └── Models.swift                # Tipos y ThemeMap
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

### themes.json

El extractor usa el mismo archivo de vocabulario que el pipeline de Python. Para cada video, normaliza el nombre del archivo y lo busca en un mapa construido a partir de:

- Nombre de carpeta destino (`videoUrl`)
- Nombre de visualización (`name`)
- Variante sin paréntesis

Esto permite emparejar archivos como `Abuela2.mp4` con la entrada `"Abuela"` del JSON.

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
| Extracción batch macOS | Funcional — procesa videos, genera JSON + overlay |
| Integración con themes.json | Funcional — categorización automática |
| Inferencia iOS en vivo | Funcional — Top-5, esqueleto, debug |
| Modelo incluido | `coreml_models` — 187 clases, 85×135 input |
| Multi-modelo | Soporte en código (`AVAILABLE_MODELS`); solo v1 incluido en bundle |

## Relación con TrainingPipeline

Este repositorio cierra el ciclo que antes quedaba fuera del repo de Python:

| Antes (TrainingPipeline README) | Ahora (AppLSMTests) |
|---|---|
| "Extracción de landmarks no incluida" | **LSMExtractorGUI** genera `*_landmarks.json` |
| "Listo para arrastrar al proyecto Xcode" | **LSMMobileModelTesting** ya incluye el modelo y la app de prueba |

Para documentación del entrenamiento, vocabulario y exportación Core ML, ver [TrainingPipeline/README.md](../TrainingPipeline/README.md).
