import Foundation
import Vision
import CoreImage
import CoreML
import Combine
import Accelerate

// ── Feature constants ─────────────────────────────────────────────────────────
let N_FRAMES     = 85
let FEATURE_DIM  = 135
let N_LEFT_HAND  = 42
let N_RIGHT_HAND = 42
let N_POSE       = 51

// ── Hand joint order ──────────────────────────────────────────────────────────
let HAND_JOINTS: [VNHumanHandPoseObservation.JointName] = [
    .wrist,
    .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
    .indexMCP, .indexPIP, .indexDIP, .indexTip,
    .middleMCP, .middlePIP, .middleDIP, .middleTip,
    .ringMCP, .ringPIP, .ringDIP, .ringTip,
    .littleMCP, .littlePIP, .littleDIP, .littleTip,
]

let HAND_CONNECTIONS: [(Int, Int)] = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

// ── Pose joint map ────────────────────────────────────────────────────────────
let POSE_JOINT_MAP: [Int: VNHumanBodyPoseObservation.JointName] = [
    0: .nose,
    1: .leftEye,  2: .leftEye,  3: .leftEye,
    4: .rightEye, 5: .rightEye, 6: .rightEye,
    7: .leftEar,  8: .rightEar,
    9: .neck,     10: .neck,
    11: .leftShoulder,  12: .rightShoulder,
    13: .leftElbow,     14: .rightElbow,
    15: .leftWrist,     16: .rightWrist,
]

let POSE_CONNECTIONS: [(Int, Int)] = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
]

// ── Skeleton data (published for overlay) ─────────────────────────────────────

struct SkeletonData {
    var leftHand:  [CGPoint] = []   // 21 points, normalized 0-1
    var rightHand: [CGPoint] = []
    var pose:      [CGPoint] = []   // 17 points
}

// ── Model descriptor ──────────────────────────────────────────────────────────

struct LSMModel: Identifiable, Equatable {
    let id          = UUID()
    let name:        String
    let bundleName:  String
    let description: String
    let isFingerprint: Bool

    init(name: String, bundleName: String, description: String, isFingerprint: Bool = false) {
        self.name = name
        self.bundleName = bundleName
        self.description = description
        self.isFingerprint = isFingerprint
    }

    static func == (lhs: LSMModel, rhs: LSMModel) -> Bool {
        lhs.bundleName == rhs.bundleName && lhs.isFingerprint == rhs.isFingerprint
    }
}

// ── MODELOS DISPONIBLES ───────────────────────────────────────────────────────
let AVAILABLE_MODELS: [LSMModel] = [
    LSMModel(
        name:        "Fase 1 — Fingerprint+DTW",
        bundleName:  "fingerprints",
        description: "Huella estadística + DTW (sin entrenamiento)",
        isFingerprint: true
    ),
    LSMModel(
        name:        "Fase 2 — Core ML v1",
        bundleName:  "coreml_models",
        description: "Red neuronal entrenada"
    ),
]

// ── Landmark extraction ───────────────────────────────────────────────────────

struct FrameData {
    var vector:   [Float]     // 135 values
    var skeleton: SkeletonData
}

func extractFrameData(from pixelBuffer: CVPixelBuffer) -> FrameData {
    var vec      = [Float](repeating: 0, count: FEATURE_DIM)
    var skeleton = SkeletonData()

    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                        orientation: .up, options: [:])

    // ── Hands ─────────────────────────────────────────────────────────────────
    let handRequest = VNDetectHumanHandPoseRequest()
    handRequest.maximumHandCount = 2
    try? handler.perform([handRequest])

    if let observations = handRequest.results {
        for obs in observations {
            let isLeft = obs.chirality == .left
            let offset = isLeft ? 0 : N_LEFT_HAND
            var points: [CGPoint] = []

            for (i, joint) in HAND_JOINTS.enumerated() {
                if let p = try? obs.recognizedPoint(joint), p.confidence > 0.1 {
                    vec[offset + i * 2]     = Float(p.location.x)
                    vec[offset + i * 2 + 1] = Float(1.0 - p.location.y)
                    points.append(CGPoint(x: p.location.x, y: p.location.y))
                } else {
                    points.append(.zero)
                }
            }
            if isLeft { skeleton.leftHand  = points }
            else       { skeleton.rightHand = points }
        }
    }

    // ── Pose ──────────────────────────────────────────────────────────────────
    let poseRequest = VNDetectHumanBodyPoseRequest()
    try? handler.perform([poseRequest])

    var posePoints = [CGPoint](repeating: .zero, count: 17)
    if let obs = poseRequest.results?.first {
        let poseOffset = N_LEFT_HAND + N_RIGHT_HAND
        for idx in 0..<17 {
            guard let jointName = POSE_JOINT_MAP[idx],
                  let p = try? obs.recognizedPoint(jointName),
                  p.confidence > 0.1 else { continue }
            let base = poseOffset + idx * 3
            vec[base]     = Float(p.location.x)
            vec[base + 1] = Float(1.0 - p.location.y)
            vec[base + 2] = Float(p.confidence)
            posePoints[idx] = CGPoint(x: p.location.x, y: p.location.y)
        }
    }
    skeleton.pose = posePoints

    return FrameData(vector: vec, skeleton: skeleton)
}

// ── Temporal interpolation ────────────────────────────────────────────────────

func interpolateSequence(_ seq: [[Float]], to targetLen: Int) -> [[Float]] {
    let T = seq.count
    guard T > 0 else {
        return Array(repeating: [Float](repeating: 0, count: FEATURE_DIM), count: targetLen)
    }
    if T == targetLen { return seq }
    return (0..<targetLen).map { newIdx in
        let t     = Double(newIdx) / Double(targetLen - 1) * Double(T - 1)
        let lo    = min(Int(t), T - 1)
        let hi    = min(lo + 1, T - 1)
        let alpha = Float(t - Double(lo))
        return zip(seq[lo], seq[hi]).map { a, b in a + alpha * (b - a) }
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────

func softmax(_ logits: [Float]) -> [Float] {
    let maxVal = logits.max() ?? 0
    let exps   = logits.map { expf($0 - maxVal) }
    let sum    = exps.reduce(0, +)
    return exps.map { $0 / sum }
}

// ── Prediction ────────────────────────────────────────────────────────────────

struct Prediction: Identifiable {
    let id         = UUID()
    let label:      String
    let confidence: Float
}

// ── Debug info ────────────────────────────────────────────────────────────────

struct DebugInfo {
    var fps:              Double = 0
    var inferenceMs:      Double = 0
    var cosineMs:         Double = 0
    var dtwMs:            Double = 0
    var framesInBuffer:   Int    = 0
    var handLeft:         Bool   = false
    var handRight:        Bool   = false
    var poseDetected:     Bool   = false
    var modelName:        String = "-"
    var nClasses:         Int    = 0
    var topConfidence:    Float  = 0
}

// ── Main pipeline ─────────────────────────────────────────────────────────────

@MainActor
class LandmarkPipeline: ObservableObject {

    @Published var predictions:  [Prediction] = []
    @Published var isDetecting:  Bool = false
    @Published var skeleton:     SkeletonData = SkeletonData()
    @Published var debug:        DebugInfo = DebugInfo()
    @Published var activeModel:  LSMModel? = nil
    @Published var loadError:    String? = nil

    private var frameBuffer: [[Float]] = []
    private var model:   MLModel?
    private var fingerprintMatcher: FingerprintMatcher?
    private var classes: [String] = []
    private var useFingerprint = false

    private let inferenceStride = 15

    // FPS tracking
    private var frameCount = 0
    private var lastFPSTime = Date()

    init() {
        #if DEBUG
        Phase1ParityCheck.verify()
        #endif
        if let first = AVAILABLE_MODELS.first { loadModel(first) }
    }

    // MARK: - Model switching

    func loadModel(_ descriptor: LSMModel) {
        reset()
        loadError   = nil
        model       = nil
        fingerprintMatcher = nil
        classes     = []
        useFingerprint = descriptor.isFingerprint
        activeModel = descriptor

        if descriptor.isFingerprint {
            let matcher = FingerprintMatcher(bundleName: descriptor.bundleName)
            if let err = matcher.loadError {
                loadError = err
                return
            }
            fingerprintMatcher = matcher
            classes = matcher.classes
            debug.modelName = descriptor.name
            debug.nClasses  = classes.count
            print("✓ Fingerprint DB: \(classes.count) clases, \(matcher.fingerprints.count) videos")
            return
        }

        let url = Bundle.main.url(forResource: descriptor.bundleName,
                                  withExtension: "mlpackage")
               ?? Bundle.main.url(forResource: descriptor.bundleName,
                                  withExtension: "mlmodelc")

        guard let modelURL = url else {
            loadError = "No se encontró \(descriptor.bundleName).mlpackage"
            return
        }

        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            model = try MLModel(contentsOf: modelURL, configuration: config)

            // Print ALL metadata for diagnosis
            let desc = model!.modelDescription
            print("📐 Inputs:")
            for (name, feat) in desc.inputDescriptionsByName {
                print("   '\(name)': \(feat.type)")
            }
            print("📐 Outputs:")
            for (name, feat) in desc.outputDescriptionsByName {
                print("   '\(name)': \(feat.type)")
            }
            print("📐 Metadata keys:")
            for (key, val) in desc.metadata {
                let valStr = "\(val)".prefix(120)
                print("   '\(key)' → \(valStr)")
            }

            // coremltools guarda user_defined_metadata con prefijo en iOS
            // Iterar todas las claves para encontrar "classes"
            var found = false
            for (key, value) in desc.metadata {
                let keyStr = "\(key)"
                if keyStr.hasSuffix("classes") || keyStr == "classes" {
                    let valStr = "\(value)"
                    if let data = valStr.data(using: .utf8),
                       let decoded = try? JSONDecoder().decode([String].self, from: data) {
                        classes = decoded
                        print("✓ Clases desde metadata '\(keyStr)': \(classes.count)")
                        found = true
                        break
                    }
                }
            }

            // Fallback: classes.json en el bundle
            if !found {
                let jsonNames = [descriptor.bundleName + "_classes", "classes"]
                for jsonName in jsonNames {
                    if let jsonURL = Bundle.main.url(forResource: jsonName,
                                                    withExtension: "json"),
                       let data = try? Data(contentsOf: jsonURL),
                       let decoded = try? JSONDecoder().decode([String].self, from: data) {
                        classes = decoded
                        print("✓ Clases desde \(jsonName).json: \(classes.count)")
                        found = true
                        break
                    }
                }
            }

            if !found {
                print("⚠ No se encontraron clases — agrega classes.json al bundle")
            }

            debug.modelName = descriptor.name
            debug.nClasses  = classes.count
            print("✓ Modelo: \(descriptor.name) — \(classes.count) clases")
        } catch {
            loadError = error.localizedDescription
        }
    }

    // MARK: - Frame processing

    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        let frameData = extractFrameData(from: pixelBuffer)

        // Update skeleton for overlay
        skeleton = frameData.skeleton

        // Update debug detection flags
        debug.handLeft     = !frameData.skeleton.leftHand.isEmpty &&
                              frameData.skeleton.leftHand.contains { $0 != .zero }
        debug.handRight    = !frameData.skeleton.rightHand.isEmpty &&
                              frameData.skeleton.rightHand.contains { $0 != .zero }
        debug.poseDetected = frameData.skeleton.pose.contains { $0 != .zero }

        // Buffer
        frameBuffer.append(frameData.vector)
        if frameBuffer.count > N_FRAMES * 2 {
            frameBuffer.removeFirst(frameBuffer.count - N_FRAMES * 2)
        }
        debug.framesInBuffer = min(frameBuffer.count, N_FRAMES)

        // FPS
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFPSTime)
        if elapsed >= 1.0 {
            debug.fps   = Double(frameCount) / elapsed
            frameCount  = 0
            lastFPSTime = now
        }

        // Inference
        if frameBuffer.count >= N_FRAMES && frameBuffer.count % inferenceStride == 0 {
            runInference()
        }
    }

    // MARK: - Inference

    private func runInference() {
        let t0 = Date()

        if useFingerprint {
            runFingerprintInference(startTime: t0)
            return
        }

        guard let model else {
            print("⚠ runInference: modelo es nil")
            return
        }
        guard !classes.isEmpty else {
            print("⚠ runInference: classes está vacío")
            return
        }

        let recent       = Array(frameBuffer.suffix(N_FRAMES))
        let interpolated = interpolateSequence(recent, to: N_FRAMES)
        let flat         = interpolated.flatMap { $0 }

        guard let mlArray = try? MLMultiArray(
            shape: [1, N_FRAMES as NSNumber, FEATURE_DIM as NSNumber],
            dataType: .float32) else {
            print("⚠ runInference: fallo al crear MLMultiArray")
            return
        }
        for (i, val) in flat.enumerated() { mlArray[i] = NSNumber(value: val) }

        guard let input = try? MLDictionaryFeatureProvider(
            dictionary: ["landmarks": mlArray]) else {
            print("⚠ runInference: fallo al crear input")
            return
        }

        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: input)
        } catch {
            print("⚠ runInference: error en predicción: \(error)")
            return
        }

        // Intentar obtener logits — puede tener nombre distinto
        let featureNames = output.featureNames
        print("🔍 Output features: \(featureNames)")

        guard let logitsArray = output.featureValue(for: "logits")?.multiArrayValue else {
            print("⚠ runInference: no se encontró feature 'logits'. Disponibles: \(featureNames)")
            // Intentar con el primer feature disponible
            if let firstName = featureNames.first,
               let arr = output.featureValue(for: firstName)?.multiArrayValue {
                print("  Usando '\(firstName)' con \(arr.count) valores")
                let logits = (0..<arr.count).map { Float(truncating: arr[$0]) }
                let probs  = softmax(logits)
                let top5   = probs.enumerated()
                    .sorted { $0.element > $1.element }
                    .prefix(5)
                    .map { i, p in Prediction(
                        label:      i < classes.count ? classes[i] : "clase_\(i)",
                        confidence: p) }
                debug.inferenceMs   = Date().timeIntervalSince(t0) * 1000
                debug.topConfidence = top5.first?.confidence ?? 0
                isDetecting = (top5.first?.confidence ?? 0) > 0.15
                predictions = top5
            }
            return
        }

        let logits = (0..<logitsArray.count).map { Float(truncating: logitsArray[$0]) }
        let probs  = softmax(logits)
        let top5   = probs.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(5)
            .map { i, p in Prediction(
                label:      i < classes.count ? classes[i] : "clase_\(i)",
                confidence: p) }

        debug.inferenceMs  = Date().timeIntervalSince(t0) * 1000
        debug.topConfidence = top5.first?.confidence ?? 0
        isDetecting = (top5.first?.confidence ?? 0) > 0.15
        predictions = top5
    }

    private func runFingerprintInference(startTime t0: Date) {
        guard let matcher = fingerprintMatcher, matcher.isLoaded else {
            loadError = fingerprintMatcher?.loadError ?? "Fingerprint matcher no cargado"
            return
        }

        guard let sequence = SequencePreprocessor.prepareSequence(from: frameBuffer) else {
            return
        }

        let tMatch = Date()
        let results = matcher.match(querySequence: sequence)
        debug.inferenceMs = Date().timeIntervalSince(t0) * 1000
        debug.dtwMs = Date().timeIntervalSince(tMatch) * 1000

        let top5 = results.prefix(5).map {
            Prediction(label: $0.label, confidence: $0.confidence)
        }

        debug.topConfidence = top5.first?.confidence ?? 0
        isDetecting = (top5.first?.confidence ?? 0) > LSMConstants.confidenceThreshold
        predictions = Array(top5)
    }

    // MARK: - Reset

    func reset() {
        frameBuffer  = []
        predictions  = []
        isDetecting  = false
        skeleton     = SkeletonData()
        debug.framesInBuffer = 0
        debug.topConfidence  = 0
    }
}
