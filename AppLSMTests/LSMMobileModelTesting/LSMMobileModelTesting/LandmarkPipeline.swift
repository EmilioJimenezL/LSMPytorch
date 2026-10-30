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

enum ModelFamily: String, CaseIterable, Identifiable {
    case fingerprint
    case cnn
    case tcn
    case cnn3d
    case legacy

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fingerprint: return "Fase 1"
        case .cnn: return "CNN"
        case .tcn: return "TCN"
        case .cnn3d: return "3D CNN"
        case .legacy: return "Legacy"
        }
    }

    var sortIndex: Int {
        switch self {
        case .fingerprint: return 0
        case .cnn: return 1
        case .tcn: return 2
        case .cnn3d: return 3
        case .legacy: return 4
        }
    }
}

enum SignSet: String, CaseIterable, Identifiable {
    case lsmoutput307
    case min12
    case min7
    case min5
    case legacy187

    var id: String { rawValue }

    var title: String {
        switch self {
        case .lsmoutput307: return "LSMOutput"
        case .min12: return "Al menos 12 videos"
        case .min7: return "Al menos 7 videos"
        case .min5: return "Al menos 5 videos"
        case .legacy187: return "Legacy"
        }
    }

    var detail: String {
        switch self {
        case .lsmoutput307: return "307 clases, al menos 2 videos"
        case .min12: return "28 clases"
        case .min7: return "54 clases, incluye las de 12 o más"
        case .min5: return "286 clases, incluye las de 7 y 12"
        case .legacy187: return "187 clases, anterior a LSMOutput"
        }
    }
}

struct LSMModel: Identifiable, Equatable {
    let id          = UUID()
    let name:        String
    let bundleName:  String
    /// Bundle resource base name for `{classesBundleName}.json` (without extension).
    /// Fase 1 embeds labels in `fingerprints.bin`; leave nil for fingerprint models.
    let classesBundleName: String?
    let description: String
    let isFingerprint: Bool
    let family: ModelFamily
    let signSet: SignSet

    init(name: String, bundleName: String, classesBundleName: String? = nil,
         description: String, isFingerprint: Bool = false,
         family: ModelFamily, signSet: SignSet) {
        self.name = name
        self.bundleName = bundleName
        self.classesBundleName = classesBundleName
        self.description = description
        self.isFingerprint = isFingerprint
        self.family = family
        self.signSet = signSet
    }

    static func == (lhs: LSMModel, rhs: LSMModel) -> Bool {
        lhs.bundleName == rhs.bundleName
            && lhs.classesBundleName == rhs.classesBundleName
            && lhs.isFingerprint == rhs.isFingerprint
            && lhs.family == rhs.family
            && lhs.signSet == rhs.signSet
    }
}

// ── MODELOS DISPONIBLES ───────────────────────────────────────────────────────
let AVAILABLE_MODELS: [LSMModel] = [
    LSMModel(
        name:        "Fase 1 — Fingerprint+DTW",
        bundleName:  "fingerprints",
        description: "307 clases, LOO Top-1 6.57%",
        isFingerprint: true,
        family: .fingerprint,
        signSet: .lsmoutput307
    ),
    LSMModel(
        name:        "Fase 2 — CNN LSMOutput",
        bundleName:  "runs_lsmoutput_cnn_lsmoutput",
        classesBundleName: "runs_lsmoutput_cnn_classes",
        description: "307 clases, val Top-1 47.16% (recomendado)",
        family: .cnn,
        signSet: .lsmoutput307
    ),
    LSMModel(
        name:        "Fase 2 — TCN LSMOutput",
        bundleName:  "runs_lsmoutput_tcn_lsmoutput",
        classesBundleName: "runs_lsmoutput_tcn_classes",
        description: "307 clases, val Top-1 44.78%",
        family: .tcn,
        signSet: .lsmoutput307
    ),
    LSMModel(
        name:        "Fase 2 — 3D CNN LSMOutput",
        bundleName:  "runs_lsmoutput_3dcnn_lsmoutput",
        classesBundleName: "runs_lsmoutput_3dcnn_classes",
        description: "307 clases, val Top-1 15.52%",
        family: .cnn3d,
        signSet: .lsmoutput307
    ),
    LSMModel(
        name:        "CNN — al menos 12 videos",
        bundleName:  "runs_m4_min12_cnn_lsmoutput",
        classesBundleName: "runs_m4_min12_cnn_classes",
        description: "28 clases, val Top-1 48.21%",
        family: .cnn,
        signSet: .min12
    ),
    LSMModel(
        name:        "TCN — al menos 12 videos",
        bundleName:  "runs_m4_min12_tcn_lsmoutput",
        classesBundleName: "runs_m4_min12_tcn_classes",
        description: "28 clases, val Top-1 35.71%",
        family: .tcn,
        signSet: .min12
    ),
    LSMModel(
        name:        "3D CNN — al menos 12 videos",
        bundleName:  "runs_m4_min12_3dcnn_lsmoutput",
        classesBundleName: "runs_m4_min12_3dcnn_classes",
        description: "28 clases, val Top-1 3.57%",
        family: .cnn3d,
        signSet: .min12
    ),
    LSMModel(
        name:        "CNN — al menos 7 videos",
        bundleName:  "runs_m4_min7_cnn_lsmoutput",
        classesBundleName: "runs_m4_min7_cnn_classes",
        description: "54 clases, val Top-1 52.44%",
        family: .cnn,
        signSet: .min7
    ),
    LSMModel(
        name:        "TCN — al menos 7 videos",
        bundleName:  "runs_m4_min7_tcn_lsmoutput",
        classesBundleName: "runs_m4_min7_tcn_classes",
        description: "54 clases, val Top-1 39.02%",
        family: .tcn,
        signSet: .min7
    ),
    LSMModel(
        name:        "3D CNN — al menos 7 videos",
        bundleName:  "runs_m4_min7_3dcnn_lsmoutput",
        classesBundleName: "runs_m4_min7_3dcnn_classes",
        description: "54 clases, val Top-1 15.85%",
        family: .cnn3d,
        signSet: .min7
    ),
    LSMModel(
        name:        "CNN — al menos 5 videos",
        bundleName:  "runs_m4_min5_cnn_lsmoutput",
        classesBundleName: "runs_m4_min5_cnn_classes",
        description: "286 clases, val Top-1 47.45%",
        family: .cnn,
        signSet: .min5
    ),
    LSMModel(
        name:        "TCN — al menos 5 videos",
        bundleName:  "runs_m4_min5_tcn_lsmoutput",
        classesBundleName: "runs_m4_min5_tcn_classes",
        description: "286 clases, val Top-1 47.45%",
        family: .tcn,
        signSet: .min5
    ),
    LSMModel(
        name:        "3D CNN — al menos 5 videos",
        bundleName:  "runs_m4_min5_3dcnn_lsmoutput",
        classesBundleName: "runs_m4_min5_3dcnn_classes",
        description: "286 clases, val Top-1 13.69%",
        family: .cnn3d,
        signSet: .min5
    ),
    LSMModel(
        name:        "Fase 2 — Legacy v1",
        bundleName:  "coreml_models",
        classesBundleName: "classes",
        description: "187 clases, pre-LSMOutput (val 9.68%)",
        family: .legacy,
        signSet: .legacy187
    ),
]

// MARK: - Class label loading

private func isClassesMetadataKey(_ key: String) -> Bool {
    key == "classes" || key.hasSuffix(".classes")
}

private func classesFromMetadata(_ metadata: [AnyHashable: Any]) -> [String]? {
    for (key, value) in metadata {
        let keyStr = "\(key)"
        guard isClassesMetadataKey(keyStr) else { continue }
        let valStr: String?
        if let s = value as? String {
            valStr = s
        } else if let s = value as? NSString {
            valStr = s as String
        } else {
            valStr = "\(value)"
        }
        guard let valStr,
              let data = valStr.data(using: .utf8),
              let decoded = try? JSONDecoder().decode([String].self, from: data) else { continue }
        print("✓ Clases desde metadata '\(keyStr)': \(decoded.count)")
        return decoded
    }
    return nil
}

private func classesFromBundle(_ resourceName: String) -> [String]? {
    guard let jsonURL = Bundle.main.url(forResource: resourceName, withExtension: "json"),
          let data = try? Data(contentsOf: jsonURL),
          let decoded = try? JSONDecoder().decode([String].self, from: data) else { return nil }
    print("✓ Clases desde \(resourceName).json: \(decoded.count)")
    return decoded
}

private func classLabelCandidates(for descriptor: LSMModel) -> [String] {
    var names: [String] = []
    if let explicit = descriptor.classesBundleName {
        names.append(explicit)
    }
    // e.g. runs_lsmoutput_cnn_lsmoutput → runs_lsmoutput_cnn_classes
    if descriptor.bundleName.hasSuffix("_lsmoutput") {
        let base = String(descriptor.bundleName.dropLast("_lsmoutput".count))
        names.append(base + "_classes")
    }
    names.append(descriptor.bundleName + "_classes")
    var seen = Set<String>()
    return names.filter { seen.insert($0).inserted }
}

private func loadClassLabels(for descriptor: LSMModel,
                             modelDescription: MLModelDescription?) -> [String] {
    if let metadata = modelDescription?.metadata,
       let fromMetadata = classesFromMetadata(metadata) {
        return fromMetadata
    }
    for name in classLabelCandidates(for: descriptor) {
        if let fromBundle = classesFromBundle(name) {
            return fromBundle
        }
    }
    return []
}

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

struct Prediction: Identifiable, Equatable {
    let id         = UUID()
    let label:      String
    let confidence: Float
}

// ── Model load state ──────────────────────────────────────────────────────────

enum ModelLoadPhase: Equatable {
    case idle
    case loading(modelName: String, progress: Double)
    case succeeded(modelName: String, warning: String?)
    case failed(modelName: String, message: String, keptModelName: String?)

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }
}

private struct LoadedModelState {
    let model: MLModel?
    let fingerprintMatcher: FingerprintMatcher?
    let classes: [String]
    let useFingerprint: Bool
    let warning: String?
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
    @Published var pendingGrading: GradingRound?
    @Published var isDetecting:  Bool = false
    @Published var skeleton:     SkeletonData = SkeletonData()
    @Published var debug:        DebugInfo = DebugInfo()
    @Published var activeModel:  LSMModel? = nil
    @Published var loadError:    String? = nil
    @Published var modelLoadPhase: ModelLoadPhase = .idle
    @Published private(set) var isBufferRecording = false

    var isLoadingModel: Bool { modelLoadPhase.isLoading }

    var availableClassLabels: [String] { classes }

    private var frameBuffer: [[Float]] = []
    private var model:   MLModel?
    private var fingerprintMatcher: FingerprintMatcher?
    private var classes: [String] = []
    private var useFingerprint = false
    private var loadTask: Task<Void, Never>?
    private var inferenceTask: Task<Void, Never>?
    private var inferenceInFlight = false

    private let inferenceStride = 15

    // FPS tracking
    private var frameCount = 0
    private var lastFPSTime = Date()

    init() {
        #if DEBUG
        Phase1ParityCheck.verify()
        #endif
    }

    // MARK: - Model switching

    func loadModel(_ descriptor: LSMModel) {
        if activeModel == descriptor, !modelLoadPhase.isLoading {
            switch modelLoadPhase {
            case .failed:
                break
            default:
                return
            }
        }

        loadTask?.cancel()
        inferenceTask?.cancel()
        inferenceInFlight = false
        loadError = nil

        let keptName = activeModel?.name
        modelLoadPhase = .loading(modelName: descriptor.name, progress: 0)

        loadTask = Task { @MainActor in
            let outcome = await fetchModelState(descriptor: descriptor) { [weak self] progress in
                Task { @MainActor in
                    guard let self, !Task.isCancelled else { return }
                    self.modelLoadPhase = .loading(modelName: descriptor.name, progress: progress)
                }
            }

            guard !Task.isCancelled else { return }

            switch outcome {
            case .success(let state):
                commitLoadedState(state, descriptor: descriptor)
                modelLoadPhase = .succeeded(modelName: descriptor.name, warning: state.warning)
            case .failure(let message):
                loadError = message
                modelLoadPhase = .failed(
                    modelName: descriptor.name,
                    message: message,
                    keptModelName: keptName
                )
            }
        }
    }

    private enum ModelFetchOutcome {
        case success(LoadedModelState)
        case failure(String)
    }

    private func fetchModelState(
        descriptor: LSMModel,
        progress: @escaping (Double) -> Void
    ) async -> ModelFetchOutcome {
        if descriptor.isFingerprint {
            return await fetchFingerprintState(descriptor: descriptor, progress: progress)
        }
        return await fetchCoreMLState(descriptor: descriptor, progress: progress)
    }

    private func fetchFingerprintState(
        descriptor: LSMModel,
        progress: @escaping (Double) -> Void
    ) async -> ModelFetchOutcome {
        let bundleName = descriptor.bundleName
        let matcher = await Task.detached(priority: .userInitiated) {
            FingerprintMatcher(bundleName: bundleName, progress: progress)
        }.value

        if let err = matcher.loadError {
            return .failure(err)
        }
        guard matcher.isLoaded else {
            return .failure("No se pudo cargar la base de fingerprints")
        }
        if matcher.classes.isEmpty {
            return .failure("La base de fingerprints no contiene etiquetas de clase")
        }

        print("✓ Fingerprint DB: \(matcher.classes.count) clases, \(matcher.fingerprints.count) videos")
        return .success(LoadedModelState(
            model: nil,
            fingerprintMatcher: matcher,
            classes: matcher.classes,
            useFingerprint: true,
            warning: nil
        ))
    }

    private func fetchCoreMLState(
        descriptor: LSMModel,
        progress: @escaping (Double) -> Void
    ) async -> ModelFetchOutcome {
        progress(0.08)

        let modelURL = Bundle.main.url(forResource: descriptor.bundleName, withExtension: "mlpackage")
            ?? Bundle.main.url(forResource: descriptor.bundleName, withExtension: "mlmodelc")

        guard let modelURL else {
            return .failure("No se encontró \(descriptor.bundleName).mlpackage en el bundle")
        }

        progress(0.18)

        let loadedModel: MLModel
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            loadedModel = try await Task.detached(priority: .userInitiated) {
                try MLModel(contentsOf: modelURL, configuration: config)
            }.value
        } catch {
            return .failure(error.localizedDescription)
        }

        progress(0.72)

        let desc = loadedModel.modelDescription
        let loadedClasses = loadClassLabels(for: descriptor, modelDescription: desc)
        var warning: String?

        if loadedClasses.isEmpty {
            let tried = classLabelCandidates(for: descriptor).joined(separator: ", ")
            return .failure("No se encontraron etiquetas de clase (probado: \(tried).json)")
        }

        if let expected = expectedLogitCount(from: desc), loadedClasses.count != expected {
            warning = "Etiquetas (\(loadedClasses.count)) no coinciden con salida del modelo (\(expected))"
            print("⚠ \(warning!)")
        }

        progress(1.0)
        print("✓ Modelo: \(descriptor.name) — \(loadedClasses.count) clases")

        return .success(LoadedModelState(
            model: loadedModel,
            fingerprintMatcher: nil,
            classes: loadedClasses,
            useFingerprint: false,
            warning: warning
        ))
    }

    private func expectedLogitCount(from desc: MLModelDescription) -> Int? {
        for (_, feat) in desc.outputDescriptionsByName {
            guard let constraint = feat.multiArrayConstraint else { continue }
            let shape = constraint.shape.map(\.intValue)
            if let last = shape.last, last > 1 { return last }
        }
        return nil
    }

    private func commitLoadedState(_ state: LoadedModelState, descriptor: LSMModel) {
        model = state.model
        fingerprintMatcher = state.fingerprintMatcher
        classes = state.classes
        useFingerprint = state.useFingerprint
        activeModel = descriptor
        loadError = nil
        debug.modelName = descriptor.name
        debug.nClasses = state.classes.count
        reset()
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

        guard isBufferRecording else {
            debug.framesInBuffer = 0
            return
        }

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

        let recent = Array(frameBuffer.suffix(N_FRAMES))
        guard let sequence = SequencePreprocessor.prepareSequence(from: recent) else {
            return
        }
        let flat = sequence.flatMap { $0 }

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
                applyInferenceResults(Array(top5), inferenceMs: Date().timeIntervalSince(t0) * 1000)
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

        applyInferenceResults(Array(top5), inferenceMs: Date().timeIntervalSince(t0) * 1000)
    }

    private func applyInferenceResults(_ top5: [Prediction], inferenceMs: Double) {
        let results = Array(top5.prefix(5))
        debug.inferenceMs = inferenceMs
        debug.topConfidence = results.first?.confidence ?? 0
        isDetecting = (results.first?.confidence ?? 0) > 0.15
        predictions = results
        offerGradingRound(predictions: results, inferenceMs: inferenceMs)
    }

    private func offerGradingRound(predictions: [Prediction], inferenceMs: Double) {
        guard pendingGrading == nil, !predictions.isEmpty, activeModel != nil else { return }
        pendingGrading = GradingRound(
            predictions: predictions,
            modelName: debug.modelName,
            inferenceMs: inferenceMs,
            capturedAt: Date()
        )
    }

    func acknowledgeGradingRound() {
        pendingGrading = nil
        predictions = []
        isDetecting = false
        debug.topConfidence = 0
    }

    private func runFingerprintInference(startTime t0: Date) {
        guard let matcher = fingerprintMatcher, matcher.isLoaded else {
            if !isLoadingModel {
                loadError = fingerprintMatcher?.loadError ?? "Fingerprint matcher no cargado"
            }
            return
        }
        guard !inferenceInFlight else { return }

        guard let sequence = SequencePreprocessor.prepareSequence(from: frameBuffer) else {
            return
        }

        inferenceInFlight = true
        inferenceTask = Task.detached(priority: .userInitiated) { [weak self] in
            let output = matcher.match(querySequence: sequence)
            let totalMs = Date().timeIntervalSince(t0) * 1000
            let top5 = output.results.prefix(5).map {
                Prediction(label: $0.label, confidence: $0.confidence)
            }

            await MainActor.run {
                guard let self else { return }
                self.inferenceInFlight = false
                self.debug.cosineMs = output.cosineMs
                self.debug.dtwMs = output.dtwMs
                self.applyInferenceResults(Array(top5), inferenceMs: totalMs)
            }
        }
    }

    // MARK: - Reset

    func commenceBufferRecording() {
        guard activeModel != nil else { return }
        isBufferRecording = true
        reset()
    }

    func reset() {
        frameBuffer  = []
        predictions  = []
        pendingGrading = nil
        isDetecting  = false
        skeleton     = SkeletonData()
        debug.framesInBuffer = 0
        debug.topConfidence  = 0
    }
}
