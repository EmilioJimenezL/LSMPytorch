import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var pipeline   = LandmarkPipeline()
    @StateObject private var camera     = CameraManager()
    @StateObject private var gradingStore = GradingSessionStore()
    @State private var showDebug        = false
    @State private var showSkeleton     = true
    @State private var showOtraSearch   = false
    @State private var showDeleteAlert  = false
    @State private var exportPayload: ExportSharePayload?
    @State private var isPreparingExport = false
    @State private var lastLoggedMessage: String?
    private let bridge = PipelineBridgeHolder()

    var body: some View {
        ZStack {
            // ── Camera ────────────────────────────────────────────────────
            CameraPreview(camera: camera,
                          isFrontCamera: camera.isFrontCamera,
                          configurationToken: camera.configurationToken)
                .ignoresSafeArea()

            // ── Skeleton overlay ──────────────────────────────────────────
            if showSkeleton {
                SkeletonOverlay(skeleton: pipeline.skeleton,
                                mirrorX: camera.isFrontCamera)
                    .ignoresSafeArea()
            }

            // ── UI ────────────────────────────────────────────────────────
            VStack(spacing: 0) {
                topBar
                    .padding(.top, 8)
                    .padding(.horizontal, 12)

                Spacer()

                if showDebug {
                    debugPanel
                        .padding(.horizontal, 12)
                        .padding(.bottom, 8)
                }

                predictionsPanel
                    .padding(.horizontal, 12)
                    .padding(.bottom, 32)
            }

            if pipeline.modelLoadPhase != .idle {
                ModelLoadingOverlay(
                    phase: pipeline.modelLoadPhase,
                    onDismiss: { pipeline.dismissModelLoadOverlay() }
                )
                .transition(.opacity.combined(with: .scale(scale: 0.96)))
                .zIndex(10)
            }

            if isPreparingExport {
                ExportPreparingOverlay()
                    .zIndex(11)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: pipeline.modelLoadPhase)
        .onChange(of: pipeline.pendingGrading?.id) { _, newID in
            if newID != nil { lastLoggedMessage = nil }
        }
        .onAppear {
            bridge.pipeline = pipeline
            camera.delegate = bridge
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                camera.start()
            }
        }
        .onDisappear { camera.stop() }
        .sheet(isPresented: $showOtraSearch) {
            OtraSignSearchSheet(labels: pipeline.availableClassLabels) { sign in
                submitGradingSelection(sign, source: .otraSearch)
            }
        }
        .sheet(item: $exportPayload) { payload in
            ShareSheet(url: payload.url)
        }
        .alert("Eliminar registros", isPresented: $showDeleteAlert) {
            Button("No", role: .cancel) {}
            Button("Sí, eliminar", role: .destructive) {
                let n = gradingStore.deleteAllEntries()
                lastLoggedMessage = "Se eliminaron \(n) registros"
            }
        } message: {
            Text("Se eliminarán \(gradingStore.entryCount) registro(s) de prueba. Esta acción no se puede deshacer.")
        }
    }

    private func submitGradingSelection(_ sign: String, source: GradingSelectionSource) {
        guard let round = pipeline.pendingGrading else { return }
        gradingStore.log(round: round, selectedSign: sign, source: source)
        pipeline.acknowledgeGradingRound()
        lastLoggedMessage = "Registrado: \(sign)"
    }

    private func beginCSVExport() {
        guard !isPreparingExport, gradingStore.entryCount > 0 else { return }
        isPreparingExport = true

        Task { @MainActor in
            let started = ContinuousClock.now
            await Task.yield()

            gradingStore.refreshCount()
            let url = await gradingStore.prepareExportCopy()

            let minDisplay: Duration = .milliseconds(500)
            let elapsed = started.duration(to: ContinuousClock.now)
            if elapsed < minDisplay {
                try? await Task.sleep(for: minDisplay - elapsed)
            }

            isPreparingExport = false

            guard let url else {
                lastLoggedMessage = "No se pudo preparar el CSV"
                return
            }

            try? await Task.sleep(for: .milliseconds(200))
            exportPayload = ExportSharePayload(url: url)
        }
    }

    // MARK: - Top bar

    var topBar: some View {
        HStack(spacing: 10) {
            // Model picker
            Menu {
                ForEach(AVAILABLE_MODELS) { m in
                    Button {
                        pipeline.loadModel(m)
                    } label: {
                        Label(m.name,
                              systemImage: pipeline.activeModel == m ? "checkmark.circle.fill" : "circle")
                    }
                    .disabled(pipeline.modelLoadPhase.isLoading || isPreparingExport)
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: pipeline.modelLoadPhase.isLoading ? "arrow.triangle.2.circlepath" : "brain")
                        .symbolEffect(.rotate, isActive: pipeline.modelLoadPhase.isLoading)
                    Text(pipeline.activeModel?.name ?? "Sin modelo")
                        .font(.caption)
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
            }

            Spacer()

            // Export grading log
            Button {
                beginCSVExport()
            } label: {
                Image(systemName: "square.and.arrow.up")
                    .foregroundStyle(gradingStore.entryCount > 0 && !isPreparingExport ? .white : .white.opacity(0.35))
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .disabled(gradingStore.entryCount == 0 || isPreparingExport)

            // Delete grading log
            Button {
                gradingStore.refreshCount()
                showDeleteAlert = true
            } label: {
                Image(systemName: "trash")
                    .foregroundStyle(gradingStore.entryCount > 0 ? .white : .white.opacity(0.35))
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial, in: Circle())
            }
            .disabled(gradingStore.entryCount == 0)

            // Skeleton toggle
            Button {
                withAnimation { showSkeleton.toggle() }
            } label: {
                Image(systemName: showSkeleton ? "figure.stand" : "figure.stand.line.dotted.figure.stand")
                    .foregroundStyle(showSkeleton ? .green : .white.opacity(0.5))
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial, in: Circle())
            }

            // Debug toggle
            Button {
                withAnimation { showDebug.toggle() }
            } label: {
                Image(systemName: "info.circle")
                    .foregroundStyle(showDebug ? .yellow : .white.opacity(0.5))
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial, in: Circle())
            }

            // Flip camera
            Button {
                pipeline.reset()
                camera.flipCamera()
            } label: {
                Image(systemName: "camera.rotate")
                    .foregroundStyle(.white)
                    .frame(width: 36, height: 36)
                    .background(.ultraThinMaterial, in: Circle())
            }

            // Buffer ring
            BufferRing(progress: Double(pipeline.debug.framesInBuffer) / Double(N_FRAMES))
                .frame(width: 36, height: 36)
        }
    }

    // MARK: - Debug panel

    var debugPanel: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Debug")
                    .font(.caption.bold())
                    .foregroundStyle(.yellow)
                Spacer()
            }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                debugRow("FPS",        String(format: "%.0f", pipeline.debug.fps))
                debugRow("Inferencia", String(format: "%.1f ms", pipeline.debug.inferenceMs))
                if pipeline.debug.cosineMs > 0 {
                    debugRow("Cosine", String(format: "%.1f ms", pipeline.debug.cosineMs))
                }
                if pipeline.debug.dtwMs > 0 {
                    debugRow("DTW", String(format: "%.1f ms", pipeline.debug.dtwMs))
                }
                debugRow("Buffer",     "\(pipeline.debug.framesInBuffer)/\(N_FRAMES)")
                debugRow("Modelo",     pipeline.debug.modelName)
                debugRow("Clases",     "\(pipeline.debug.nClasses)")
                debugRow("Confianza",  String(format: "%.1f%%", pipeline.debug.topConfidence * 100))
                HStack(spacing: 8) {
                    detectionPill("L", active: pipeline.debug.handLeft,  color: .green)
                    detectionPill("R", active: pipeline.debug.handRight, color: .blue)
                    detectionPill("P", active: pipeline.debug.poseDetected, color: .cyan)
                }
            }
        }
        .padding(10)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 10))
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(.white)
    }

    func debugRow(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(.white.opacity(0.6))
            Text(value).foregroundStyle(.white)
        }
    }

    func detectionPill(_ label: String, active: Bool, color: Color) -> some View {
        Text(label)
            .font(.caption2.bold())
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(active ? color.opacity(0.3) : Color.white.opacity(0.05))
            .foregroundStyle(active ? color : .white.opacity(0.3))
            .clipShape(Capsule())
            .overlay(Capsule().stroke(active ? color : .clear, lineWidth: 1))
    }

    // MARK: - Predictions panel

    var predictionsPanel: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Circle()
                    .fill(pipeline.pendingGrading != nil ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(panelStatusText)
                    .font(.subheadline.bold())
                    .foregroundStyle(.white)
                Spacer()
                if gradingStore.entryCount > 0 {
                    Text("\(gradingStore.entryCount)")
                        .font(.caption2.bold().monospacedDigit())
                        .foregroundStyle(.white.opacity(0.7))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.white.opacity(0.12), in: Capsule())
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            Divider().background(.white.opacity(0.15))

            if let error = pipeline.loadError, pipeline.activeModel == nil {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text(error).font(.caption).foregroundStyle(.white.opacity(0.8))
                }
                .padding(14)
            } else if pipeline.activeModel == nil {
                Text("Cargando modelo inicial…")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.4))
                    .multilineTextAlignment(.center)
                    .padding(14)
                    .frame(maxWidth: .infinity)
            } else if !pipeline.isBufferRecording {
                VStack(spacing: 14) {
                    Text("La cámara está activa. Cuando estés listo, inicia la captura del buffer para inferencia.")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.55))
                        .multilineTextAlignment(.center)

                    Button {
                        pipeline.commenceBufferRecording()
                    } label: {
                        Label("Comenzar captura", systemImage: "record.circle")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                }
                .padding(14)
            } else if let round = pipeline.pendingGrading {
                VStack(spacing: 8) {
                    Text("Selecciona la seña correcta")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.55))
                        .frame(maxWidth: .infinity, alignment: .leading)

                    ForEach(Array(round.predictions.enumerated()), id: \.element.id) { rank, pred in
                        GradingOptionButton(
                            rank: rank + 1,
                            label: pred.label,
                            confidence: pred.confidence,
                            isTop: rank == 0
                        ) {
                            submitGradingSelection(pred.label, source: .topPick)
                        }
                    }

                    Button {
                        showOtraSearch = true
                    } label: {
                        HStack {
                            Image(systemName: "magnifyingglass")
                            Text("Otra")
                                .font(.subheadline.weight(.semibold))
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.caption)
                                .opacity(0.5)
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)

                    Text(String(format: "Inferencia: %.0f ms · %@", round.inferenceMs, round.modelName))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.45))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            } else {
                VStack(spacing: 6) {
                    Text("Apunta la cámara hacia alguien haciendo una seña")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.4))
                        .multilineTextAlignment(.center)
                    if let msg = lastLoggedMessage {
                        Label(msg, systemImage: "checkmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.green.opacity(0.85))
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity)
            }

            if pipeline.activeModel != nil, pipeline.isBufferRecording {
                Divider().background(.white.opacity(0.15))
                resetBufferButton
                    .padding(14)
            }
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var resetBufferButton: some View {
        Button {
            pipeline.reset()
        } label: {
            Label("Reiniciar buffer", systemImage: "arrow.counterclockwise")
                .font(.subheadline.weight(.semibold))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
        }
        .buttonStyle(.borderedProminent)
        .tint(.green)
        .accessibilityLabel("Reiniciar buffer y opciones de calificación")
    }

    private var panelStatusText: String {
        if pipeline.pendingGrading != nil {
            return "Calificar predicción"
        }
        if !pipeline.isBufferRecording {
            return "Listo para comenzar"
        }
        return pipeline.isDetecting ? "Detectando…" : "Esperando seña…"
    }
}

// MARK: - Model loading overlay

struct ModelLoadingOverlay: View {
    let phase: ModelLoadPhase
    var onDismiss: () -> Void

    var body: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
                .onTapGesture {
                    if !phase.isLoading { onDismiss() }
                }

            VStack(spacing: 20) {
                iconView
                    .font(.system(size: 44))
                    .symbolEffect(.bounce, value: phase)

                VStack(spacing: 8) {
                    Text(titleText)
                        .font(.headline)
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)

                    if let subtitle = subtitleText {
                        Text(subtitle)
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.75))
                            .multilineTextAlignment(.center)
                    }
                }

                if case .loading(_, let progress) = phase {
                    VStack(spacing: 8) {
                        ProgressView(value: progress)
                            .tint(.white)
                            .frame(maxWidth: 220)
                        Text("\(Int(progress * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }

                if let warning = warningText {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.yellow)
                        Text(warning)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.85))
                            .multilineTextAlignment(.leading)
                    }
                    .padding(12)
                    .frame(maxWidth: 280)
                    .background(.yellow.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                }

                if let error = errorText {
                    HStack(alignment: .top, spacing: 8) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.red)
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.9))
                            .multilineTextAlignment(.leading)
                    }
                    .padding(12)
                    .frame(maxWidth: 280)
                    .background(.red.opacity(0.15), in: RoundedRectangle(cornerRadius: 10))
                }

                if let kept = keptModelText {
                    Text(kept)
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.65))
                        .multilineTextAlignment(.center)
                }

                if !phase.isLoading {
                    Button("Continuar", action: onDismiss)
                        .buttonStyle(.borderedProminent)
                        .tint(.white.opacity(0.2))
                        .foregroundStyle(.white)
                }
            }
            .padding(28)
            .frame(maxWidth: 320)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .strokeBorder(.white.opacity(0.15), lineWidth: 1)
            )
            .padding(.horizontal, 24)
        }
    }

    @ViewBuilder
    private var iconView: some View {
        switch phase {
        case .idle:
            EmptyView()
        case .loading:
            ProgressView()
                .scaleEffect(1.4)
                .tint(.white)
        case .succeeded:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.octagon.fill")
                .foregroundStyle(.red)
        }
    }

    private var titleText: String {
        switch phase {
        case .idle:
            return ""
        case .loading(_, _):
            return "Cargando modelo"
        case .succeeded:
            return "Modelo cargado correctamente"
        case .failed:
            return "Error al cargar el modelo"
        }
    }

    private var subtitleText: String? {
        switch phase {
        case .loading(let name, _):
            return name
        case .succeeded(let name, _):
            return name
        case .failed(let name, _, _):
            return name
        default:
            return nil
        }
    }

    private var warningText: String? {
        if case .succeeded(_, let warning) = phase { return warning }
        return nil
    }

    private var errorText: String? {
        if case .failed(_, let message, _) = phase { return message }
        return nil
    }

    private var keptModelText: String? {
        if case .failed(_, _, let kept) = phase, let kept {
            return "Se mantiene el modelo activo: \(kept)"
        }
        return nil
    }
}

// MARK: - Skeleton overlay

struct SkeletonOverlay: View {
    let skeleton: SkeletonData
    /// Front-camera preview is mirrored; Vision coords are not. Flip X so the
    /// skeleton sits on the hands the user sees.
    var mirrorX: Bool = false

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height

            Canvas { ctx, size in
                // Vision coords: x=0 left, y=0 bottom, unmirrored (same as training).
                func toScreen(_ p: CGPoint) -> CGPoint {
                    guard p != .zero else { return .zero }
                    let x = (mirrorX ? (1.0 - p.x) : p.x) * w
                    let y = (1.0 - p.y) * h
                    return CGPoint(x: x, y: y)
                }

                func drawConnections(_ points: [CGPoint],
                                     connections: [(Int, Int)],
                                     color: Color,
                                     lineWidth: CGFloat = 2) {
                    for (a, b) in connections {
                        guard a < points.count, b < points.count else { continue }
                        let pa = toScreen(points[a])
                        let pb = toScreen(points[b])
                        guard pa != .zero, pb != .zero else { continue }
                        var path = Path()
                        path.move(to: pa)
                        path.addLine(to: pb)
                        ctx.stroke(path, with: .color(color), lineWidth: lineWidth)
                    }
                }

                func drawPoints(_ points: [CGPoint],
                                color: Color,
                                radius: CGFloat = 4) {
                    for pt in points {
                        let p = toScreen(pt)
                        guard p != .zero else { continue }
                        let rect = CGRect(x: p.x - radius, y: p.y - radius,
                                          width: radius * 2, height: radius * 2)
                        ctx.fill(Path(ellipseIn: rect), with: .color(color))
                        ctx.stroke(Path(ellipseIn: rect), with: .color(.black.opacity(0.5)),
                                   lineWidth: 1)
                    }
                }

                // Pose
                if !skeleton.pose.isEmpty {
                    drawConnections(skeleton.pose, connections: POSE_CONNECTIONS,
                                    color: .cyan, lineWidth: 2)
                    drawPoints(skeleton.pose, color: .cyan, radius: 4)
                }

                // Left hand
                if !skeleton.leftHand.isEmpty {
                    drawConnections(skeleton.leftHand, connections: HAND_CONNECTIONS,
                                    color: .green, lineWidth: 2)
                    drawPoints(skeleton.leftHand, color: .green, radius: 4)
                }

                // Right hand
                if !skeleton.rightHand.isEmpty {
                    drawConnections(skeleton.rightHand, connections: HAND_CONNECTIONS,
                                    color: Color(red: 0.3, green: 0.7, blue: 1.0), lineWidth: 2)
                    drawPoints(skeleton.rightHand,
                               color: Color(red: 0.3, green: 0.7, blue: 1.0), radius: 4)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Prediction Row

struct PredictionRow: View {
    let rank:       Int
    let prediction: Prediction
    let isTop:      Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("\(rank)")
                    .font(.caption2.bold())
                    .foregroundStyle(isTop ? .yellow : .white.opacity(0.4))
                    .frame(width: 14)
                Text(prediction.label)
                    .font(isTop ? .body.bold() : .subheadline)
                    .foregroundStyle(.white)
                    .lineLimit(1)
                Spacer()
                Text(String(format: "%.1f%%", prediction.confidence * 100))
                    .font(.caption.bold())
                    .foregroundStyle(isTop ? .yellow : .white.opacity(0.6))
                    .monospacedDigit()
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3).fill(.white.opacity(0.1)).frame(height: 4)
                    RoundedRectangle(cornerRadius: 3)
                        .fill(isTop ? Color.yellow : .white.opacity(0.4))
                        .frame(width: geo.size.width * CGFloat(prediction.confidence), height: 4)
                        .animation(.easeOut(duration: 0.2), value: prediction.confidence)
                }
            }
            .frame(height: 4)
        }
        .padding(.vertical, 3)
    }
}

// MARK: - Buffer Ring

struct BufferRing: View {
    let progress: Double
    var body: some View {
        ZStack {
            Circle().stroke(.white.opacity(0.2), lineWidth: 3)
            Circle()
                .trim(from: 0, to: min(progress, 1.0))
                .stroke(progress >= 1.0 ? Color.green : Color.orange,
                        style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 0.1), value: progress)
            Text("\(Int(progress * 100))%")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(.white)
        }
    }
}

// MARK: - Camera Preview

struct CameraPreview: UIViewRepresentable {
    let camera: CameraManager
    let isFrontCamera: Bool
    let configurationToken: Int

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        let layer = AVCaptureVideoPreviewLayer(session: camera.session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = UIScreen.main.bounds
        view.layer.addSublayer(layer)
        context.coordinator.previewLayer = layer
        context.coordinator.applyMirroring(isFrontCamera: isFrontCamera)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        let front = isFrontCamera
        _ = configurationToken
        DispatchQueue.main.async {
            context.coordinator.previewLayer?.frame = uiView.bounds
            context.coordinator.applyMirroring(isFrontCamera: front)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?

        func applyMirroring(isFrontCamera: Bool) {
            guard let connection = previewLayer?.connection,
                  connection.isVideoMirroringSupported else { return }
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = isFrontCamera
        }
    }
}

// MARK: - Bridge

// Using a class (reference type) to avoid SwiftUI re-creating it on each render
class PipelineBridgeHolder: NSObject, CameraManagerDelegate {
    var pipeline: LandmarkPipeline?

    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer) {
        guard let pipeline else { return }
        Task { @MainActor in
            pipeline.processFrame(pixelBuffer)
        }
    }
}
