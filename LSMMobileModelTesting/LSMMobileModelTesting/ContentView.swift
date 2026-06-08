import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var pipeline   = LandmarkPipeline()
    @StateObject private var camera     = CameraManager()
    @State private var showDebug        = false
    @State private var showSkeleton     = true
    private let bridge = PipelineBridgeHolder()

    var body: some View {
        ZStack {
            // ── Camera ────────────────────────────────────────────────────
            CameraPreview(camera: camera)
                .ignoresSafeArea()

            // ── Skeleton overlay ──────────────────────────────────────────
            if showSkeleton {
                SkeletonOverlay(skeleton: pipeline.skeleton)
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
        }
        .onAppear {
            bridge.pipeline = pipeline
            camera.delegate = bridge
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                camera.start()
            }
        }
        .onDisappear { camera.stop() }
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
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "brain")
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
                    .fill(pipeline.isDetecting ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                    .animation(.easeInOut(duration: 0.3), value: pipeline.isDetecting)
                Text(pipeline.isDetecting ? "Detectando" : "Esperando seña...")
                    .font(.subheadline.bold())
                    .foregroundStyle(.white)
                Spacer()
                Button { pipeline.reset() } label: {
                    Image(systemName: "arrow.counterclockwise")
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            Divider().background(.white.opacity(0.15))

            if let error = pipeline.loadError {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                    Text(error).font(.caption).foregroundStyle(.white.opacity(0.8))
                }
                .padding(14)
            } else if pipeline.predictions.isEmpty {
                Text("Apunta la cámara hacia alguien haciendo una seña")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.4))
                    .multilineTextAlignment(.center)
                    .padding(14)
                    .frame(maxWidth: .infinity)
            } else {
                VStack(spacing: 5) {
                    ForEach(Array(pipeline.predictions.enumerated()), id: \.element.id) { rank, pred in
                        PredictionRow(rank: rank + 1, prediction: pred, isTop: rank == 0)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
        }
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Skeleton overlay

struct SkeletonOverlay: View {
    let skeleton: SkeletonData

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height

            Canvas { ctx, size in
                // Vision coords: x=0 left, y=0 bottom
                // AVCaptureVideoPreviewLayer already mirrors front camera
                // so we just flip Y here, no X mirror needed
                func toScreen(_ p: CGPoint) -> CGPoint {
                    guard p != .zero else { return .zero }
                    let x = p.x * w
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

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .black
        let layer = AVCaptureVideoPreviewLayer(session: camera.session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = UIScreen.main.bounds
        view.layer.addSublayer(layer)
        context.coordinator.previewLayer = layer
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            context.coordinator.previewLayer?.frame = uiView.bounds
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
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
