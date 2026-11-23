import AVFoundation
import SwiftUI

struct PreparationSnapshot: Equatable {
    var modelTitle: String
    var modelDetail: String
    var progress: Double?
    var modelReady: Bool
    var cameraReady: Bool
    var cameraDetail: String
    var cameraBlocked: Bool
    var failureMessage: String?
    var keptModelName: String?
    var warning: String?
}

struct PreparationStatusView: View {
    let snapshot: PreparationSnapshot
    var onRetry: () -> Void
    var onCancel: () -> Void
    var onKeepPrevious: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            VStack(spacing: 6) {
                Text("Preparando sesión")
                    .font(.title2.weight(.semibold))
                Text(snapshot.modelTitle)
                    .font(.headline)
                    .multilineTextAlignment(.center)
                Text(snapshot.modelDetail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            VStack(alignment: .leading, spacing: 14) {
                statusRow(
                    title: "Modelo",
                    ready: snapshot.modelReady,
                    detail: modelDetail,
                    showsProgress: snapshot.failureMessage == nil && !snapshot.modelReady && snapshot.progress == nil
                )
                if let progress = snapshot.progress, !snapshot.modelReady, snapshot.failureMessage == nil {
                    ProgressView(value: progress)
                        .accessibilityLabel("Avance de carga del modelo")
                        .accessibilityValue("\(Int(progress * 100)) por ciento")
                }
                statusRow(
                    title: "Cámara",
                    ready: snapshot.cameraReady,
                    detail: snapshot.cameraDetail,
                    showsProgress: !snapshot.cameraReady && !snapshot.cameraBlocked
                )
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let warning = snapshot.warning {
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            if let failure = snapshot.failureMessage {
                Text(failure)
                    .font(.body)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .accessibilityLabel("Error. \(failure)")
            }

            if snapshot.cameraBlocked {
                Text("La cámara no tiene permiso. Actívala en Ajustes para continuar.")
                    .font(.body)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            }

            VStack(spacing: 10) {
                if snapshot.failureMessage != nil {
                    Button("Reintentar", action: onRetry)
                        .buttonStyle(.borderedProminent)
                }
                if let kept = snapshot.keptModelName {
                    Button("Seguir con \(kept)", action: onKeepPrevious)
                        .disabled(!snapshot.cameraReady)
                }
                Button("Volver al catálogo", action: onCancel)
            }
        }
        .padding(24)
        .accessibilityElement(children: .contain)
    }

    private var modelDetail: String {
        if snapshot.modelReady { return "Listo para inferencia" }
        if snapshot.failureMessage != nil { return "No se pudo cargar" }
        if let progress = snapshot.progress {
            return "\(Int(progress * 100))%"
        }
        return "Cargando"
    }

    private func statusRow(title: String, ready: Bool, detail: String, showsProgress: Bool) -> some View {
        HStack(spacing: 12) {
            Image(systemName: ready ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(ready ? .green : .secondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(detail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if showsProgress {
                ProgressView()
                    .accessibilityHidden(true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(title). \(detail)")
    }
}

struct ModelPreparingView: View {
    @ObservedObject var pipeline: LandmarkPipeline
    @ObservedObject var camera: CameraManager
    let model: LSMModel
    let onReady: () -> Void
    let onCancel: () -> Void
    let onKeepPrevious: () -> Void

    @State private var didAdvance = false

    var body: some View {
        PreparationStatusView(
            snapshot: snapshot,
            onRetry: retry,
            onCancel: onCancel,
            onKeepPrevious: onKeepPrevious
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .navigationTitle("Cargando")
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden(true)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancelar", action: onCancel)
            }
        }
        .task { begin() }
        .onChange(of: pipeline.modelLoadPhase) { _, _ in advanceIfReady() }
        .onChange(of: pipeline.activeModel) { _, _ in advanceIfReady() }
        .onChange(of: camera.isRunning) { _, _ in advanceIfReady() }
        .onChange(of: camera.isAuthorized) { _, granted in
            if granted { camera.start() }
            advanceIfReady()
        }
        .onChange(of: camera.configurationToken) { _, _ in advanceIfReady() }
    }

    private var snapshot: PreparationSnapshot {
        PreparationSnapshot(
            modelTitle: model.name,
            modelDetail: "\(model.family.title) · \(model.signSet.title). \(model.description)",
            progress: loadProgress,
            modelReady: modelReady,
            cameraReady: cameraReady,
            cameraDetail: cameraDetail,
            cameraBlocked: cameraDenied,
            failureMessage: failureMessage,
            keptModelName: keptModelName,
            warning: warning
        )
    }

    private var modelReady: Bool {
        guard pipeline.activeModel == model, !pipeline.availableClassLabels.isEmpty else { return false }
        switch pipeline.modelLoadPhase {
        case .succeeded(let name, _) where name == model.name:
            return true
        case .idle, .loading, .failed, .succeeded:
            return false
        }
    }

    private var cameraReady: Bool {
        camera.isAuthorized && camera.isRunning && camera.configurationToken > 0
    }

    private var loadProgress: Double? {
        if case .loading(let name, let progress) = pipeline.modelLoadPhase, name == model.name {
            return progress
        }
        return nil
    }

    private var failureMessage: String? {
        if case .failed(let name, let message, _) = pipeline.modelLoadPhase, name == model.name {
            return message
        }
        return nil
    }

    private var keptModelName: String? {
        guard case .failed(let name, _, let kept) = pipeline.modelLoadPhase,
              name == model.name,
              kept != nil,
              pipeline.activeModel != nil else { return nil }
        return pipeline.activeModel?.name
    }

    private var warning: String? {
        if case .succeeded(let name, let warning) = pipeline.modelLoadPhase, name == model.name {
            return warning
        }
        return nil
    }

    private var cameraDenied: Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        return status == .denied || status == .restricted
    }

    private var cameraDetail: String {
        if cameraReady { return "Inicializada" }
        if cameraDenied { return "Sin permiso" }
        if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
            return "Esperando permiso"
        }
        return "Iniciando"
    }

    private func begin() {
        camera.start()
        pipeline.loadModel(model)
        advanceIfReady()
    }

    private func retry() {
        didAdvance = false
        camera.start()
        pipeline.loadModel(model)
    }

    private func advanceIfReady() {
        guard !didAdvance, modelReady, cameraReady, failureMessage == nil else { return }
        didAdvance = true
        onReady()
    }
}

#Preview("Cargando") {
    NavigationStack {
        PreparationStatusView(
            snapshot: PreparationSnapshot(
                modelTitle: "CNN — al menos 7 videos",
                modelDetail: "CNN · Al menos 7 videos. 54 clases, val Top-1 52.44%",
                progress: 0.4,
                modelReady: false,
                cameraReady: false,
                cameraDetail: "Iniciando",
                cameraBlocked: false,
                failureMessage: nil,
                keptModelName: nil,
                warning: nil
            ),
            onRetry: {},
            onCancel: {},
            onKeepPrevious: {}
        )
    }
}

#Preview("Error") {
    NavigationStack {
        PreparationStatusView(
            snapshot: PreparationSnapshot(
                modelTitle: "CNN — al menos 7 videos",
                modelDetail: "CNN · Al menos 7 videos. 54 clases, val Top-1 52.44%",
                progress: nil,
                modelReady: false,
                cameraReady: true,
                cameraDetail: "Inicializada",
                cameraBlocked: false,
                failureMessage: "No se encontró el modelo en el bundle",
                keptModelName: "Fase 1 — Fingerprint+DTW",
                warning: nil
            ),
            onRetry: {},
            onCancel: {},
            onKeepPrevious: {}
        )
    }
}
