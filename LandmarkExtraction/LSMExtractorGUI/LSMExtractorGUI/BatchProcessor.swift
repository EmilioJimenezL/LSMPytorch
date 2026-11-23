import Foundation
import Combine

@MainActor
class BatchProcessor: ObservableObject {

    // MARK: - Published State

    @Published var inputURL:        URL? = nil
    @Published var outputURL:       URL? = nil
    @Published var manifestURL:     URL? = nil
    @Published var generateOverlay: Bool = true
    @Published var videos:          [VideoItem] = []
    @Published var manifest:        DatasetManifest? = nil
    @Published var isRunning:       Bool = false
    @Published var progress:        Double = 0.0
    @Published var log:             [String] = []
    @Published var isDone:          Bool = false
    @Published var report:          BatchReport? = nil

    private let extractor = LandmarkExtractor()

    // MARK: - Manifest

    func loadManifest(from url: URL) {
        manifestURL = url
        do {
            manifest = try DatasetScanner.loadManifest(from: url)
            appendLog("📋 Manifiesto cargado: \(manifest?.totals.categories ?? 0) categorías, \(manifest?.totals.words ?? 0) palabras")
            if inputURL != nil { reloadDataset() }
        } catch {
            appendLog("❌ Error al leer manifiesto: \(error.localizedDescription)")
            manifest = nil
        }
    }

    // MARK: - Dataset Discovery

    func loadDataset(from url: URL) {
        inputURL = url
        reloadDataset()
    }

    private func reloadDataset() {
        guard let inputURL else { return }
        isDone = false
        progress = 0
        report = nil

        let outputPath = outputURL?.path ?? ""
        let scan = DatasetScanner.scan(inputRoot: inputURL, outputRoot: outputPath)

        if let existing = manifest, manifestURL != nil {
            manifest = DatasetScanner.merge(existing: existing, scan: scan, outputRoot: outputPath)
        } else {
            manifest = scan.manifest
        }

        videos = scan.videos
        let cats = Set(scan.videos.map(\.categoryName)).count
        appendLog("📂 \(scan.videos.count) videos en \(cats) categorías (estructura category/word/)")

        if !scan.skippedPaths.isEmpty {
            appendLog("⚠ \(scan.skippedPaths.count) videos fuera de category/word/:")
            scan.skippedPaths.prefix(10).forEach { appendLog("   - \($0)") }
            if scan.skippedPaths.count > 10 {
                appendLog("   ... y \(scan.skippedPaths.count - 10) más")
            }
        }
    }

    // MARK: - Batch Processing

    func startProcessing() {
        guard outputURL != nil, !videos.isEmpty, !isRunning else { return }
        Task { await runBatch() }
    }

    private func runBatch() async {
        guard let outputURL, var currentManifest = manifest else { return }

        isRunning = true
        isDone = false
        progress = 0

        let totalVideos = videos.count
        appendLog("🚀 Iniciando procesamiento de \(totalVideos) videos...")
        appendLog("📁 Salida: \(outputURL.path)")
        appendLog(generateOverlay ? "🎬 Overlay: activado" : "⚡ Modo rápido: solo JSON\n")

        let startTime  = Date()
        var processed  = 0
        var errorCount = 0
        var failedVideos: [FailedVideo] = []

        currentManifest.outputRoot = outputURL.path
        currentManifest.updatedAt = ISO8601DateFormatter().string(from: Date())

        for index in videos.indices {
            let item = videos[index]

            videos[index].status = .processing
            appendLog("▶ [\(index + 1)/\(totalVideos)] \(item.relativePath)/\(item.fileName)")

            currentManifest.updateVideo(
                relativePath: item.relativePath,
                filename: item.fileName,
                status: "processing",
                frames: 0,
                landmarksFile: nil,
                overlayFile: nil,
                error: nil
            )

            let overlayFlag = generateOverlay
            let result = await Task.detached(priority: .userInitiated) {
                do {
                    let extraction = try await self.extractor.processVideo(
                        item: item,
                        outputDir: outputURL,
                        generateOverlay: overlayFlag,
                        onProgress: { _ in }
                    )
                    return Result<ExtractionResult, Error>.success(extraction)
                } catch {
                    return Result<ExtractionResult, Error>.failure(error)
                }
            }.value

            switch result {
            case .success(let extraction):
                processed += 1
                videos[index].status = .done
                videos[index].framesProcessed = extraction.frameCount
                currentManifest.updateVideo(
                    relativePath: item.relativePath,
                    filename: item.fileName,
                    status: "done",
                    frames: extraction.frameCount,
                    landmarksFile: extraction.landmarksFile,
                    overlayFile: extraction.overlayFile,
                    error: nil
                )
                appendLog("  ✓ \(extraction.frameCount) frames → \(item.relativePath)/")

            case .failure(let error):
                errorCount += 1
                let msg = error.localizedDescription
                videos[index].status = .error
                videos[index].errorMessage = msg
                currentManifest.updateVideo(
                    relativePath: item.relativePath,
                    filename: item.fileName,
                    status: "error",
                    frames: 0,
                    landmarksFile: nil,
                    overlayFile: nil,
                    error: msg
                )
                failedVideos.append(FailedVideo(
                    className: item.wordFolder,
                    fileName:  item.fileName,
                    error:     msg
                ))
                appendLog("  ❌ Error: \(msg)")
            }

            progress = Double(index + 1) / Double(totalVideos)
        }

        let elapsed = Date().timeIntervalSince(startTime)
        manifest = currentManifest

        let manifestDest = manifestURL ?? outputURL.appendingPathComponent("dataset_manifest.json")
        do {
            try DatasetScanner.saveManifest(currentManifest, to: manifestDest)
            appendLog("   Manifiesto : \(manifestDest.path)")
        } catch {
            appendLog("   ⚠ Error guardando manifiesto: \(error.localizedDescription)")
        }

        let batchReport = BatchReport(
            timestamp:    ISO8601DateFormatter().string(from: Date()),
            inputDir:     inputURL?.path ?? "",
            outputDir:    outputURL.path,
            totalVideos:  totalVideos,
            processed:    processed,
            skipped:      0,
            errors:       errorCount,
            elapsedSeconds: elapsed,
            failed:       failedVideos
        )

        let reportURL = outputURL.appendingPathComponent("_batch_report.json")
        if let data = try? JSONEncoder().encode(batchReport) {
            try? data.write(to: reportURL)
        }

        report    = batchReport
        isRunning = false
        isDone    = true
        progress  = 1.0
        appendLog("\n✅ Completado en \(formatTime(elapsed))")
        appendLog("   Procesados : \(processed)/\(totalVideos)")
        if errorCount > 0 { appendLog("   Errores    : \(errorCount)") }
        appendLog("   Reporte    : \(reportURL.path)")
    }

    func cancelProcessing() {
        isRunning = false
        appendLog("⏹ Procesamiento cancelado")
    }

    // MARK: - Helpers

    private func appendLog(_ message: String) {
        log.append(message)
    }

    private func formatTime(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return "\(m)m \(s)s"
    }

    var canStart: Bool {
        inputURL != nil && outputURL != nil && !videos.isEmpty && !isRunning
    }

    var summaryText: String {
        guard let r = report else { return "" }
        return "\(r.processed)/\(r.totalVideos) videos procesados en \(String(format: "%.1f", r.elapsedSeconds))s"
    }
}
