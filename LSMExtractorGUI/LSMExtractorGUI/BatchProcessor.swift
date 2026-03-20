import Foundation
import Combine

@MainActor
class BatchProcessor: ObservableObject {

    // MARK: - Published State

    @Published var inputURL:   URL? = nil
    @Published var outputURL:  URL? = nil
    @Published var themesURL:  URL? = nil
    @Published var themeMap:   ThemeMap? = nil
    @Published var videos:     [VideoItem] = []
    @Published var isRunning:  Bool = false
    @Published var progress:   Double = 0.0
    @Published var log:        [String] = []
    @Published var isDone:     Bool = false
    @Published var report:     BatchReport? = nil

    private let extractor = LandmarkExtractor()

    // MARK: - Load themes.json

    func loadThemes(from url: URL) {
        themesURL = url
        do {
            let data = try Data(contentsOf: url)
            let categories = try JSONDecoder().decode([ThemeCategory].self, from: data)
            themeMap = ThemeMap(categories: categories)
            let totalWords = categories.reduce(0) { $0 + $1.words.count }
            appendLog("📋 Temas cargados: \(categories.count) categorías, \(totalWords) palabras")
            // Si ya hay un dataset cargado, re-clasificar
            if inputURL != nil { reloadDataset() }
        } catch {
            appendLog("❌ Error al leer themes.json: \(error.localizedDescription)")
            themeMap = nil
        }
    }

    // MARK: - Dataset Discovery

    func loadDataset(from url: URL) {
        inputURL = url
        reloadDataset()
    }

    private func reloadDataset() {
        guard let inputURL else { return }
        videos = []
        isDone = false
        progress = 0
        report = nil

        let fm = FileManager.default
        let videoExtensions = ["mp4", "mov", "m4v"]

        // Recopilar todos los videos de la carpeta (estructura plana o anidada)
        var allVideos: [URL] = []
        if let enumerator = fm.enumerator(
            at: inputURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) {
            for case let fileURL as URL in enumerator {
                if videoExtensions.contains(fileURL.pathExtension.lowercased()) {
                    allVideos.append(fileURL)
                }
            }
        }
        allVideos.sort { $0.lastPathComponent < $1.lastPathComponent }

        var found:   [VideoItem] = []
        var skipped: [String]    = []

        for videoURL in allVideos {
            let stem = videoURL.deletingPathExtension().lastPathComponent

            // Validar formato {palabra}[N].mp4 — debe tener solo letras/espacios + número opcional
            guard isValidName(stem) else {
                skipped.append(videoURL.lastPathComponent)
                continue
            }

            // Si hay themes.json, buscar la categoría
            if let themeMap {
                if let match = themeMap.match(for: stem) {
                    found.append(VideoItem(
                        categoryName: match.category,
                        wordFolder:   match.word.destFolder,
                        fileName:     videoURL.lastPathComponent,
                        inputURL:     videoURL
                    ))
                } else {
                    skipped.append(videoURL.lastPathComponent)
                }
            } else {
                // Sin themes.json: usar el nombre del archivo como clase
                let wordFolder = stem.replacingOccurrences(
                    of: "\\d+$", with: "", options: .regularExpression)
                    .trimmingCharacters(in: .whitespaces)
                found.append(VideoItem(
                    categoryName: "Sin categoría",
                    wordFolder:   wordFolder,
                    fileName:     videoURL.lastPathComponent,
                    inputURL:     videoURL
                ))
            }
        }

        videos = found
        let cats = Set(found.map(\.categoryName)).count
        appendLog("📂 \(found.count) videos en \(cats) categorías")
        if !skipped.isEmpty {
            appendLog("⚠ \(skipped.count) omitidos (sin match en themes.json):")
            skipped.prefix(10).forEach { appendLog("   - \($0)") }
            if skipped.count > 10 { appendLog("   ... y \(skipped.count - 10) más") }
        }
    }

    /// Valida que el nombre sea {palabra}[N] — letras, espacios, guiones y número opcional al final.
    private func isValidName(_ stem: String) -> Bool {
        let pattern = "^[\\p{L}\\s\\-_\\(\\)]+\\d*$"
        return stem.range(of: pattern, options: .regularExpression) != nil
    }

    // MARK: - Batch Processing

    func startProcessing() {
        guard outputURL != nil, !videos.isEmpty, !isRunning else { return }
        Task { await runBatch() }
    }

    private func runBatch() async {
        guard let outputURL else { return }

        isRunning = true
        isDone = false
        progress = 0
        log = []

        let totalVideos = videos.count
        appendLog("🚀 Iniciando procesamiento de \(totalVideos) videos...")
        appendLog("📁 Salida: \(outputURL.path)\n")

        let startTime  = Date()
        var processed  = 0
        var errorCount = 0
        var skipped    = 0
        var failedVideos: [FailedVideo] = []

        for index in videos.indices {
            let item = videos[index]

            videos[index].status = .processing
            appendLog("▶ [\(index + 1)/\(totalVideos)] \(item.categoryName)/\(item.wordFolder)/\(item.fileName)")

            let result = await Task.detached(priority: .userInitiated) {
                do {
                    let frames = try await self.extractor.processVideo(
                        item: item,
                        outputDir: outputURL,
                        onProgress: { _ in }
                    )
                    return Result<Int, Error>.success(frames)
                } catch {
                    return Result<Int, Error>.failure(error)
                }
            }.value

            switch result {
            case .success(let frameCount):
                processed += 1
                videos[index].status = .done
                videos[index].framesProcessed = frameCount
                appendLog("  ✓ \(frameCount) frames → \(item.categoryName)/\(item.wordFolder)/")

            case .failure(let error):
                errorCount += 1
                let msg = error.localizedDescription
                videos[index].status = .error
                videos[index].errorMessage = msg
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

        let batchReport = BatchReport(
            timestamp:    ISO8601DateFormatter().string(from: Date()),
            inputDir:     inputURL?.path ?? "",
            outputDir:    outputURL.path,
            totalVideos:  totalVideos,
            processed:    processed,
            skipped:      skipped,
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
