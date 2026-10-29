import Foundation

struct CLIOptions {
    let inputURL: URL
    let outputURL: URL
    let manifestURL: URL?
    let generateOverlay: Bool
    let skipExisting: Bool
}

struct BatchRunner {

    private let extractor = LandmarkExtractor()

    func run(_ options: CLIOptions) async -> Int32 {
        do {
            try FileManager.default.createDirectory(
                at: options.outputURL,
                withIntermediateDirectories: true
            )
        } catch {
            eprint("No se pudo crear la carpeta de salida: \(error.localizedDescription)")
            return 1
        }

        let scan = DatasetScanner.scan(
            inputRoot: options.inputURL,
            outputRoot: options.outputURL.path
        )

        var manifest: DatasetManifest
        if let manifestURL = options.manifestURL {
            do {
                let existing = try DatasetScanner.loadManifest(from: manifestURL)
                log("Manifiesto cargado: \(existing.totals.categories) categorías, \(existing.totals.words) palabras")
                manifest = DatasetScanner.merge(
                    existing: existing,
                    scan: scan,
                    outputRoot: options.outputURL.path
                )
            } catch {
                eprint("Error al leer manifiesto: \(error.localizedDescription)")
                return 1
            }
        } else {
            manifest = scan.manifest
        }

        let videos = scan.videos
        let categories = Set(videos.map(\.categoryName)).count
        log("\(videos.count) videos en \(categories) categorías (estructura category/word/)")

        if !scan.skippedPaths.isEmpty {
            log("\(scan.skippedPaths.count) videos fuera de category/word/:")
            for path in scan.skippedPaths.prefix(10) {
                log("  - \(path)")
            }
            if scan.skippedPaths.count > 10 {
                log("  ... y \(scan.skippedPaths.count - 10) más")
            }
        }

        log("Salida: \(options.outputURL.path)")
        log(options.generateOverlay ? "Overlay: activado" : "Overlay: desactivado (solo JSON)")
        if options.skipExisting {
            log("Omitir existentes: activado")
        }

        manifest.outputRoot = options.outputURL.path
        manifest.updatedAt = ISO8601DateFormatter().string(from: Date())

        let startTime = Date()
        var processed = 0
        var skipped = 0
        var errorCount = 0
        var failedVideos: [FailedVideo] = []
        let totalVideos = videos.count

        for (index, item) in videos.enumerated() {
            log("[\(index + 1)/\(totalVideos)] \(item.relativePath)/\(item.fileName)")

            if options.skipExisting, let existing = existingLandmarks(for: item, outputDir: options.outputURL) {
                skipped += 1
                manifest.updateVideo(
                    relativePath: item.relativePath,
                    filename: item.fileName,
                    status: "skipped",
                    frames: existing.frames,
                    landmarksFile: existing.landmarksFile,
                    overlayFile: existing.overlayFile,
                    error: nil
                )
                log("  omitido (ya existe \(existing.landmarksFile))")
                continue
            }

            manifest.updateVideo(
                relativePath: item.relativePath,
                filename: item.fileName,
                status: "processing",
                frames: 0,
                landmarksFile: nil,
                overlayFile: nil,
                error: nil
            )

            do {
                let extraction = try await extractor.processVideo(
                    item: item,
                    outputDir: options.outputURL,
                    generateOverlay: options.generateOverlay,
                    onProgress: { _ in }
                )
                processed += 1
                manifest.updateVideo(
                    relativePath: item.relativePath,
                    filename: item.fileName,
                    status: "done",
                    frames: extraction.frameCount,
                    landmarksFile: extraction.landmarksFile,
                    overlayFile: extraction.overlayFile,
                    error: nil
                )
                log("  \(extraction.frameCount) frames → \(item.relativePath)/")
            } catch {
                errorCount += 1
                let message = error.localizedDescription
                manifest.updateVideo(
                    relativePath: item.relativePath,
                    filename: item.fileName,
                    status: "error",
                    frames: 0,
                    landmarksFile: nil,
                    overlayFile: nil,
                    error: message
                )
                failedVideos.append(FailedVideo(
                    className: item.wordFolder,
                    fileName: item.fileName,
                    error: message
                ))
                log("  error: \(message)")
            }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let manifestDest = options.manifestURL
            ?? options.outputURL.appendingPathComponent("dataset_manifest.json")

        var exitCode: Int32 = errorCount > 0 ? 2 : 0
        do {
            try DatasetScanner.saveManifest(manifest, to: manifestDest)
            log("Manifiesto : \(manifestDest.path)")
        } catch {
            eprint("Error guardando manifiesto: \(error.localizedDescription)")
            if exitCode == 0 { exitCode = 1 }
        }

        let batchReport = BatchReport(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            inputDir: options.inputURL.path,
            outputDir: options.outputURL.path,
            totalVideos: totalVideos,
            processed: processed,
            skipped: skipped,
            errors: errorCount,
            elapsedSeconds: elapsed,
            failed: failedVideos
        )

        let reportURL = options.outputURL.appendingPathComponent("_batch_report.json")
        do {
            let data = try JSONEncoder().encode(batchReport)
            try data.write(to: reportURL)
            log("Reporte    : \(reportURL.path)")
        } catch {
            eprint("Error guardando reporte: \(error.localizedDescription)")
            if exitCode == 0 { exitCode = 1 }
        }

        log("Completado en \(formatTime(elapsed))")
        log("Procesados : \(processed)/\(totalVideos)")
        log("Omitidos   : \(skipped)")
        if errorCount > 0 {
            log("Errores    : \(errorCount)")
        }
        return exitCode
    }

    // MARK: - Skip existing

    private struct ExistingLandmarks {
        let frames: Int
        let landmarksFile: String
        let overlayFile: String?
    }

    private func existingLandmarks(for item: VideoItem, outputDir: URL) -> ExistingLandmarks? {
        let baseName = item.inputURL.deletingPathExtension().lastPathComponent
        let landmarksFile = "\(baseName)_landmarks.json"
        let directory = outputDir.appendingPathComponent(item.relativePath)
        let landmarksURL = directory.appendingPathComponent(landmarksFile)
        guard FileManager.default.fileExists(atPath: landmarksURL.path) else { return nil }

        let overlayFile = "\(baseName)_overlay.mp4"
        let overlayURL = directory.appendingPathComponent(overlayFile)
        let hasOverlay = FileManager.default.fileExists(atPath: overlayURL.path)

        return ExistingLandmarks(
            frames: frameCount(at: landmarksURL),
            landmarksFile: landmarksFile,
            overlayFile: hasOverlay ? overlayFile : nil
        )
    }

    private func frameCount(at url: URL) -> Int {
        guard let data = try? Data(contentsOf: url),
              let frames = try? JSONDecoder().decode([FrameLandmarks].self, from: data)
        else { return 0 }
        return frames.count
    }

    // MARK: - Logging

    private func log(_ message: String) {
        print(message)
        fflush(stdout)
    }

    private func eprint(_ message: String) {
        fputs(message + "\n", stderr)
    }

    private func formatTime(_ seconds: Double) -> String {
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        let minutes = Int(seconds) / 60
        let remainder = Int(seconds) % 60
        return "\(minutes)m \(remainder)s"
    }
}
