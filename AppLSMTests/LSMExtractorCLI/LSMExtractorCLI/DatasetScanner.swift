import Foundation

struct ScanResult {
    let videos: [VideoItem]
    let skippedPaths: [String]
    let manifest: DatasetManifest
}

enum DatasetScanner {

    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]

    /// Walk inputRoot/category/word/*.mp4 and build a fresh manifest + video list.
    static func scan(inputRoot: URL, outputRoot: String) -> ScanResult {
        let fm = FileManager.default
        var videos: [VideoItem] = []
        var skipped: [String] = []
        var categoryMap: [String: [String: [ManifestVideo]]] = [:]

        let now = ISO8601DateFormatter().string(from: Date())

        guard let categoryDirs = try? fm.contentsOfDirectory(
            at: inputRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return ScanResult(
                videos: [],
                skippedPaths: [],
                manifest: emptyManifest(source: inputRoot.path, output: outputRoot, now: now)
            )
        }

        for categoryDir in categoryDirs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let categoryName = categoryDir.lastPathComponent
            if categoryName.hasPrefix("_") || categoryName.hasPrefix(".") { continue }

            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: categoryDir.path, isDirectory: &isDir), isDir.boolValue else {
                continue
            }

            guard let wordDirs = try? fm.contentsOfDirectory(
                at: categoryDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            for wordDir in wordDirs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let wordName = wordDir.lastPathComponent
                if wordName.hasPrefix("_") || wordName.hasPrefix(".") { continue }

                var wordIsDir: ObjCBool = false
                guard fm.fileExists(atPath: wordDir.path, isDirectory: &wordIsDir), wordIsDir.boolValue else {
                    continue
                }

                let relativePath = "\(categoryName)/\(wordName)"

                guard let files = try? fm.contentsOfDirectory(
                    at: wordDir,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }

                for fileURL in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                    let ext = fileURL.pathExtension.lowercased()
                    guard videoExtensions.contains(ext) else { continue }

                    let fileName = fileURL.lastPathComponent
                    videos.append(VideoItem(
                        categoryName: categoryName,
                        wordFolder: wordName,
                        relativePath: relativePath,
                        fileName: fileName,
                        inputURL: fileURL
                    ))

                    categoryMap[categoryName, default: [:]][wordName, default: []].append(
                        .pending(filename: fileName)
                    )
                }
            }
        }

        // Detect videos at wrong depth (directly under root or category)
        if let enumerator = fm.enumerator(at: inputRoot, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) {
            for case let url as URL in enumerator {
                let ext = url.pathExtension.lowercased()
                guard videoExtensions.contains(ext) else { continue }

                let rel = url.path.replacingOccurrences(of: inputRoot.path + "/", with: "")
                let components = rel.split(separator: "/")
                if components.count != 3 {
                    skipped.append(rel)
                }
            }
        }

        var manifest = buildManifest(from: categoryMap, source: inputRoot.path, output: outputRoot, now: now)
        manifest.recomputeCounts()

        return ScanResult(videos: videos, skippedPaths: skipped, manifest: manifest)
    }

    /// Merge disk scan into an existing manifest (add new entries, preserve orphans).
    static func merge(existing: DatasetManifest, scan: ScanResult, outputRoot: String) -> DatasetManifest {
        var merged = existing
        merged.sourceRoot = scan.manifest.sourceRoot
        merged.outputRoot = outputRoot
        merged.updatedAt = ISO8601DateFormatter().string(from: Date())

        for scanCat in scan.manifest.categories {
            if let catIdx = merged.categories.firstIndex(where: { $0.name == scanCat.name }) {
                for scanWord in scanCat.words {
                    if let wordIdx = merged.categories[catIdx].words.firstIndex(where: { $0.relativePath == scanWord.relativePath }) {
                        for scanVideo in scanWord.videos {
                            if !merged.categories[catIdx].words[wordIdx].videos.contains(where: { $0.filename == scanVideo.filename }) {
                                merged.categories[catIdx].words[wordIdx].videos.append(scanVideo)
                            }
                        }
                    } else {
                        merged.categories[catIdx].words.append(scanWord)
                    }
                }
            } else {
                merged.categories.append(scanCat)
            }
        }

        merged.categories.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        for i in merged.categories.indices {
            merged.categories[i].words.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
            for j in merged.categories[i].words.indices {
                merged.categories[i].words[j].videos.sort { $0.filename.localizedCaseInsensitiveCompare($1.filename) == .orderedAscending }
            }
        }

        merged.recomputeCounts()
        return merged
    }

    static func loadManifest(from url: URL) throws -> DatasetManifest {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(DatasetManifest.self, from: data)
    }

    static func saveManifest(_ manifest: DatasetManifest, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: url, options: .atomic)
    }

    // MARK: - Private

    private static func emptyManifest(source: String, output: String, now: String) -> DatasetManifest {
        DatasetManifest(
            version: DatasetManifest.currentVersion,
            generatedAt: now,
            updatedAt: now,
            sourceRoot: source,
            outputRoot: output,
            categories: [],
            totals: ManifestTotals(categories: 0, words: 0, videos: 0)
        )
    }

    private static func buildManifest(
        from map: [String: [String: [ManifestVideo]]],
        source: String,
        output: String,
        now: String
    ) -> DatasetManifest {
        let categories = map.keys.sorted().map { catName -> ManifestCategory in
            let words = map[catName]!.keys.sorted().map { wordName -> ManifestWord in
                ManifestWord(
                    name: wordName,
                    relativePath: "\(catName)/\(wordName)",
                    videoCount: map[catName]![wordName]!.count,
                    videos: map[catName]![wordName]!
                )
            }
            let catVideos = words.reduce(0) { $0 + $1.videoCount }
            return ManifestCategory(
                name: catName,
                videoCount: catVideos,
                wordCount: words.count,
                words: words
            )
        }

        return DatasetManifest(
            version: DatasetManifest.currentVersion,
            generatedAt: now,
            updatedAt: now,
            sourceRoot: source,
            outputRoot: output,
            categories: categories,
            totals: ManifestTotals(categories: 0, words: 0, videos: 0)
        )
    }
}
