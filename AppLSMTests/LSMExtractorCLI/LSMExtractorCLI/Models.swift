import Foundation

// ── Vector structure per frame ────────────────────────────────────────────────
// [0:42]   left hand  — 21 points × (x, y)
// [42:84]  right hand — 21 points × (x, y)
// [84:135] pose       — 17 points × (x, y, visibility)
// Total: 135 values per frame — compatible with Vision Framework

let POSE_INDICES = Array(0...16)

let POSE_LABELS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

// ── Dataset manifest ──────────────────────────────────────────────────────────

struct ManifestTotals: Codable {
    var categories: Int
    var words: Int
    var videos: Int
}

struct ManifestVideo: Codable, Equatable {
    var filename: String
    var landmarksFile: String?
    var overlayFile: String?
    var status: String
    var frames: Int
    var error: String?

    enum CodingKeys: String, CodingKey {
        case filename
        case landmarksFile = "landmarks_file"
        case overlayFile   = "overlay_file"
        case status, frames, error
    }

    static func pending(filename: String) -> ManifestVideo {
        ManifestVideo(
            filename: filename,
            landmarksFile: nil,
            overlayFile: nil,
            status: "pending",
            frames: 0,
            error: nil
        )
    }
}

struct ManifestWord: Codable, Equatable {
    var name: String
    var relativePath: String
    var videoCount: Int
    var videos: [ManifestVideo]

    enum CodingKeys: String, CodingKey {
        case name
        case relativePath = "relative_path"
        case videoCount   = "video_count"
        case videos
    }
}

struct ManifestCategory: Codable, Equatable {
    var name: String
    var videoCount: Int
    var wordCount: Int
    var words: [ManifestWord]

    enum CodingKeys: String, CodingKey {
        case name
        case videoCount = "video_count"
        case wordCount  = "word_count"
        case words
    }
}

struct DatasetManifest: Codable {
    var version: Int
    var generatedAt: String
    var updatedAt: String
    var sourceRoot: String
    var outputRoot: String
    var categories: [ManifestCategory]
    var totals: ManifestTotals

    enum CodingKeys: String, CodingKey {
        case version
        case generatedAt = "generated_at"
        case updatedAt   = "updated_at"
        case sourceRoot  = "source_root"
        case outputRoot  = "output_root"
        case categories, totals
    }

    static let currentVersion = 1

    mutating func recomputeCounts() {
        for i in categories.indices {
            var catVideos = 0
            for j in categories[i].words.indices {
                categories[i].words[j].videoCount = categories[i].words[j].videos.count
                catVideos += categories[i].words[j].videoCount
            }
            categories[i].wordCount = categories[i].words.count
            categories[i].videoCount = catVideos
        }
        totals = ManifestTotals(
            categories: categories.count,
            words: categories.reduce(0) { $0 + $1.wordCount },
            videos: categories.reduce(0) { $0 + $1.videoCount }
        )
    }

    mutating func updateVideo(
        relativePath: String,
        filename: String,
        status: String,
        frames: Int,
        landmarksFile: String?,
        overlayFile: String?,
        error: String?
    ) {
        let parts = relativePath.split(separator: "/").map(String.init)
        guard parts.count == 2 else { return }
        let (category, word) = (parts[0], parts[1])

        guard let catIdx = categories.firstIndex(where: { $0.name == category }),
              let wordIdx = categories[catIdx].words.firstIndex(where: { $0.name == word }),
              let vidIdx = categories[catIdx].words[wordIdx].videos.firstIndex(where: { $0.filename == filename })
        else { return }

        categories[catIdx].words[wordIdx].videos[vidIdx].status = status
        categories[catIdx].words[wordIdx].videos[vidIdx].frames = frames
        categories[catIdx].words[wordIdx].videos[vidIdx].landmarksFile = landmarksFile
        categories[catIdx].words[wordIdx].videos[vidIdx].overlayFile = overlayFile
        categories[catIdx].words[wordIdx].videos[vidIdx].error = error
        recomputeCounts()
    }
}

// ── Landmark JSON models ──────────────────────────────────────────────────────

struct Point2D: Codable {
    let x: Float
    let y: Float
}

struct PosePoint2D: Codable {
    let id: Int
    let x: Float
    let y: Float
    let visibility: Float
}

struct FrameLandmarks: Codable {
    let frame: Int
    let timestampMs: Double
    let leftHand: [Point2D]?
    let rightHand: [Point2D]?
    let pose: [PosePoint2D]?

    enum CodingKeys: String, CodingKey {
        case frame
        case timestampMs = "timestamp_ms"
        case leftHand    = "leftHand"
        case rightHand   = "rightHand"
        case pose
    }
}

// ── Processing state ──────────────────────────────────────────────────────────

enum VideoStatus: String {
    case pending    = "Pendiente"
    case skipped    = "Omitido"
    case processing = "Procesando"
    case done       = "Completado"
    case error      = "Error"
}

struct VideoItem: Identifiable {
    let id           = UUID()
    let categoryName: String
    let wordFolder:   String
    let relativePath: String
    let className:    String
    let fileName:     String
    let inputURL:     URL
    var status:       VideoStatus = .pending
    var framesProcessed: Int = 0
    var errorMessage: String? = nil

    init(categoryName: String, wordFolder: String, relativePath: String,
         fileName: String, inputURL: URL) {
        self.categoryName = categoryName
        self.wordFolder   = wordFolder
        self.relativePath = relativePath
        self.className    = wordFolder
        self.fileName     = fileName
        self.inputURL     = inputURL
    }
}

struct ExtractionResult {
    let frameCount: Int
    let landmarksFile: String
    let overlayFile: String?
}

struct BatchReport: Codable {
    let timestamp: String
    let inputDir: String
    let outputDir: String
    let totalVideos: Int
    let processed: Int
    let skipped: Int
    let errors: Int
    let elapsedSeconds: Double
    let failed: [FailedVideo]

    enum CodingKeys: String, CodingKey {
        case timestamp
        case inputDir       = "input_dir"
        case outputDir      = "output_dir"
        case totalVideos    = "total_videos"
        case processed, skipped, errors
        case elapsedSeconds = "elapsed_seconds"
        case failed
    }
}

struct FailedVideo: Codable {
    let className: String
    let fileName: String
    let error: String

    enum CodingKeys: String, CodingKey {
        case className = "class"
        case fileName  = "file"
        case error
    }
}
