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

// ── Themes JSON models ────────────────────────────────────────────────────────

struct ThemeWord: Codable {
    let name: String
    let videoUrl: String

    /// Nombre de carpeta destino = último segmento del videoUrl
    var destFolder: String {
        videoUrl.split(separator: "/").last.map(String.init) ?? name
    }
}

struct ThemeCategory: Codable {
    let name: String
    let words: [ThemeWord]
}

/// Mapa normalizado: nombreNormalizado → (categoria, ThemeWord)
struct ThemeMap {
    private var map: [String: (category: String, word: ThemeWord)] = [:]

    init(categories: [ThemeCategory]) {
        for cat in categories {
            for word in cat.words {
                let key = normalize(word.destFolder)
                if map[key] == nil { map[key] = (cat.name, word) }

                let key2 = normalize(word.name)
                if map[key2] == nil { map[key2] = (cat.name, word) }

                // Sin paréntesis: "Fanta (Refresco)" → "Fanta"
                let sinParen = word.destFolder
                    .replacingOccurrences(of: "\\(.*?\\)", with: "", options: .regularExpression)
                    .trimmingCharacters(in: .whitespaces)
                let key3 = normalize(sinParen)
                if !key3.isEmpty, map[key3] == nil { map[key3] = (cat.name, word) }
            }
        }
    }

    /// Busca match para un nombre de archivo sin extensión.
    /// Soporta {palabra}[N] — quita el número final antes de buscar.
    func match(for stem: String) -> (category: String, word: ThemeWord)? {
        let stemSinNumero = stem.replacingOccurrences(
            of: "\\d+$", with: "", options: .regularExpression)
        for candidate in [stemSinNumero, stem] {
            if let result = map[normalize(candidate)] { return result }
        }
        return nil
    }

    private func normalize(_ text: String) -> String {
        var s = text.lowercased()
        let replacements: [(String, String)] = [
            ("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),
            ("ü","u"),("ñ","n"),("à","a"),("è","e"),("ì","i"),("ò","o"),("ù","u"),
        ]
        for (src, dst) in replacements { s = s.replacingOccurrences(of: src, with: dst) }
        s = s.replacingOccurrences(of: "[-_/\\\\]+", with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        return s
    }
}

// ── JSON Output Models ────────────────────────────────────────────────────────

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

// ── Processing State ──────────────────────────────────────────────────────────

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
    let className:    String
    let fileName:     String
    let inputURL:     URL
    var status:       VideoStatus = .pending
    var framesProcessed: Int = 0
    var errorMessage: String? = nil

    init(categoryName: String, wordFolder: String, fileName: String, inputURL: URL) {
        self.categoryName = categoryName
        self.wordFolder   = wordFolder
        self.className    = wordFolder
        self.fileName     = fileName
        self.inputURL     = inputURL
    }
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
