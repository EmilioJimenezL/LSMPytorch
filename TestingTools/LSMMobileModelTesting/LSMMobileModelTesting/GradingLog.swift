import Foundation
import Combine // Required for ObservableObject / @Published on GradingSessionStore

// MARK: - Types

struct GradingRound: Identifiable, Equatable {
    let id = UUID()
    let predictions: [Prediction]
    let modelName: String
    let inferenceMs: Double
    let capturedAt: Date
}

enum GradingSelectionSource: String {
    case topPick = "top_pick"
    case otraSearch = "otra_search"
}

struct ExportSharePayload: Identifiable {
    let id = UUID()
    let url: URL
}

// MARK: - Store

@MainActor
final class GradingSessionStore: ObservableObject {

    @Published private(set) var entryCount = 0

    let sessionID: String

    private let fileURL: URL
    private static let sessionKey = "grading_session_id"
    private static let fileName = "live_test_log.csv"

    private static let header = [
        "session_id", "timestamp", "model_name", "inference_ms",
        "guess1", "conf1_pct", "guess2", "conf2_pct",
        "guess3", "conf3_pct", "guess4", "conf4_pct",
        "guess5", "conf5_pct", "selected_sign", "selection_source"
    ].joined(separator: ",")

    init() {
        if let existing = UserDefaults.standard.string(forKey: Self.sessionKey) {
            sessionID = existing
        } else {
            let id = Self.makeSessionID()
            UserDefaults.standard.set(id, forKey: Self.sessionKey)
            sessionID = id
        }
        fileURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Self.fileName)
        ensureHeader()
        refreshCount()
    }

    static func makeSessionID() -> String {
        let day = ISO8601DateFormatter().string(from: Date()).prefix(10)
        return "\(day)-\(UUID().uuidString.prefix(8))"
    }

    var exportFileURL: URL { fileURL }

    /// Copies the log to an export file in Documents (share sheet–friendly; avoid `tmp/`).
    func prepareExportCopy() async -> URL? {
        let source = fileURL
        let session = sessionID
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return await Task.detached(priority: .userInitiated) {
            guard FileManager.default.fileExists(atPath: source.path) else { return nil as URL? }
            let dest = documents.appendingPathComponent("live_test_log_\(session)_export.csv")
            try? FileManager.default.removeItem(at: dest)
            do {
                try FileManager.default.copyItem(at: source, to: dest)
                return dest
            } catch {
                return nil
            }
        }.value
    }

    func log(round: GradingRound, selectedSign: String, source: GradingSelectionSource) {
        var fields: [String] = [
            csvField(sessionID),
            csvField(isoTimestamp(round.capturedAt)),
            csvField(round.modelName),
            String(format: "%.2f", round.inferenceMs)
        ]

        let preds = round.predictions.prefix(5)
        for i in 0 ..< 5 {
            if i < preds.count {
                let p = preds[preds.index(preds.startIndex, offsetBy: i)]
                fields.append(csvField(p.label))
                fields.append(String(format: "%.2f", p.confidence * 100))
            } else {
                fields.append("")
                fields.append("")
            }
        }

        fields.append(csvField(selectedSign))
        fields.append(csvField(source.rawValue))

        appendLine(fields.joined(separator: ","))
        refreshCount()
        print("✓ Grading log: \(selectedSign) (\(source.rawValue))")
    }

    @discardableResult
    func deleteAllEntries() -> Int {
        let count = entryCount
        try? FileManager.default.removeItem(at: fileURL)
        ensureHeader()
        refreshCount()
        return count
    }

    func refreshCount() {
        guard let data = try? String(contentsOf: fileURL, encoding: .utf8) else {
            entryCount = 0
            return
        }
        let lines = data.split(separator: "\n", omittingEmptySubsequences: false)
        entryCount = max(0, lines.count - 1)
    }

    // MARK: - Private

    private func ensureHeader() {
        guard !FileManager.default.fileExists(atPath: fileURL.path) else { return }
        try? (Self.header + "\n").write(to: fileURL, atomically: true, encoding: .utf8)
    }

    private func appendLine(_ line: String) {
        ensureHeader()
        guard let handle = try? FileHandle(forWritingTo: fileURL) else { return }
        defer { try? handle.close() }
        try? handle.seekToEnd()
        if let data = (line + "\n").data(using: .utf8) {
            try? handle.write(contentsOf: data)
        }
    }

    private func isoTimestamp(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: date)
    }

    private func csvField(_ value: String) -> String {
        if value.contains(",") || value.contains("\"") || value.contains("\n") {
            return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return value
    }
}

