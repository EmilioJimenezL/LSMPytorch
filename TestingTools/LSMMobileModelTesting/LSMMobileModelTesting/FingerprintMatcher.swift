import Foundation
import Accelerate

struct FingerprintMatchResult {
    let label: String
    let confidence: Float
    let rank: Int
}

/// Loads fingerprints.bin and performs two-stage cosine + DTW matching.
final class FingerprintMatcher {

    private(set) var fingerprints: [[Float]] = []
    private(set) var labels: [UInt32] = []
    private(set) var sequences: [[[Float]]] = []
    private(set) var classes: [String] = []
    private(set) var loadError: String?

    var isLoaded: Bool { loadError == nil && !fingerprints.isEmpty }

    typealias ProgressHandler = (Double) -> Void

    init(bundleName: String = "fingerprints", progress: ProgressHandler? = nil) {
        progress?(0.05)
        if let url = Bundle.main.url(forResource: bundleName, withExtension: "bin") {
            do {
                progress?(0.12)
                let data = try Data(contentsOf: url)
                progress?(0.18)
                parse(data: data, progress: progress)
            } catch {
                loadError = error.localizedDescription
            }
        } else {
            loadError = "No se encontró \(bundleName).bin en el bundle"
        }
    }

    init(data: Data, progress: ProgressHandler? = nil) {
        parse(data: data, progress: progress)
    }

    private func parse(data: Data, progress: ProgressHandler? = nil) {
        guard data.count >= LSMConstants.binHeaderSize else {
            loadError = "Archivo demasiado pequeño"
            return
        }

        let magic = data.readLEUInt32(at: 0)
        let version = data.readLEUInt32(at: 4)
        let nVideos = Int(data.readLEUInt32(at: 8))
        let nClasses = Int(data.readLEUInt32(at: 12))
        let fpDim = Int(data.readLEUInt32(at: 16))
        let nFrames = Int(data.readLEUInt32(at: 20))
        let featDim = Int(data.readLEUInt32(at: 24))

        guard magic == LSMConstants.binMagic else {
            loadError = "Magic inválido"
            return
        }
        guard version == LSMConstants.binVersion else {
            loadError = "Versión no soportada: \(version)"
            return
        }

        var offset = LSMConstants.binHeaderSize
        progress?(0.25)

        do {
            fingerprints = try data.readFloatMatrix(at: &offset, rows: nVideos, cols: fpDim)
            progress?(0.42)

            labels = (0 ..< nVideos).map { _ in
                let v = data.readLEUInt32(at: offset)
                offset += 4
                return v
            }
            progress?(0.48)

            sequences = try (0 ..< nVideos).map { idx in
                if idx % 150 == 0 || idx == nVideos - 1 {
                    let frac = Double(idx + 1) / Double(max(nVideos, 1))
                    progress?(0.48 + 0.40 * frac)
                }
                return try data.readFloatMatrix(at: &offset, rows: nFrames, cols: featDim)
            }
            progress?(0.90)

            classes = []
            for i in 0 ..< nClasses {
                guard offset + 2 <= data.count else { break }
                let length = Int(data.readLEUInt16(at: offset))
                offset += 2
                guard offset + length <= data.count else { break }
                if let name = String(data: data.subdata(in: offset ..< offset + length), encoding: .utf8) {
                    classes.append(name)
                }
                offset += length
                if i % 40 == 0 || i == nClasses - 1 {
                    let frac = Double(i + 1) / Double(max(nClasses, 1))
                    progress?(0.90 + 0.10 * frac)
                }
            }

            loadError = nil
            progress?(1.0)
        } catch {
            loadError = error.localizedDescription
        }
    }

    func match(querySequence: [[Float]]) -> (results: [FingerprintMatchResult], cosineMs: Double, dtwMs: Double) {
        guard isLoaded else { return ([], 0, 0) }

        let queryFP = FingerprintExtractor.extract(from: querySequence)

        let tCosine = CFAbsoluteTimeGetCurrent()
        var candidates: [(index: Int, cosineDist: Float)] = []
        candidates.reserveCapacity(fingerprints.count)
        for i in 0 ..< fingerprints.count {
            candidates.append((i, cosineDistance(queryFP, fingerprints[i])))
        }
        candidates.sort { $0.cosineDist < $1.cosineDist }
        let topFilter = Array(candidates.prefix(LSMConstants.topKFilter))
        let cosineMs = (CFAbsoluteTimeGetCurrent() - tCosine) * 1000

        let tDtw = CFAbsoluteTimeGetCurrent()
        var dtwResults: [(index: Int, dtwDist: Float)] = []
        dtwResults.reserveCapacity(topFilter.count)
        for c in topFilter {
            dtwResults.append((c.index, dtwDistance(querySequence, sequences[c.index])))
        }
        dtwResults.sort { $0.dtwDist < $1.dtwDist }
        let dtwMs = (CFAbsoluteTimeGetCurrent() - tDtw) * 1000

        var classScores: [Int: Float] = [:]
        for item in dtwResults.prefix(LSMConstants.topKDTW) {
            let label = Int(labels[item.index])
            classScores[label, default: 0] += 1.0 / (item.dtwDist + 1e-6)
        }

        let total = classScores.values.reduce(0, +)
        let ranked = classScores.sorted { $0.value > $1.value }

        var results: [FingerprintMatchResult] = []
        for (rank, entry) in ranked.enumerated() {
            let normScore = entry.value / max(total, 1e-6)
            if normScore < LSMConstants.confidenceThreshold && rank > 0 { break }
            let labelIdx = entry.key
            let name = labelIdx < classes.count ? classes[labelIdx] : "clase_\(labelIdx)"
            results.append(FingerprintMatchResult(label: name, confidence: normScore, rank: rank + 1))
        }
        return (results, cosineMs, dtwMs)
    }

    private func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        return 1.0 - dot
    }

    private func dtwDistance(_ a: [[Float]], _ b: [[Float]]) -> Float {
        let n = a.count
        let m = b.count
        guard n > 0, m > 0 else { return .infinity }
        let radius = LSMConstants.dtwRadius

        var prev = [Float](repeating: .infinity, count: m + 1)
        var curr = [Float](repeating: .infinity, count: m + 1)
        prev[0] = 0

        for i in 1 ... n {
            curr[0] = .infinity
            let jStart = max(1, i - radius)
            let jEnd = min(m, i + radius)
            for j in jStart ... jEnd {
                let cost = frameDistance(a[i - 1], b[j - 1])
                curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
            }
            swap(&prev, &curr)
        }
        return prev[m]
    }

    private func frameDistance(_ a: [Float], _ b: [Float]) -> Float {
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        var sq: Float = 0
        vDSP_svesq(diff, 1, &sq, vDSP_Length(a.count))
        return sqrtf(sq)
    }
}

// MARK: - Binary parsing

private enum BinParseError: Error {
    case outOfBounds
}

private extension Data {
    func readLEUInt32(at offset: Int) -> UInt32 {
        subdata(in: offset ..< offset + 4).withUnsafeBytes { $0.load(as: UInt32.self) }
    }

    func readLEUInt16(at offset: Int) -> UInt16 {
        subdata(in: offset ..< offset + 2).withUnsafeBytes { $0.load(as: UInt16.self) }
    }

    func readFloats(at offset: inout Int, count: Int) throws -> [Float] {
        let byteCount = count * MemoryLayout<Float>.size
        guard offset + byteCount <= self.count else { throw BinParseError.outOfBounds }
        let slice = subdata(in: offset ..< offset + byteCount)
        offset += byteCount
        return slice.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Float.self)
            return Array(UnsafeBufferPointer(start: ptr.baseAddress, count: count))
        }
    }

    func readFloatMatrix(at offset: inout Int, rows: Int, cols: Int) throws -> [[Float]] {
        let flat = try readFloats(at: &offset, count: rows * cols)
        return stride(from: 0, to: flat.count, by: cols).map { start in
            Array(flat[start ..< start + cols])
        }
    }
}
