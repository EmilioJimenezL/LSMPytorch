import Foundation

/// Shoulder-center + scale + per-joint interpolate (mirrors features.prepare_sequence_v2).
enum SequencePreprocessorV2 {

    private static var poseOffset: Int { LSMConstants.poseOffsetV2 }
    private static var leftShoulder: Int { LSMConstants.leftShoulderIdxV2 }
    private static var rightShoulder: Int { LSMConstants.rightShoulderIdxV2 }

    private static func jointBases() -> [Int] {
        var bases: [Int] = []
        for i in 0..<21 {
            bases.append(i * 3)
            bases.append(LSMConstants.nLeftHandV2 + i * 3)
        }
        for i in 0..<12 {
            bases.append(poseOffset + i * 3)
        }
        return bases
    }

    static func computeAnchor(_ vec: [Float]) -> (Float, Float)? {
        let ls = poseOffset + leftShoulder * 3
        let rs = poseOffset + rightShoulder * 3
        let leftOK = vec[ls + 2] >= LSMConstants.visibilityThreshold
        let rightOK = vec[rs + 2] >= LSMConstants.visibilityThreshold
        if leftOK && rightOK {
            return ((vec[ls] + vec[rs]) / 2, (vec[ls + 1] + vec[rs + 1]) / 2)
        }
        if leftOK { return (vec[ls], vec[ls + 1]) }
        if rightOK { return (vec[rs], vec[rs + 1]) }
        return nil
    }

    static func shoulderWidth(_ vec: [Float]) -> Float? {
        let ls = poseOffset + leftShoulder * 3
        let rs = poseOffset + rightShoulder * 3
        guard vec[ls + 2] >= LSMConstants.visibilityThreshold,
              vec[rs + 2] >= LSMConstants.visibilityThreshold else { return nil }
        let dx = vec[ls] - vec[rs]
        let dy = vec[ls + 1] - vec[rs + 1]
        let w = sqrt(dx * dx + dy * dy)
        return w > 1e-4 ? w : nil
    }

    static func isValidFrame(_ vec: [Float]) -> Bool {
        computeAnchor(vec) != nil
    }

    static func prepareSequence(from buffer: [[Float]]) -> [[Float]]? {
        let valid = buffer.filter { isValidFrame($0) && $0.count == LSMConstants.featureDimV2 }
        guard valid.count >= LSMConstants.minValidFrames else { return nil }

        var widths: [Float] = []
        for frame in valid {
            if let w = shoulderWidth(frame) { widths.append(w) }
        }
        widths.sort()
        var scale: Float = 1.0
        if !widths.isEmpty {
            scale = widths[widths.count / 2]
            if scale < 1e-4 { scale = 1.0 }
        }

        var normalized: [[Float]] = []
        normalized.reserveCapacity(valid.count)
        for frame in valid {
            guard let anchor = computeAnchor(frame) else {
                normalized.append([Float](repeating: 0, count: LSMConstants.featureDimV2))
                continue
            }
            var out = frame
            let (ax, ay) = anchor
            for base in jointBases() {
                if out[base + 2] < LSMConstants.visibilityThreshold {
                    out[base] = 0; out[base + 1] = 0; out[base + 2] = 0
                    continue
                }
                out[base] = (out[base] - ax) / scale
                out[base + 1] = (out[base + 1] - ay) / scale
                if out[base + 2] >= LSMConstants.visibilityBinThreshold {
                    out[base + 2] = 1.0
                }
            }
            normalized.append(out)
        }

        return interpolateJoints(normalized, to: LSMConstants.nFrames)
    }

    /// Per-joint interpolation; missing joints stay zero outside their span.
    private static func interpolateJoints(_ seq: [[Float]], to targetLen: Int) -> [[Float]] {
        let T = seq.count
        let D = LSMConstants.featureDimV2
        guard T > 0 else {
            return Array(repeating: [Float](repeating: 0, count: D), count: targetLen)
        }
        if T == targetLen { return seq }

        var out = Array(repeating: [Float](repeating: 0, count: D), count: targetLen)
        let tNew: [Float] = (0..<targetLen).map {
            Float($0) / Float(max(targetLen - 1, 1))
        }
        let tOld: [Float] = (0..<T).map {
            Float($0) / Float(max(T - 1, 1))
        }

        for base in jointBases() {
            let present = (0..<T).filter { seq[$0][base + 2] >= LSMConstants.visibilityThreshold }
            guard !present.isEmpty else { continue }

            if present.count == 1 {
                let t0 = present[0]
                var nearest = 0
                var best = Float.greatestFiniteMagnitude
                for (i, tn) in tNew.enumerated() {
                    let d = abs(tn - tOld[t0])
                    if d < best { best = d; nearest = i }
                }
                out[nearest][base] = seq[t0][base]
                out[nearest][base + 1] = seq[t0][base + 1]
                out[nearest][base + 2] = seq[t0][base + 2]
                continue
            }

            let tPresent = present.map { tOld[$0] }
            let tMin = tPresent.first!
            let tMax = tPresent.last!

            for ch in 0..<3 {
                let vals = present.map { seq[$0][base + ch] }
                for (i, tn) in tNew.enumerated() {
                    if tn < tMin - 1e-6 || tn > tMax + 1e-6 { continue }
                    // Find segment
                    var lo = 0
                    while lo + 1 < tPresent.count && tPresent[lo + 1] < tn { lo += 1 }
                    let hi = min(lo + 1, tPresent.count - 1)
                    let t0 = tPresent[lo], t1 = tPresent[hi]
                    let alpha: Float = abs(t1 - t0) < 1e-8 ? 0 : (tn - t0) / (t1 - t0)
                    out[i][base + ch] = vals[lo] + alpha * (vals[hi] - vals[lo])
                }
            }
        }
        return out
    }
}
