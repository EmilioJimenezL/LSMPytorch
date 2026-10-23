import Foundation

/// Shoulder-anchor normalization and valid-frame filtering (mirrors preprocess.py).
enum SequencePreprocessor {

    private static func shoulderBase(_ idx: Int) -> Int {
        LSMConstants.poseOffset + idx * 3
    }

    static func computeAnchor(_ vec: [Float]) -> (Float, Float)? {
        let ls = shoulderBase(LSMConstants.leftShoulderIdx)
        let rs = shoulderBase(LSMConstants.rightShoulderIdx)

        let leftVis = vec[ls + 2] > LSMConstants.visibilityThreshold &&
            (vec[ls] != 0 || vec[ls + 1] != 0)
        let rightVis = vec[rs + 2] > LSMConstants.visibilityThreshold &&
            (vec[rs] != 0 || vec[rs + 1] != 0)

        if leftVis && rightVis {
            return ((vec[ls] + vec[rs]) / 2, (vec[ls + 1] + vec[rs + 1]) / 2)
        }
        if leftVis { return (vec[ls], vec[ls + 1]) }
        if rightVis { return (vec[rs], vec[rs + 1]) }
        return nil
    }

    static func isValidFrame(_ vec: [Float]) -> Bool {
        guard computeAnchor(vec) != nil else { return false }

        let hasLeft = handHasPoints(vec, offset: 0, dim: LSMConstants.nLeftHand)
        let hasRight = handHasPoints(vec, offset: LSMConstants.nLeftHand, dim: LSMConstants.nRightHand)
        let poseCount = countNonzeroPoseXY(vec)

        return hasLeft || hasRight || poseCount >= 3
    }

    private static func handHasPoints(_ vec: [Float], offset: Int, dim: Int) -> Bool {
        for i in offset ..< offset + dim {
            if vec[i] != 0 { return true }
        }
        return false
    }

    private static func countNonzeroPoseXY(_ vec: [Float]) -> Int {
        var count = 0
        for i in 0 ..< 17 {
            let base = LSMConstants.poseOffset + i * 3
            if vec[base] != 0 || vec[base + 1] != 0 { count += 1 }
        }
        return count
    }

    static func normalizeFrame(_ vec: [Float]) -> [Float]? {
        guard let anchor = computeAnchor(vec) else { return nil }
        var out = vec
        let (ax, ay) = anchor

        for i in 0 ..< LSMConstants.nLeftHand / 2 {
            let base = i * 2
            if out[base] != 0 || out[base + 1] != 0 {
                out[base] -= ax
                out[base + 1] -= ay
            }
        }

        for i in 0 ..< LSMConstants.nRightHand / 2 {
            let base = LSMConstants.nLeftHand + i * 2
            if out[base] != 0 || out[base + 1] != 0 {
                out[base] -= ax
                out[base + 1] -= ay
            }
        }

        for i in 0 ..< 17 {
            let base = LSMConstants.poseOffset + i * 3
            if out[base] != 0 || out[base + 1] != 0 {
                out[base] -= ax
                out[base + 1] -= ay
            }
            out[base + 2] = out[base + 2] >= LSMConstants.visibilityBinThreshold ? 1.0 : 0.0
        }

        return out
    }

    static func filterValidFrames(_ seq: [[Float]]) -> [[Float]] {
        seq.filter { isValidFrame($0) }
    }

    static func interpolateSequence(_ seq: [[Float]], to targetLen: Int) -> [[Float]] {
        let T = seq.count
        guard T > 0 else {
            return Array(repeating: [Float](repeating: 0, count: LSMConstants.featureDim), count: targetLen)
        }
        if T == targetLen { return seq }

        return (0 ..< targetLen).map { newIdx in
            let t = Float(newIdx) / Float(targetLen - 1) * Float(T - 1)
            let lo = min(Int(t), T - 1)
            let hi = min(lo + 1, T - 1)
            let alpha = t - Float(lo)
            return zip(seq[lo], seq[hi]).map { a, b in a + alpha * (b - a) }
        }
    }

    /// Normalize a raw frame vector; returns nil if invalid.
    static func preprocessFrame(_ vec: [Float]) -> [Float]? {
        guard isValidFrame(vec) else { return nil }
        return normalizeFrame(vec)
    }

    /// Full buffer pipeline: filter valid → normalize → interpolate to 85.
    static func prepareSequence(from buffer: [[Float]]) -> [[Float]]? {
        let valid = buffer.compactMap { frame -> [Float]? in
            guard isValidFrame(frame) else { return nil }
            return normalizeFrame(frame)
        }
        guard valid.count >= LSMConstants.minValidFrames else { return nil }
        return interpolateSequence(valid, to: LSMConstants.nFrames)
    }
}
