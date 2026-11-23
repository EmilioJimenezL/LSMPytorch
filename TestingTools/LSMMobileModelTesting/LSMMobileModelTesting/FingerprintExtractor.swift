import Foundation
import Accelerate

/// Statistical fingerprint extractor: 135 channels × 6 stats = 810, L2-normalized.
enum FingerprintExtractor {

    static func extract(from sequence: [[Float]]) -> [Float] {
        guard !sequence.isEmpty else { return [Float](repeating: 0, count: LSMConstants.fingerprintDim) }

        var fingerprint = [Float]()
        fingerprint.reserveCapacity(LSMConstants.fingerprintDim)

        for ch in 0 ..< LSMConstants.featureDim {
            let channel = sequence.map { $0[ch] }
            let n = vDSP_Length(channel.count)

            var mean: Float = 0
            vDSP_meanv(channel, 1, &mean, n)

            var std: Float = 0
            if n > 1 {
                var centered = channel
                var negMean = -mean
                vDSP_vsadd(centered, 1, &negMean, &centered, 1, n)
                vDSP_measqv(centered, 1, &std, n)
                std = sqrtf(std)
            }

            var minVal: Float = 0
            var maxVal: Float = 0
            vDSP_minv(channel, 1, &minVal, n)
            vDSP_maxv(channel, 1, &maxVal, n)

            let sorted = channel.sorted()
            let p25 = percentile(sorted, 25)
            let p75 = percentile(sorted, 75)

            fingerprint.append(contentsOf: [mean, std, minVal, maxVal, p25, p75])
        }

        var norm: Float = 0
        vDSP_svesq(fingerprint, 1, &norm, vDSP_Length(fingerprint.count))
        norm = sqrtf(norm)
        if norm > 0 {
            var divisor = norm
            vDSP_vsdiv(fingerprint, 1, &divisor, &fingerprint, 1, vDSP_Length(fingerprint.count))
        }

        return fingerprint
    }

    private static func percentile(_ sorted: [Float], _ p: Float) -> Float {
        guard !sorted.isEmpty else { return 0 }
        if sorted.count == 1 { return sorted[0] }
        let rank = (p / 100.0) * Float(sorted.count - 1)
        let lo = Int(floor(rank))
        let hi = min(lo + 1, sorted.count - 1)
        let frac = rank - Float(lo)
        return sorted[lo] + frac * (sorted[hi] - sorted[lo])
    }
}
