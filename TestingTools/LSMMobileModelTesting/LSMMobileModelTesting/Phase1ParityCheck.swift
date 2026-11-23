import Foundation

#if DEBUG
/// Validates Swift fingerprint extraction against Python golden fixture.
enum Phase1ParityCheck {
    private struct GoldenFixture: Decodable {
        let fingerprint: [Float]
        let sequence: [[Float]]
    }

    static func verify() {
        guard let url = Bundle.main.url(forResource: "golden_sequence", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let fixture = try? JSONDecoder().decode(GoldenFixture.self, from: data)
        else {
            print("⚠ Parity: golden_sequence.json not in bundle")
            return
        }

        let swiftFP = FingerprintExtractor.extract(from: fixture.sequence)
        guard swiftFP.count == fixture.fingerprint.count else {
            print("⚠ Parity: dim mismatch \(swiftFP.count) vs \(fixture.fingerprint.count)")
            return
        }

        var maxDiff: Float = 0
        for (a, b) in zip(swiftFP, fixture.fingerprint) {
            maxDiff = max(maxDiff, abs(a - b))
        }

        if maxDiff < 1e-3 {
            print("✓ Parity check passed (max diff \(maxDiff))")
        } else {
            print("⚠ Parity check FAILED (max diff \(maxDiff))")
        }
    }
}
#endif
