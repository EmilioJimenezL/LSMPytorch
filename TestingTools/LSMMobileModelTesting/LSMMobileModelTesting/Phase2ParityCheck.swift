import Foundation
import CoreML

#if DEBUG
/// Validates Swift v2 preprocessing + Core ML logits against Python golden fixtures.
enum Phase2ParityCheck {
    private struct Fixture: Decodable {
        let prepared: [[Float]]?
        let prepared_sequence: [[Float]]?
        let logits: [Float]
        let feature_layout: String?
        let layout: String?
        let model_bundle: String?

        var preparedTensor: [[Float]]? { prepared ?? prepared_sequence }
        var resolvedLayout: String { feature_layout ?? layout ?? LSMConstants.layoutV2 }
    }

    static func verifyBundledFixtures() {
        let names = [
            "runs_lab_cap5_cnn_phase2_fixture",
            "runs_lab_cap5_tcn_phase2_fixture",
        ]
        for name in names {
            verify(resourceName: name)
        }
    }

    static func verify(resourceName: String) {
        guard let url = Bundle.main.url(forResource: resourceName, withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let fixture = try? JSONDecoder().decode(Fixture.self, from: data)
        else {
            print("⚠ Phase2 parity: \(resourceName).json not in bundle (ok until export)")
            return
        }

        // If fixture includes raw frames under alternate keys, prefer prepared tensor compare.
        // prepared is already the Python tensor — Swift should match when re-run on raw frames
        // if present; otherwise just verify Core ML logits against fixture.logits.
        let layout = fixture.resolvedLayout
        guard layout == LSMConstants.layoutV2 else {
            print("⚠ Phase2 parity \(resourceName): unexpected layout \(layout)")
            return
        }

        guard let prepared = fixture.preparedTensor,
              prepared.count == LSMConstants.nFrames,
              prepared.first?.count == LSMConstants.featureDimV2 else {
            print("⚠ Phase2 parity \(resourceName): bad prepared shape")
            return
        }

        if let raw = loadRawFrames(from: data) {
            if let swiftPrepared = SequencePreprocessorV2.prepareSequence(from: raw) {
                var maxDiff: Float = 0
                for (a, b) in zip(swiftPrepared.flatMap { $0 }, prepared.flatMap { $0 }) {
                    maxDiff = max(maxDiff, abs(a - b))
                }
                if maxDiff < 1e-3 {
                    print("✓ Phase2 preprocess parity (\(resourceName)) max diff \(maxDiff)")
                } else {
                    print("⚠ Phase2 preprocess FAILED (\(resourceName)) max diff \(maxDiff)")
                }
            } else {
                print("⚠ Phase2 preprocess: Swift returned nil for \(resourceName)")
            }
        } else {
            print("✓ Phase2 fixture loaded (\(resourceName)) prepared \(prepared.count)×\(prepared[0].count)")
        }

        let bundleStem = fixture.model_bundle
            ?? resourceName.replacingOccurrences(of: "_phase2_fixture", with: "_lsmoutput")
        guard let modelURL = Bundle.main.url(forResource: bundleStem, withExtension: "mlpackage")
                ?? Bundle.main.url(forResource: bundleStem, withExtension: "mlmodelc") else {
            print("⚠ Phase2 parity: model \(bundleStem) not in bundle")
            return
        }

        do {
            let config = MLModelConfiguration()
            // Simulator Espresso/MPS Graph can throw "incompatible OS"; CPU is enough for parity.
            config.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let flat = prepared.flatMap { $0 }
            let mlArray = try MLMultiArray(
                shape: [1, LSMConstants.nFrames as NSNumber, LSMConstants.featureDimV2 as NSNumber],
                dataType: .float32
            )
            for (i, v) in flat.enumerated() { mlArray[i] = NSNumber(value: v) }
            let input = try MLDictionaryFeatureProvider(dictionary: ["landmarks": mlArray])
            let output = try model.prediction(from: input)
            guard let arr = output.featureValue(for: "logits")?.multiArrayValue
                    ?? output.featureNames.compactMap({ output.featureValue(for: $0)?.multiArrayValue }).first
            else {
                print("⚠ Phase2 parity: no logits for \(resourceName)")
                return
            }
            let logits = (0..<arr.count).map { Float(truncating: arr[$0]) }
            guard logits.count == fixture.logits.count else {
                print("⚠ Phase2 parity: logit count \(logits.count) vs \(fixture.logits.count)")
                return
            }
            var maxDiff: Float = 0
            for (a, b) in zip(logits, fixture.logits) {
                maxDiff = max(maxDiff, abs(a - b))
            }
            let ptTop = fixture.logits.enumerated().max(by: { $0.element < $1.element })!.offset
            let cmTop = logits.enumerated().max(by: { $0.element < $1.element })!.offset
            if maxDiff < 0.05 && ptTop == cmTop {
                print("✓ Phase2 Core ML parity (\(resourceName)) maxDiff=\(maxDiff) top=\(cmTop)")
            } else {
                print("⚠ Phase2 Core ML FAILED (\(resourceName)) maxDiff=\(maxDiff) tops \(ptTop)/\(cmTop)")
            }
        } catch {
            print("⚠ Phase2 parity error (\(resourceName)): \(error)")
        }
    }

    private static func loadRawFrames(from data: Data) -> [[Float]]? {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let raw = obj["raw_frames"] as? [[NSNumber]] else { return nil }
        return raw.map { $0.map { $0.floatValue } }
    }
}
#endif
