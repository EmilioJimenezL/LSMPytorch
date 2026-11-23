import Foundation

/// Rest-to-rest segmenter for live v2 inference.
/// A sign starts when a hand is present and wrist motion rises; ends after rest
/// frames or at max length.
final class LiveSegmenter {
    enum State: String {
        case idle
        case signing
        case ready   // segment complete, waiting to be consumed
    }

    private(set) var state: State = .idle
    private var frames: [[Float]] = []
    private var restCount = 0
    private var lastWrist: (Float, Float)?

    var motionThreshold: Float = 0.012
    var restFramesRequired = 12
    var minFrames = LSMConstants.minValidFrames
    var maxFrames = LSMConstants.nFrames * 2  // 170

    var debugLabel: String { state.rawValue }

    func reset() {
        state = .idle
        frames = []
        restCount = 0
        lastWrist = nil
    }

    /// Feed one raw v2 frame. Returns a completed segment when ready.
    func push(_ frame: [Float]) -> [[Float]]? {
        let motion = wristMotion(frame)
        let handPresent = hasHand(frame)

        switch state {
        case .idle:
            if handPresent && motion > motionThreshold {
                state = .signing
                frames = [frame]
                restCount = 0
            }
            return nil

        case .signing:
            frames.append(frame)
            if !handPresent || motion < motionThreshold * 0.5 {
                restCount += 1
            } else {
                restCount = 0
            }
            if restCount >= restFramesRequired || frames.count >= maxFrames {
                let segment = frames
                reset()
                if segment.count >= minFrames {
                    state = .idle
                    return segment
                }
            }
            return nil

        case .ready:
            return nil
        }
    }

    private func hasHand(_ frame: [Float]) -> Bool {
        // Any left/right hand joint with confidence
        for i in 0..<21 {
            if frame[i * 3 + 2] >= LSMConstants.visibilityThreshold { return true }
            if frame[LSMConstants.nLeftHandV2 + i * 3 + 2] >= LSMConstants.visibilityThreshold { return true }
        }
        return false
    }

    private func wristMotion(_ frame: [Float]) -> Float {
        // Prefer pose left/right wrist (v2 indices 10, 11)
        let bases = [
            LSMConstants.poseOffsetV2 + 10 * 3,
            LSMConstants.poseOffsetV2 + 11 * 3,
            0,  // hand left wrist
            LSMConstants.nLeftHandV2, // hand right wrist
        ]
        var cx: Float = 0, cy: Float = 0, n: Float = 0
        for b in bases {
            guard b + 2 < frame.count, frame[b + 2] >= LSMConstants.visibilityThreshold else { continue }
            cx += frame[b]; cy += frame[b + 1]; n += 1
        }
        guard n > 0 else {
            lastWrist = nil
            return 0
        }
        cx /= n; cy /= n
        defer { lastWrist = (cx, cy) }
        guard let prev = lastWrist else { return 0 }
        let dx = cx - prev.0, dy = cy - prev.1
        return sqrt(dx * dx + dy * dy)
    }
}
