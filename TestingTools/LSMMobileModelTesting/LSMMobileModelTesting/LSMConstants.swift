import Foundation
import Vision

enum LSMConstants {
    // ── v1 (legacy / Fase 1) ─────────────────────────────────────────────────
    static let nLeftHand = 42
    static let nRightHand = 42
    static let nPose = 51
    static let featureDim = 135
    static let nFrames = 85
    static let fingerprintDim = 810

    static let poseOffset = 84
    static let leftShoulderIdx = 11
    static let rightShoulderIdx = 12
    static let visibilityThreshold: Float = 0.1
    static let visibilityBinThreshold: Float = 0.5
    static let minValidFrames = 10

    static let topKFilter = 20
    static let topKDTW = 5
    static let confidenceThreshold: Float = 0.15
    static let dtwRadius = 10

    static let binMagic: UInt32 = 0x4C534D46  // "LSMF"
    static let binVersion: UInt32 = 1
    static let binHeaderSize = 64

    // ── v2 ───────────────────────────────────────────────────────────────────
    static let featureDimV2 = 162
    static let nLeftHandV2 = 63   // 21 × (x,y,c)
    static let nRightHandV2 = 63
    static let nPoseV2 = 36       // 12 × (x,y,c)
    static let poseOffsetV2 = 126
    static let leftShoulderIdxV2 = 6
    static let rightShoulderIdxV2 = 7
    static let layoutV1 = "v1_135"
    static let layoutV2 = "v2_162"

    static let poseV2Joints: [(id: Int, name: String, joint: VNHumanBodyPoseObservation.JointName)] = [
        (0,  "nose",           .nose),
        (1,  "left_eye",       .leftEye),
        (2,  "right_eye",      .rightEye),
        (3,  "left_ear",       .leftEar),
        (4,  "right_ear",      .rightEar),
        (5,  "neck",           .neck),
        (6,  "left_shoulder",  .leftShoulder),
        (7,  "right_shoulder", .rightShoulder),
        (8,  "left_elbow",     .leftElbow),
        (9,  "right_elbow",    .rightElbow),
        (10, "left_wrist",     .leftWrist),
        (11, "right_wrist",    .rightWrist),
    ]
}
