import Foundation

enum LSMConstants {
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
}
