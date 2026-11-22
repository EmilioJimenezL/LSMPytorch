import Foundation
import Vision
import AVFoundation
import CoreMedia
import CoreVideo
import CoreGraphics

// ── Hand connections for overlay drawing ─────────────────────────────────────
let HAND_CONNECTIONS: [(Int, Int)] = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

let POSE_CONNECTIONS: [(Int, Int)] = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
]

// ── Colors ────────────────────────────────────────────────────────────────────
let COLOR_POSE        = CGColor(red: 0,    green: 1,    blue: 1,    alpha: 1)
let COLOR_LEFT_HAND   = CGColor(red: 0,    green: 1,    blue: 0.3,  alpha: 1)
let COLOR_RIGHT_HAND  = CGColor(red: 0.3,  green: 0.7,  blue: 1,    alpha: 1)
let COLOR_BLACK       = CGColor(red: 0,    green: 0,    blue: 0,    alpha: 1)
let COLOR_PROGRESS    = CGColor(red: 0,    green: 0.78, blue: 0.39, alpha: 1)
let COLOR_TRACK_BG    = CGColor(red: 0.15, green: 0.15, blue: 0.15, alpha: 1)

let HAND_JOINT_NAMES: [VNHumanHandPoseObservation.JointName] = [
    .wrist,
    .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
    .indexMCP, .indexPIP, .indexDIP, .indexTip,
    .middleMCP, .middlePIP, .middleDIP, .middleTip,
    .ringMCP, .ringPIP, .ringDIP, .ringTip,
    .littleMCP, .littlePIP, .littleDIP, .littleTip,
]

/// Vision joint names for the 12 distinct pose slots (schema v2).
let POSE_V2_VISION: [VNHumanBodyPoseObservation.JointName] = [
    .nose, .leftEye, .rightEye, .leftEar, .rightEar, .neck,
    .leftShoulder, .rightShoulder, .leftElbow, .rightElbow, .leftWrist, .rightWrist,
]

/// Legacy 17-slot map used only for overlay drawing (duplicates fill missing MediaPipe indices).
let POSE_OVERLAY_MAP: [Int: VNHumanBodyPoseObservation.JointName] = [
    0: .nose, 1: .leftEye, 2: .leftEye, 3: .leftEye,
    4: .rightEye, 5: .rightEye, 6: .rightEye,
    7: .leftEar, 8: .rightEar, 9: .neck, 10: .neck,
    11: .leftShoulder, 12: .rightShoulder,
    13: .leftElbow, 14: .rightElbow,
    15: .leftWrist, 16: .rightWrist,
]

// ── Main extractor ────────────────────────────────────────────────────────────

class LandmarkExtractor {

    // MARK: - Orientation

    /// Map AVFoundation preferredTransform to Vision CGImagePropertyOrientation.
    static func orientation(from transform: CGAffineTransform) -> (vision: CGImagePropertyOrientation, label: String) {
        // Common phone/camera transforms.
        if transform.a == 0 && transform.b == 1.0 && transform.c == -1.0 && transform.d == 0 {
            return (.right, "right")
        }
        if transform.a == 0 && transform.b == -1.0 && transform.c == 1.0 && transform.d == 0 {
            return (.left, "left")
        }
        if transform.a == -1.0 && transform.b == 0 && transform.c == 0 && transform.d == -1.0 {
            return (.down, "down")
        }
        return (.up, "up")
    }

    // MARK: - Single-pass Vision detection

    struct FrameDetection {
        /// Raw Vision coords (Y=0 at bottom) for overlay.
        var leftHandRaw: [Point2D?]
        var rightHandRaw: [Point2D?]
        /// Overlay pose in legacy 17-slot layout (raw Vision coords).
        var poseOverlay: [PosePoint2D]
        /// Schema v2 pose (12 joints, raw Vision coords).
        var poseV2: [PosePoint2D]
    }

    func detectAll(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) throws -> FrameDetection {
        let handRequest = VNDetectHumanHandPoseRequest()
        handRequest.maximumHandCount = 2
        let poseRequest = VNDetectHumanBodyPoseRequest()

        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )
        try handler.perform([handRequest, poseRequest])

        var left: [Point2D?] = Array(repeating: nil, count: 21)
        var right: [Point2D?] = Array(repeating: nil, count: 21)
        var sawLeft = false
        var sawRight = false

        if let observations = handRequest.results {
            for obs in observations {
                let points = extractHandPointsOptional(from: obs)
                if obs.chirality == .left {
                    left = points
                    sawLeft = true
                } else {
                    right = points
                    sawRight = true
                }
            }
        }
        if !sawLeft { left = Array(repeating: nil, count: 21) }
        if !sawRight { right = Array(repeating: nil, count: 21) }

        let poseOverlay: [PosePoint2D]
        let poseV2: [PosePoint2D]
        if let observation = poseRequest.results?.first {
            poseOverlay = POSE_INDICES.map { idx in
                guard let jointName = POSE_OVERLAY_MAP[idx],
                      let point = try? observation.recognizedPoint(jointName),
                      point.confidence > 0.1
                else {
                    return PosePoint2D(id: idx, x: 0, y: 0, visibility: 0)
                }
                return PosePoint2D(
                    id: idx,
                    x: Float(point.location.x),
                    y: Float(point.location.y),
                    visibility: Float(point.confidence)
                )
            }
            poseV2 = zip(POSE_V2_JOINTS, POSE_V2_VISION).map { meta, jointName in
                if let point = try? observation.recognizedPoint(jointName),
                   point.confidence > 0.1 {
                    return PosePoint2D(
                        id: meta.id,
                        x: Float(point.location.x),
                        y: Float(point.location.y),
                        visibility: Float(point.confidence),
                        name: meta.name
                    )
                }
                return PosePoint2D(
                    id: meta.id, x: 0, y: 0, visibility: 0, name: meta.name
                )
            }
        } else {
            poseOverlay = POSE_INDICES.map {
                PosePoint2D(id: $0, x: 0, y: 0, visibility: 0)
            }
            poseV2 = POSE_V2_JOINTS.map {
                PosePoint2D(id: $0.id, x: 0, y: 0, visibility: 0, name: $0.name)
            }
        }

        return FrameDetection(
            leftHandRaw: left,
            rightHandRaw: right,
            poseOverlay: poseOverlay,
            poseV2: poseV2
        )
    }

    private func extractHandPointsOptional(
        from obs: VNHumanHandPoseObservation
    ) -> [Point2D?] {
        HAND_JOINT_NAMES.map { name in
            if let p = try? obs.recognizedPoint(name), p.confidence > 0.1 {
                return Point2D(
                    x: Float(p.location.x),
                    y: Float(p.location.y),
                    c: Float(p.confidence)
                )
            }
            return nil
        }
    }

    // MARK: - Overlay Drawing

    func drawOverlay(
        on pixelBuffer: CVPixelBuffer,
        leftHand: [Point2D?]?,
        rightHand: [Point2D?]?,
        pose: [PosePoint2D]?,
        frameIdx: Int,
        totalFrames: Int
    ) {
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let w = CGFloat(width)
        let h = CGFloat(height)

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: width, height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                        CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return }

        ctx.setLineWidth(2.0)

        func toPixel(_ pt: Point2D) -> CGPoint {
            CGPoint(x: CGFloat(pt.x) * w, y: CGFloat(pt.y) * h)
        }

        func drawPoints(_ points: [Point2D?], color: CGColor, connections: [(Int, Int)]) {
            ctx.setStrokeColor(color)
            for (a, b) in connections {
                guard a < points.count, b < points.count,
                      let paOpt = points[a], let pbOpt = points[b] else { continue }
                let pa = toPixel(paOpt)
                let pb = toPixel(pbOpt)
                ctx.move(to: pa)
                ctx.addLine(to: pb)
                ctx.strokePath()
            }
            for ptOpt in points {
                guard let pt = ptOpt else { continue }
                let p = toPixel(pt)
                ctx.setFillColor(color)
                ctx.fillEllipse(in: CGRect(x: p.x-4, y: p.y-4, width: 8, height: 8))
                ctx.setStrokeColor(COLOR_BLACK)
                ctx.strokeEllipse(in: CGRect(x: p.x-4, y: p.y-4, width: 8, height: 8))
            }
        }

        if let pose = pose {
            var byIdx = [Int: CGPoint]()
            for pt in pose {
                if pt.visibility > 0.1 {
                    byIdx[pt.id] = toPixel(Point2D(x: pt.x, y: pt.y))
                }
            }
            ctx.setStrokeColor(COLOR_POSE)
            for (a, b) in POSE_CONNECTIONS {
                guard let pa = byIdx[a], let pb = byIdx[b] else { continue }
                ctx.move(to: pa)
                ctx.addLine(to: pb)
                ctx.strokePath()
            }
            for pt in pose where pt.visibility > 0.1 {
                let p = toPixel(Point2D(x: pt.x, y: pt.y))
                ctx.setFillColor(COLOR_POSE)
                ctx.fillEllipse(in: CGRect(x: p.x-5, y: p.y-5, width: 10, height: 10))
                ctx.setStrokeColor(COLOR_BLACK)
                ctx.strokeEllipse(in: CGRect(x: p.x-5, y: p.y-5, width: 10, height: 10))
            }
        }

        if let lh = leftHand  { drawPoints(lh, color: COLOR_LEFT_HAND,  connections: HAND_CONNECTIONS) }
        if let rh = rightHand { drawPoints(rh, color: COLOR_RIGHT_HAND, connections: HAND_CONNECTIONS) }

        let progress = CGFloat(frameIdx + 1) / CGFloat(max(totalFrames, 1))
        let barY = h - 16
        ctx.setFillColor(COLOR_TRACK_BG)
        ctx.fill(CGRect(x: 10, y: barY, width: w - 20, height: 6))
        ctx.setFillColor(COLOR_PROGRESS)
        ctx.fill(CGRect(x: 10, y: barY, width: (w - 20) * progress, height: 6))
    }

    // MARK: - Process Single Video

    func processVideo(
        item: VideoItem,
        outputDir: URL,
        generateOverlay: Bool = true,
        onProgress: @escaping (Int) -> Void
    ) async throws -> ExtractionResult {
        let baseName = item.inputURL.deletingPathExtension().lastPathComponent

        let classOutputDir = outputDir.appendingPathComponent(item.relativePath)
        try FileManager.default.createDirectory(at: classOutputDir, withIntermediateDirectories: true)

        let asset = AVAsset(url: item.inputURL)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else {
            throw NSError(domain: "LSMExtractor", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No video track found"])
        }

        let fps         = try await videoTrack.load(.nominalFrameRate)
        let size        = try await videoTrack.load(.naturalSize)
        let transform   = try await videoTrack.load(.preferredTransform)
        let duration    = try await asset.load(.duration)
        let totalFrames = Int(CMTimeGetSeconds(duration) * Double(max(fps, 1)))
        let (visionOrientation, orientationLabel) = Self.orientation(from: transform)

        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        reader.add(readerOutput)
        reader.startReading()

        var writer: AVAssetWriter?
        var writerInput: AVAssetWriterInput?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?

        if generateOverlay {
            let outVideoURL = classOutputDir.appendingPathComponent("\(baseName)_overlay.mp4")
            if FileManager.default.fileExists(atPath: outVideoURL.path) {
                try FileManager.default.removeItem(at: outVideoURL)
            }
            let w = try AVAssetWriter(outputURL: outVideoURL, fileType: .mp4)
            let input = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: [
                    AVVideoCodecKey: AVVideoCodecType.h264,
                    AVVideoWidthKey: Int(size.width),
                    AVVideoHeightKey: Int(size.height),
                ]
            )
            input.expectsMediaDataInRealTime = false
            let ad = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: nil
            )
            w.add(input)
            w.startWriting()
            w.startSession(atSourceTime: .zero)
            writer = w
            writerInput = input
            adaptor = ad
        }

        var frames: [FrameLandmarks] = []
        var frameIdx = 0

        while let sampleBuffer = readerOutput.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let timestampMs = CMTimeGetSeconds(pts) * 1000.0

            let detection = (try? detectAll(in: pixelBuffer, orientation: visionOrientation))
                ?? FrameDetection(
                    leftHandRaw: Array(repeating: nil, count: 21),
                    rightHandRaw: Array(repeating: nil, count: 21),
                    poseOverlay: POSE_INDICES.map { PosePoint2D(id: $0, x: 0, y: 0, visibility: 0) },
                    poseV2: POSE_V2_JOINTS.map {
                        PosePoint2D(id: $0.id, x: 0, y: 0, visibility: 0, name: $0.name)
                    }
                )

            // Flip Y for JSON (Y=0 at top), keep confidence; nulls stay null.
            let leftHandJSON: [Point2D?] = detection.leftHandRaw.map { pt in
                guard let pt else { return nil }
                return Point2D(x: pt.x, y: 1.0 - pt.y, c: pt.c)
            }
            let rightHandJSON: [Point2D?] = detection.rightHandRaw.map { pt in
                guard let pt else { return nil }
                return Point2D(x: pt.x, y: 1.0 - pt.y, c: pt.c)
            }
            let poseJSON: [PosePoint2D] = detection.poseV2.map { pt in
                if pt.visibility <= 0.1 {
                    return PosePoint2D(id: pt.id, x: 0, y: 0, visibility: 0, name: pt.name)
                }
                return PosePoint2D(
                    id: pt.id, x: pt.x, y: 1.0 - pt.y,
                    visibility: pt.visibility, name: pt.name
                )
            }

            frames.append(FrameLandmarks(
                frame: frameIdx,
                timestampMs: timestampMs,
                leftHand: leftHandJSON,
                rightHand: rightHandJSON,
                pose: poseJSON
            ))

            if generateOverlay, let writerInput, let adaptor {
                drawOverlay(
                    on: pixelBuffer,
                    leftHand: detection.leftHandRaw,
                    rightHand: detection.rightHandRaw,
                    pose: detection.poseOverlay,
                    frameIdx: frameIdx,
                    totalFrames: totalFrames
                )

                while !writerInput.isReadyForMoreMediaData {
                    try await Task.sleep(nanoseconds: 5_000_000)
                }
                let presentationTime = CMTime(value: CMTimeValue(frameIdx), timescale: CMTimeScale(max(fps, 1)))
                adaptor.append(pixelBuffer, withPresentationTime: presentationTime)
            }

            frameIdx += 1
            onProgress(frameIdx)
        }

        if generateOverlay, let writerInput, let writer {
            writerInput.markAsFinished()
            await withCheckedContinuation { continuation in
                writer.finishWriting { continuation.resume() }
            }
        }

        let document = LandmarkDocument(
            schemaVersion: LANDMARK_SCHEMA_VERSION,
            fps: fps,
            width: Float(size.width),
            height: Float(size.height),
            orientation: orientationLabel,
            frames: frames
        )

        let jsonURL = classOutputDir.appendingPathComponent("\(baseName)_landmarks.json")
        let encoder = JSONEncoder()
        // Compact JSON (no prettyPrinted) — schema v2.
        let jsonData = try encoder.encode(document)
        try jsonData.write(to: jsonURL)

        let overlayName = generateOverlay ? "\(baseName)_overlay.mp4" : nil
        return ExtractionResult(
            frameCount: frameIdx,
            landmarksFile: "\(baseName)_landmarks.json",
            overlayFile: overlayName
        )
    }
}
