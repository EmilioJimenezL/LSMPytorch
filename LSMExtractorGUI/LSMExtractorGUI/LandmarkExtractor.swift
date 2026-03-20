import Foundation
import Vision
import AVFoundation
import CoreMedia
import CoreVideo
import CoreImage
import AppKit

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

// ── Main extractor ────────────────────────────────────────────────────────────

class LandmarkExtractor {

    // MARK: - Hand Detection

    func detectHands(in pixelBuffer: CVPixelBuffer)
        throws -> (left: [Point2D]?, right: [Point2D]?)
    {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 2

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        guard let observations = request.results, !observations.isEmpty else {
            return (nil, nil)
        }

        var left:  [Point2D]? = nil
        var right: [Point2D]? = nil

        for obs in observations {
            let points = (try? extractHandPoints(from: obs)) ?? []
            if obs.chirality == .left {
                left = points
            } else {
                right = points
            }
        }
        return (left, right)
    }

    private func extractHandPoints(from obs: VNHumanHandPoseObservation) throws -> [Point2D] {
        let jointNames: [VNHumanHandPoseObservation.JointName] = [
            .wrist,
            .thumbCMC, .thumbMP, .thumbIP, .thumbTip,
            .indexMCP, .indexPIP, .indexDIP, .indexTip,
            .middleMCP, .middlePIP, .middleDIP, .middleTip,
            .ringMCP, .ringPIP, .ringDIP, .ringTip,
            .littleMCP, .littlePIP, .littleDIP, .littleTip,
        ]

        return jointNames.map { name in
            if let p = try? obs.recognizedPoint(name), p.confidence > 0.1 {
                // Raw Vision coords (Y=0 at bottom) — used for drawing.
                // Y flip for JSON is applied when building FrameLandmarks.
                return Point2D(x: Float(p.location.x), y: Float(p.location.y))
            }
            return Point2D(x: 0, y: 0)
        }
    }

    // MARK: - Pose Detection

    func detectPose(in pixelBuffer: CVPixelBuffer) throws -> [PosePoint2D]? {
        let request = VNDetectHumanBodyPoseRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        guard let observation = request.results?.first else { return nil }

        // Map our indices to Vision joint names
        // Note: Vision doesn't have inner/outer eye variants or explicit mouth points
        // We map the closest available joint for each index
        let jointMap: [Int: VNHumanBodyPoseObservation.JointName] = [
            0:  .nose,
            1:  .leftEye,
            2:  .leftEye,
            3:  .leftEye,
            4:  .rightEye,
            5:  .rightEye,
            6:  .rightEye,
            7:  .leftEar,
            8:  .rightEar,
            9:  .neck,
            10: .neck,
            11: .leftShoulder,
            12: .rightShoulder,
            13: .leftElbow,
            14: .rightElbow,
            15: .leftWrist,
            16: .rightWrist,
        ]

        return POSE_INDICES.map { idx in
            guard let jointName = jointMap[idx],
                  let point = try? observation.recognizedPoint(jointName),
                  point.confidence > 0.1
            else {
                return PosePoint2D(id: idx, x: 0, y: 0, visibility: 0)
            }
            return PosePoint2D(
                id: idx,
                x: Float(point.location.x),
                y: Float(point.location.y),  // raw Vision coords — flip applied at JSON time
                visibility: Float(point.confidence)
            )
        }
    }

    // MARK: - Overlay Drawing

    func drawOverlay(
        on pixelBuffer: CVPixelBuffer,
        leftHand: [Point2D]?,
        rightHand: [Point2D]?,
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
            // Vision Y=0 is at bottom, CGContext on pixelBuffer Y=0 is also at bottom
            // So no flip needed — coordinates map directly
            CGPoint(x: CGFloat(pt.x) * w, y: CGFloat(pt.y) * h)
        }

        func drawPoints(_ points: [Point2D], color: CGColor, connections: [(Int, Int)]) {
            ctx.setStrokeColor(color)
            for (a, b) in connections {
                guard a < points.count, b < points.count else { continue }
                let pa = toPixel(points[a])
                let pb = toPixel(points[b])
                if pa == .zero || pb == .zero { continue }
                ctx.move(to: pa)
                ctx.addLine(to: pb)
                ctx.strokePath()
            }
            for pt in points {
                let p = toPixel(pt)
                if p == .zero { continue }
                ctx.setFillColor(color)
                ctx.fillEllipse(in: CGRect(x: p.x-4, y: p.y-4, width: 8, height: 8))
                ctx.setStrokeColor(COLOR_BLACK)
                ctx.strokeEllipse(in: CGRect(x: p.x-4, y: p.y-4, width: 8, height: 8))
            }
        }

        // Draw pose
        if let pose = pose {
            let posePoints = pose.map { Point2D(x: $0.x, y: $0.y) }
            var byIdx = [Int: CGPoint]()
            for (i, idx) in POSE_INDICES.enumerated() {
                byIdx[idx] = toPixel(posePoints[i])
            }
            ctx.setStrokeColor(COLOR_POSE)
            for (a, b) in POSE_CONNECTIONS {
                guard let pa = byIdx[a], let pb = byIdx[b] else { continue }
                if pa == .zero || pb == .zero { continue }
                ctx.move(to: pa)
                ctx.addLine(to: pb)
                ctx.strokePath()
            }
            for pt in posePoints {
                let p = toPixel(pt)
                if p == .zero { continue }
                ctx.setFillColor(COLOR_POSE)
                ctx.fillEllipse(in: CGRect(x: p.x-5, y: p.y-5, width: 10, height: 10))
                ctx.setStrokeColor(COLOR_BLACK)
                ctx.strokeEllipse(in: CGRect(x: p.x-5, y: p.y-5, width: 10, height: 10))
            }
        }

        if let lh = leftHand  { drawPoints(lh, color: COLOR_LEFT_HAND,  connections: HAND_CONNECTIONS) }
        if let rh = rightHand { drawPoints(rh, color: COLOR_RIGHT_HAND, connections: HAND_CONNECTIONS) }

        // Progress bar
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
        onProgress: @escaping (Int) -> Void
    ) async throws -> Int {
        let baseName = item.inputURL.deletingPathExtension().lastPathComponent

        // Estructura de salida: outputDir/categoria/palabra/
        let classOutputDir: URL
        if item.categoryName.isEmpty || item.categoryName == "Sin categoría" {
            classOutputDir = outputDir.appendingPathComponent(item.wordFolder)
        } else {
            classOutputDir = outputDir
                .appendingPathComponent(item.categoryName)
                .appendingPathComponent(item.wordFolder)
        }
        try FileManager.default.createDirectory(at: classOutputDir, withIntermediateDirectories: true)

        let asset = AVAsset(url: item.inputURL)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else {
            throw NSError(domain: "LSMExtractor", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "No video track found"])
        }

        let fps        = try await videoTrack.load(.nominalFrameRate)
        let size       = try await videoTrack.load(.naturalSize)
        let duration   = try await asset.load(.duration)
        let totalFrames = Int(CMTimeGetSeconds(duration) * Double(fps))

        // AVAssetReader for input
        let reader = try AVAssetReader(asset: asset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: videoTrack,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        reader.add(readerOutput)
        reader.startReading()

        // AVAssetWriter for overlay video
        let outVideoURL = classOutputDir.appendingPathComponent("\(baseName)_overlay.mp4")
        if FileManager.default.fileExists(atPath: outVideoURL.path) {
            try FileManager.default.removeItem(at: outVideoURL)
        }
        let writer = try AVAssetWriter(outputURL: outVideoURL, fileType: .mp4)
        let writerInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: Int(size.width),
                AVVideoHeightKey: Int(size.height),
            ]
        )
        writerInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: nil
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        var frames: [FrameLandmarks] = []
        var frameIdx = 0

        while let sampleBuffer = readerOutput.copyNextSampleBuffer() {
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { continue }

            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let timestampMs = CMTimeGetSeconds(pts) * 1000.0

            let (leftHand, rightHand) = (try? detectHands(in: pixelBuffer)) ?? (nil, nil)
            let pose = try? detectPose(in: pixelBuffer)

            // Flip Y for JSON (Y=0 at top, consistent with screen coords)
            // Drawing uses raw Vision coords (Y=0 at bottom) which matches CGContext
            let leftHandJSON  = leftHand.map  { pts in pts.map { Point2D(x: $0.x, y: 1.0 - $0.y) } }
            let rightHandJSON = rightHand.map { pts in pts.map { Point2D(x: $0.x, y: 1.0 - $0.y) } }
            let poseJSON      = pose.map { pts in pts.map {
                PosePoint2D(id: $0.id, x: $0.x, y: 1.0 - $0.y, visibility: $0.visibility)
            }}

            frames.append(FrameLandmarks(
                frame: frameIdx,
                timestampMs: timestampMs,
                leftHand: leftHandJSON,
                rightHand: rightHandJSON,
                pose: poseJSON
            ))

            drawOverlay(
                on: pixelBuffer,
                leftHand: leftHand,
                rightHand: rightHand,
                pose: pose,
                frameIdx: frameIdx,
                totalFrames: totalFrames
            )

            while !writerInput.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 5_000_000) // 5ms
            }
            let presentationTime = CMTime(value: CMTimeValue(frameIdx), timescale: CMTimeScale(fps))
            adaptor.append(pixelBuffer, withPresentationTime: presentationTime)

            frameIdx += 1
            onProgress(frameIdx)
        }

        // Finalize writer
        writerInput.markAsFinished()
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }

        // Save JSON
        let jsonURL = classOutputDir.appendingPathComponent("\(baseName)_landmarks.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted]
        let jsonData = try encoder.encode(frames)
        try jsonData.write(to: jsonURL)

        return frameIdx
    }
}
