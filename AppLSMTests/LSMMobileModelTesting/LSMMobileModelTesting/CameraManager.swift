import AVFoundation
import Combine
import UIKit

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer)
}

class CameraManager: NSObject, ObservableObject {

    weak var delegate: CameraManagerDelegate?

    let session = AVCaptureSession()
    private var currentInput: AVCaptureDeviceInput?
    private let videoOutput  = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "lsm.camera.session", qos: .userInteractive)
    private let outputQueue  = DispatchQueue(label: "lsm.camera.output",  qos: .userInteractive)

    @Published var isRunning    = false
    @Published var isAuthorized = false
    @Published var isFrontCamera = true
    /// Bumps after each session configuration so the preview can reapply mirroring.
    @Published private(set) var configurationToken = 0

    override init() {
        super.init()
        checkAuthorization()
    }

    // MARK: - Authorization

    private func checkAuthorization() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isAuthorized = true
            setupSession(position: .front)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.isAuthorized = granted
                    if granted { self?.setupSession(position: .front) }
                }
            }
        default:
            isAuthorized = false
        }
    }

    // MARK: - Setup

    private func setupSession(position: AVCaptureDevice.Position) {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1280x720

            // Remove existing input
            if let existing = self.currentInput {
                self.session.removeInput(existing)
            }

            guard let device = AVCaptureDevice.default(
                .builtInWideAngleCamera, for: .video, position: position),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }

            self.session.addInput(input)
            self.currentInput = input

            // Output (only add once)
            if self.session.outputs.isEmpty {
                self.videoOutput.videoSettings = [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ]
                self.videoOutput.alwaysDiscardsLateVideoFrames = true
                self.videoOutput.setSampleBufferDelegate(self, queue: self.outputQueue)
                if self.session.canAddOutput(self.videoOutput) {
                    self.session.addOutput(self.videoOutput)
                }
            }

            // Portrait buffers for Vision. Do not mirror this connection:
            // training landmarks come from unmirrored video files, and a
            // mirrored front-camera buffer swaps hand chirality.
            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
                if connection.isVideoMirroringSupported {
                    connection.automaticallyAdjustsVideoMirroring = false
                    connection.isVideoMirrored = false
                }
            }

            self.session.commitConfiguration()
            DispatchQueue.main.async { self.configurationToken += 1 }
        }
    }

    // MARK: - Camera flip

    func flipCamera() {
        isFrontCamera.toggle()
        let position: AVCaptureDevice.Position = isFrontCamera ? .front : .back
        setupSession(position: position)
    }

    // MARK: - Control

    func start() {
        sessionQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
            DispatchQueue.main.async { self.isRunning = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            DispatchQueue.main.async { self.isRunning = false }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        delegate?.cameraManager(self, didOutput: pixelBuffer)
    }
}
