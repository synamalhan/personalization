import AVFoundation
import CoreImage
import SwiftUI
import UIKit

/// Runs the camera and keeps only the newest frame. `nonisolated` because the frame callback
/// arrives on a background queue, not the main thread.
nonisolated final class CameraController: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "camera.frames")
    private let lock = NSLock()
    private var newest: CVPixelBuffer?
    private var configured = false
    private let ciContext = CIContext()

    enum Availability { case ready, denied, unavailable }

    /// Asks for permission if needed and sets up the session.
    func prepare() async -> Availability {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .notDetermined:
            guard await AVCaptureDevice.requestAccess(for: .video) else { return .denied }
        case .authorized: break
        default: return .denied
        }
        guard configure() else { return .unavailable }
        return .ready
    }

    private func configure() -> Bool {
        if configured { return true }
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else { return false }
        session.beginConfiguration()
        // 4:3 like the photos used for teaching (the default 16:9 preset gets cropped much tighter),
        // and small enough to be cheap: the model only needs a 256 px short side.
        session.sessionPreset = .vga640x480
        guard session.canAddInput(input) else { session.commitConfiguration(); return false }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true   // drop frames instead of queueing them
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { session.commitConfiguration(); return false }
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90   // deliver upright portrait frames
        }
        session.commitConfiguration()
        configured = true
        return true
    }

    func start() { queue.async { if !self.session.isRunning { self.session.startRunning() } } }
    func stop() { queue.async { if self.session.isRunning { self.session.stopRunning() } } }

    /// Copies the newest frame into a normal image right away. (The camera reuses its buffers, so a frame
    /// that is kept for later must be copied out now.) Frames are upright because of the 90 degree rotation above.
    func snapshot() -> UIImage? {
        guard let buffer = latestFrame() else { return nil }
        let image = CIImage(cvPixelBuffer: buffer)
        guard let cg = ciContext.createCGImage(image, from: image.extent) else { return nil }
        return UIImage(cgImage: cg)
    }

    func latestFrame() -> CVPixelBuffer? {
        lock.lock(); defer { lock.unlock() }
        return newest
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lock.lock(); newest = buffer; lock.unlock()
    }
}

/// Shows the camera in SwiftUI. A `UIViewRepresentable` wraps a UIKit view.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }
    func updateUIView(_ view: PreviewView, context: Context) {}
}
