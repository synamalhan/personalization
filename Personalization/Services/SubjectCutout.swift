import CoreImage
import UIKit
import Vision

/// Removes the background so the model sees the subject, not the room behind it.
///
/// Two people photographed in the same place look almost the same to an image model that reads the
/// whole picture. This uses Apple's on-device subject lifting (`VNGenerateForegroundInstanceMaskRequest`):
/// it finds the main subject(s), keeps only their pixels, replaces everything else with a neutral
/// gray, and crops tight around the subject. Nothing leaves the device.
nonisolated enum SubjectCutout {
    /// ImageNet mean color. After the model's normalization this gray becomes 0, so it is "neutral" to the model.
    private static let background = CIColor(red: 0.485, green: 0.456, blue: 0.406)
    /// Large photos are shrunk first; the segmentation does not need more, and it is much faster.
    private static let maxSide: CGFloat = 1024

    /// The subject on a neutral background, cropped to its extent. Returns nil if no subject was found
    /// (the caller then falls back to the whole photo).
    static func isolate(_ image: UIImage, context: CIContext) -> UIImage? {
        guard let upright = uprightCGImage(image) else { return nil }
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: upright, options: [:])
        do {
            try handler.perform([request])
            guard let observation = request.results?.first, !observation.allInstances.isEmpty else { return nil }
            let buffer = try observation.generateMaskedImage(ofInstances: observation.allInstances,
                                                             from: handler, croppedToInstancesExtent: true)
            let subject = CIImage(cvPixelBuffer: buffer)       // transparent everywhere except the subject
            let gray = CIImage(color: background).cropped(to: subject.extent)
            guard let result = context.createCGImage(subject.composited(over: gray), from: subject.extent) else { return nil }
            return UIImage(cgImage: result)
        } catch {
            return nil
        }
    }

    /// Draws the image upright (orientation applied) at no more than `maxSide` pixels.
    private static func uprightCGImage(_ image: UIImage) -> CGImage? {
        let pixels = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        let shrink = min(1, maxSide / max(pixels.width, pixels.height))
        let size = CGSize(width: (pixels.width * shrink).rounded(), height: (pixels.height * shrink).rounded())
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }.cgImage
    }
}
