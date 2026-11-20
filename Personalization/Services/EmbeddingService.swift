import CoreImage
import CoreML
import ImageIO
import UIKit

/// One embedding, how long it took, and the exact square image the model looked at
/// (after any background removal, resize and crop), so the app can show it.
struct EmbeddingResult: Sendable {
    let vector: [Float]
    let seconds: Double
    var seen: UIImage? = nil
}

enum EmbeddingError: LocalizedError {
    case modelMissing(String)
    case badImage
    case blankImage
    case badOutput
    case emptyOutput

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name): "The \(name) model isn't installed in this build."
        case .badImage: "That photo couldn't be read."
        case .blankImage: "The photo came out blank while it was being resized, so it was skipped."
        case .badOutput: "The model returned something unexpected."
        case .emptyOutput: "The model returned only zeros for this photo, so it was skipped."
        }
    }
}

/// The feature extractors the app can use. Only those whose compiled model is bundled are installed.
nonisolated enum Backbone: String, CaseIterable, Identifiable {
    case dinov3
    case dinov2
    case mobilenetv3

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .dinov3: "DINOv3-small"
        case .dinov2: "DINOv2-small"
        case .mobilenetv3: "MobileNetV3-Large"
        }
    }
    /// Name of the Core ML model in the app bundle (Xcode compiles .mlpackage into .mlmodelc).
    var resourceName: String {
        switch self {
        case .dinov3: "DINOv3Small_fp16"
        case .dinov2: "DINOv2Small_fp16"
        case .mobilenetv3: "MobileNetV3Large_fp16"
        }
    }
    var blurb: String {
        switch self {
        case .dinov3: "Newest from Meta, sharp on fine detail"
        case .dinov2: "Sharp, learns from very few photos"
        case .mobilenetv3: "Smaller and a little faster"
        }
    }
    var isInstalled: Bool { Bundle.main.url(forResource: resourceName, withExtension: "mlmodelc") != nil }
    static var installed: [Backbone] { allCases.filter(\.isInstalled) }
}

/// Loads photos without holding a huge bitmap in memory.
nonisolated enum ImageLoading {
    /// Photos above `maxPixels` (about 16 MP) are decoded straight to a smaller thumbnail (long side
    /// `maxLongSide`), with the photo's orientation applied. Normal photos are decoded unchanged, so the
    /// embedding pipeline is untouched for them (the thumbnail is still far larger than the 256 px target).
    static func image(from data: Data, maxPixels: Int = 16_000_000, maxLongSide: CGFloat = 2048) -> UIImage? {
        if let source = CGImageSourceCreateWithData(data as CFData, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
           let width = props[kCGImagePropertyPixelWidth] as? Int,
           let height = props[kCGImagePropertyPixelHeight] as? Int,
           width * height > maxPixels {
            let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                            kCGImageSourceCreateThumbnailWithTransform: true,
                                            kCGImageSourceShouldCacheImmediately: true,
                                            kCGImageSourceThumbnailMaxPixelSize: maxLongSide]
            if let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) {
                return UIImage(cgImage: cg)
            }
        }
        return UIImage(data: data)
    }
}

/// Preprocessing constants, copied exactly from research/config.py.
/// The converted model itself divides by 255 and applies the ImageNet mean/std, so the
/// app only has to resize and crop and hand over RGB pixels in the 0-255 range.
nonisolated enum Preprocessing {
    static let resizeShortSide: CGFloat = 256   // RESIZE_SHORT_SIDE
    static let cropSize: CGFloat = 224          // CROP_SIZE

    /// True if every pixel is (almost) black, which means the image failed to draw.
    static func isBlank(_ image: CGImage) -> Bool {
        var pixel = [UInt8](repeating: 0, count: 4)
        guard let ctx = CGContext(data: &pixel, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: 1, height: 1))   // averages the whole image into one pixel
        return pixel[0] < 2 && pixel[1] < 2 && pixel[2] < 2
    }

    /// Resize so the shorter side is 256, then center-crop 224 x 224, in a single draw.
    /// (Drawing the scaled image offset by half the overflow is the same as resize + center crop.)
    static func croppedImage(from image: UIImage) -> UIImage {
        let pixelSize = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        let scale = resizeShortSide / min(pixelSize.width, pixelSize.height)
        let resized = CGSize(width: pixelSize.width * scale, height: pixelSize.height * scale)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1          // 1 point = 1 pixel
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: cropSize, height: cropSize), format: format)
        return renderer.image { ctx in
            ctx.cgContext.interpolationQuality = .high
            // `draw(in:)` also applies the photo's orientation (portrait vs landscape).
            image.draw(in: CGRect(x: -(resized.width - cropSize) / 2, y: -(resized.height - cropSize) / 2,
                                  width: resized.width, height: resized.height))
        }
    }
}

/// Runs the Core ML backbone. An `actor` serializes access and runs its methods off the main
/// thread when called with `await`, so embedding never blocks the UI.
actor EmbeddingService {
    static let shared = EmbeddingService()

    private var models: [Backbone: MLModel] = [:]
    private let ciContext = CIContext()

    /// `focusSubject` removes the background first (see `SubjectCutout`). The gate check and benchmark
    /// turn it off so they stay comparable with the laptop pipeline, which has no cutout.
    func embed(_ image: UIImage, backbone: Backbone = Backbone(rawValue: AppSettings.backboneRaw) ?? .mobilenetv3,
               focusSubject: Bool = AppSettings.focusSubject) throws -> EmbeddingResult {
        let start = ContinuousClock.now
        let model = try loadModel(backbone)
        let source = focusSubject ? (SubjectCutout.isolate(image, context: ciContext) ?? image) : image
        let cropped = Preprocessing.croppedImage(from: source)
        // Catch a failed redraw (an all-black result) before it reaches the model.
        if let cg = cropped.cgImage, Preprocessing.isBlank(cg) { throw EmbeddingError.blankImage }
        guard let cgImage = cropped.cgImage,
              let constraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint
        else { throw EmbeddingError.badImage }
        // Sizes already match the model's 224 x 224 input, so Core ML does not resample again.
        let input = try MLFeatureValue(cgImage: cgImage, constraint: constraint, options: nil)
        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": input]))
        guard let array = output.featureValue(for: "embedding")?.multiArrayValue else { throw EmbeddingError.badOutput }
        let vector = (0..<array.count).map { array[$0].floatValue }
        if vector.allSatisfy({ $0 == 0 }) { throw EmbeddingError.emptyOutput }
        return EmbeddingResult(vector: vector, seconds: (ContinuousClock.now - start).seconds, seen: cropped)
    }

    /// Embeds a camera frame (upright BGRA pixel buffer).
    func embed(frame: CVPixelBuffer, backbone: Backbone = Backbone(rawValue: AppSettings.backboneRaw) ?? .mobilenetv3,
               focusSubject: Bool = AppSettings.focusSubject) throws -> EmbeddingResult {
        guard let cg = ciContext.createCGImage(CIImage(cvPixelBuffer: frame), from: CIImage(cvPixelBuffer: frame).extent)
        else { throw EmbeddingError.badImage }
        return try embed(UIImage(cgImage: cg), backbone: backbone, focusSubject: focusSubject)
    }

    private func loadModel(_ backbone: Backbone) throws -> MLModel {
        if let model = models[backbone] { return model }
        guard let url = Bundle.main.url(forResource: backbone.resourceName, withExtension: "mlmodelc") else {
            throw EmbeddingError.modelMissing(backbone.displayName)
        }
        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
        // The simulator returned all-zero outputs for this fp16 model with the default compute units,
        // so it runs on the CPU there. Real devices keep the default (.all: Neural Engine, GPU, CPU).
        configuration.computeUnits = .cpuOnly
        #endif
        let model = try MLModel(contentsOf: url, configuration: configuration)
        models[backbone] = model
        return model
    }
}

extension Duration {
    /// Whole duration as seconds.
    nonisolated var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}
