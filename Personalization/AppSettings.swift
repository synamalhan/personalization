import Foundation

/// Keys and defaults for settings stored with @AppStorage (which is UserDefaults underneath).
/// `nonisolated` lets any thread read these constants (the project defaults everything to the main actor).
nonisolated enum AppSettings {
    static let thresholdKey = "unknownThreshold"
    static let headKey = "headKind"
    static let backboneKey = "backbone"
    static let deletePhotosKey = "deletePhotosAfterEmbedding"
    static let focusSubjectKey = "focusOnSubject"
    static let libraryFocusKey = "libraryTaughtWithFocus"

    static let defaultThreshold = 0.5
    static let defaultHead = "logistic"
    /// The best bundled image model: DINOv3, then DINOv2, then MobileNetV3.
    static var defaultBackbone: String { (Backbone.installed.first ?? .mobilenetv3).rawValue }

    static var threshold: Float {
        let d = UserDefaults.standard
        return d.object(forKey: thresholdKey) == nil ? Float(defaultThreshold) : Float(d.double(forKey: thresholdKey))
    }
    static var headKindRaw: String { UserDefaults.standard.string(forKey: headKey) ?? defaultHead }
    static var backboneRaw: String { UserDefaults.standard.string(forKey: backboneKey) ?? defaultBackbone }
    /// Remove the background before embedding (on by default).
    static var focusSubject: Bool {
        let d = UserDefaults.standard
        return d.object(forKey: focusSubjectKey) == nil ? true : d.bool(forKey: focusSubjectKey)
    }
    static var deletePhotos: Bool {
        let d = UserDefaults.standard
        return d.object(forKey: deletePhotosKey) == nil ? true : d.bool(forKey: deletePhotosKey)
    }
}
