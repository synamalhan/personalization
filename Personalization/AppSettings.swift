import Foundation

/// Keys and defaults for settings stored with @AppStorage (which is UserDefaults underneath).
/// `nonisolated` lets any thread read these constants (the project defaults everything to the main actor).
nonisolated enum AppSettings {
    static let thresholdKey = "unknownThreshold"
    static let headKey = "headKind"
    static let backboneKey = "backbone"
    static let deletePhotosKey = "deletePhotosAfterEmbedding"

    static let defaultThreshold = 0.5
    static let defaultHead = "nearestClassMean"
    static let defaultBackbone = "mobilenetv3"

    static var threshold: Float {
        let d = UserDefaults.standard
        return d.object(forKey: thresholdKey) == nil ? Float(defaultThreshold) : Float(d.double(forKey: thresholdKey))
    }
    static var headKindRaw: String { UserDefaults.standard.string(forKey: headKey) ?? defaultHead }
    static var backboneRaw: String { UserDefaults.standard.string(forKey: backboneKey) ?? defaultBackbone }
    static var deletePhotos: Bool {
        let d = UserDefaults.standard
        return d.object(forKey: deletePhotosKey) == nil ? true : d.bool(forKey: deletePhotosKey)
    }
}
