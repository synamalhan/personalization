import Foundation
import Observation
import PersonalizeCore
import UIKit

/// Errors the UI shows to the user in friendly words.
enum AppError: LocalizedError {
    case busy
    case emptyName
    var errorDescription: String? {
        switch self {
        case .busy: "Still working on the last request. Try again in a moment."
        case .emptyName: "Please give the class a name."
        }
    }
}

/// Owns the one Learner and keeps it saved. `@Observable` makes SwiftUI views that read its
/// properties redraw automatically when those properties change (no publishers or objectWillChange).
/// It is passed down with `.environment(model)` and read with `@Environment(AppModel.self)`.
///
/// Each image model (backbone) has its own saved library: their embeddings have different sizes and
/// meanings, so they can never be mixed. Switching the backbone in Settings switches libraries.
@Observable
final class AppModel {
    private(set) var learner: Learner
    /// The image model whose library is currently loaded.
    private(set) var backbone: Backbone
    /// True while a teach, delete or settings change is running; blocks overlapping requests.
    private(set) var isBusy = false
    /// Whether the background was removed when the saved classes were taught (nil if nothing is taught).
    /// If this differs from the current Settings switch, old and new numbers are not comparable.
    private(set) var libraryFocusSubject: Bool?

    private let directory: URL

    static var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    static var photosURL: URL { documentsURL.appendingPathComponent("Photos") }

    init(directory: URL = AppModel.documentsURL) {
        self.directory = directory
        Self.migrateLegacyLibrary(in: directory)
        let backbone = Self.preferredBackbone()
        self.backbone = backbone
        var learner = Learner(store: Self.loadStore(at: Self.storeURL(in: directory, for: backbone)),
                              headKind: HeadKind(rawValue: AppSettings.headKindRaw) ?? .logistic)
        learner.threshold = AppSettings.threshold
        self.learner = learner
        self.libraryFocusSubject = Self.readFocus(for: backbone)
    }

    // MARK: - Libraries (one per image model)

    nonisolated static func storeURL(in directory: URL, for backbone: Backbone) -> URL {
        directory.appendingPathComponent("embeddings_\(backbone.rawValue).json")
    }

    /// The backbone chosen in Settings, if it is installed; otherwise the first installed one.
    nonisolated static func preferredBackbone() -> Backbone {
        let installed = Backbone.installed
        if let chosen = Backbone(rawValue: AppSettings.backboneRaw), installed.contains(chosen) { return chosen }
        return installed.first ?? .mobilenetv3
    }

    /// Before there were several image models the library was one file, always from MobileNetV3. Move it over.
    private static func migrateLegacyLibrary(in directory: URL) {
        let fm = FileManager.default
        let legacy = directory.appendingPathComponent("embeddings.json")
        let target = storeURL(in: directory, for: .mobilenetv3)
        if fm.fileExists(atPath: legacy.path), !fm.fileExists(atPath: target.path) {
            try? fm.moveItem(at: legacy, to: target)
        }
        let defaults = UserDefaults.standard
        let newKey = "\(AppSettings.libraryFocusKey).\(Backbone.mobilenetv3.rawValue)"
        if let old = defaults.object(forKey: AppSettings.libraryFocusKey) as? Bool, defaults.object(forKey: newKey) == nil {
            defaults.set(old, forKey: newKey)
            defaults.removeObject(forKey: AppSettings.libraryFocusKey)
        }
    }

    private static func focusKey(for backbone: Backbone) -> String { "\(AppSettings.libraryFocusKey).\(backbone.rawValue)" }
    private static func readFocus(for backbone: Backbone) -> Bool? {
        UserDefaults.standard.object(forKey: focusKey(for: backbone)) as? Bool
    }

    /// Loads a saved library. If the file exists but cannot be read, it is moved aside (not overwritten)
    /// so the next save cannot destroy data that might still be recoverable.
    nonisolated private static func loadStore(at url: URL) -> EmbeddingStore {
        guard FileManager.default.fileExists(atPath: url.path) else { return EmbeddingStore() }
        do {
            return try EmbeddingStore.load(from: url)
        } catch {
            let backup = url.deletingLastPathComponent().appendingPathComponent("embeddings.corrupt-\(Int(Date().timeIntervalSince1970)).json")
            try? FileManager.default.moveItem(at: url, to: backup)
            print("Saved embeddings were unreadable (\(error)); moved to \(backup.lastPathComponent).")
            return EmbeddingStore()
        }
    }

    /// Asks iOS for a little extra time so a refit that is running when the app goes to the
    /// background can finish and be saved instead of being cut off.
    private func withBackgroundTime<T>(_ work: () async throws -> T) async rethrows -> T {
        var id = UIBackgroundTaskIdentifier.invalid
        id = UIApplication.shared.beginBackgroundTask(withName: "refit") {
            UIApplication.shared.endBackgroundTask(id)
            id = .invalid
        }
        defer { if id != .invalid { UIApplication.shared.endBackgroundTask(id) } }
        return try await work()
    }

    var classNames: [String] { learner.classNames }

    /// True when classes were taught with a different "focus on the subject" setting than the one now selected.
    func needsReteaching(currentFocus: Bool) -> Bool {
        !classNames.isEmpty && (libraryFocusSubject ?? false) != currentFocus
    }

    func count(for className: String) -> Int { learner.count(for: className) }

    // MARK: - Teaching

    /// Adds the embeddings and refits. The refit runs off the main thread.
    /// `Task.detached` runs work on a background thread; the Learner is a value type, so we
    /// refit a copy and then swap it in.
    func teach(className: String, embeddings: [[Float]]) async throws {
        let name = className.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw AppError.emptyName }
        guard !isBusy else { throw AppError.busy }
        isBusy = true
        defer { isBusy = false }
        let snapshot = learner
        learner = try await withBackgroundTime {
            try await Task.detached(priority: .userInitiated) {
                var copy = snapshot
                try copy.teach(className: name, embeddings: embeddings)
                return copy
            }.value
        }
        libraryFocusSubject = AppSettings.focusSubject
        UserDefaults.standard.set(AppSettings.focusSubject, forKey: Self.focusKey(for: backbone))
        save()
    }

    func deleteClass(_ className: String) async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        let snapshot = learner
        learner = await withBackgroundTime {
            await Task.detached(priority: .userInitiated) {
                var copy = snapshot
                copy.deleteClass(className)
                return copy
            }.value
        }
        try? FileManager.default.removeItem(at: Self.photoDirectory(for: className))
        if learner.classNames.isEmpty {
            libraryFocusSubject = nil
            UserDefaults.standard.removeObject(forKey: Self.focusKey(for: backbone))
        }
        save()
    }

    /// Re-reads the Settings values. The threshold is instant; changing the head refits, and changing the
    /// image model loads that model's own library (both in the background).
    func applySettings() async {
        learner.threshold = AppSettings.threshold
        let kind = HeadKind(rawValue: AppSettings.headKindRaw) ?? .logistic
        let wanted = Self.preferredBackbone()
        guard kind != learner.headKind || wanted != backbone, !isBusy else { return }
        isBusy = true
        defer { isBusy = false }

        if wanted != backbone {
            let url = Self.storeURL(in: directory, for: wanted)
            let threshold = AppSettings.threshold
            learner = await Task.detached(priority: .userInitiated) {
                var fresh = Learner(store: Self.loadStore(at: url), headKind: kind)
                fresh.threshold = threshold
                return fresh
            }.value
            backbone = wanted
            libraryFocusSubject = Self.readFocus(for: wanted)
        } else {
            let snapshot = learner
            learner = await Task.detached(priority: .userInitiated) {
                var copy = snapshot
                copy.headKind = kind
                return copy
            }.value
        }
    }

    func predict(_ embedding: [Float]) -> LearnerPrediction {
        learner.predict(embedding)
    }

    // MARK: - Persistence

    private func save() {
        let url = Self.storeURL(in: directory, for: backbone)
        do { try learner.store.save(to: url) } catch { print("Could not save embeddings: \(error)") }
    }

    // MARK: - Optional photo keeping (Settings switch)

    static func photoDirectory(for className: String) -> URL {
        // Class names become folder names, so strip path separators.
        let safe = className.replacingOccurrences(of: "/", with: "_")
        return photosURL.appendingPathComponent(safe)
    }
}
