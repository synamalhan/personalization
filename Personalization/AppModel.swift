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
@Observable
final class AppModel {
    private(set) var learner: Learner
    /// True while a teach, delete or head change is running; blocks overlapping requests.
    private(set) var isBusy = false

    private let storeURL: URL

    static var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    static var defaultStoreURL: URL { documentsURL.appendingPathComponent("embeddings.json") }
    static var photosURL: URL { documentsURL.appendingPathComponent("Photos") }

    init(storeURL: URL = AppModel.defaultStoreURL) {
        self.storeURL = storeURL
        let store = Self.loadStore(at: storeURL)
        var learner = Learner(store: store,
                              headKind: HeadKind(rawValue: AppSettings.headKindRaw) ?? .nearestClassMean)
        learner.threshold = AppSettings.threshold
        self.learner = learner
    }

    /// Loads the saved store. If the file exists but cannot be read, it is moved aside (not overwritten)
    /// so the next save cannot destroy data that might still be recoverable.
    private static func loadStore(at url: URL) -> EmbeddingStore {
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

    func count(for className: String) -> Int { learner.count(for: className) }

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
        save()
    }

    /// Re-reads the Settings values (threshold is instant; changing the head refits in the background).
    func applySettings() async {
        learner.threshold = AppSettings.threshold
        let kind = HeadKind(rawValue: AppSettings.headKindRaw) ?? .nearestClassMean
        guard kind != learner.headKind, !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        let snapshot = learner
        learner = await Task.detached(priority: .userInitiated) {
            var copy = snapshot
            copy.headKind = kind
            return copy
        }.value
    }

    func predict(_ embedding: [Float]) -> LearnerPrediction {
        learner.predict(embedding)
    }

    // MARK: - Persistence

    private func save() {
        do { try learner.store.save(to: storeURL) } catch { print("Could not save embeddings: \(error)") }
    }

    // MARK: - Optional photo keeping (Settings switch)

    static func photoDirectory(for className: String) -> URL {
        // Class names become folder names, so strip path separators.
        let safe = className.replacingOccurrences(of: "/", with: "_")
        return photosURL.appendingPathComponent(safe)
    }
}
