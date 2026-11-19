import Foundation

/// Errors thrown when vectors can't be added to the store.
public enum EmbeddingStoreError: Error, Equatable {
    /// A vector was empty, or its length differs from vectors already stored.
    case dimensionMismatch(expected: Int, got: Int)
}

/// Holds the embedding vectors taught for each class, keyed by class name.
///
/// This is a `struct`: a value type that is copied on assignment, so each copy
/// is independent. `Codable` lets Swift turn it into JSON and back automatically.
public struct EmbeddingStore: Codable, Equatable, Sendable {
    private(set) public var embeddings: [String: [[Float]]]

    public init() {
        embeddings = [:]
    }

    /// Class names in alphabetical order, so results are stable.
    public var classNames: [String] {
        embeddings.keys.sorted()
    }

    /// Length of the stored vectors, or nil when the store is empty.
    public var dimension: Int? {
        embeddings.values.first?.first?.count
    }

    /// Number of vectors stored for a class (0 if the class doesn't exist).
    public func count(for className: String) -> Int {
        embeddings[className]?.count ?? 0
    }

    /// Adds vectors to a class, creating it if needed.
    /// `mutating` marks a method that changes the struct itself.
    /// Nothing is added if any vector has the wrong length.
    public mutating func add(className: String, vectors: [[Float]]) throws {
        let expected = dimension ?? vectors.first?.count ?? 0
        for vector in vectors where vector.isEmpty || vector.count != expected {
            throw EmbeddingStoreError.dimensionMismatch(expected: expected, got: vector.count)
        }
        guard !vectors.isEmpty else { return }
        embeddings[className, default: []].append(contentsOf: vectors)
    }

    /// Deletes a class and all of its vectors. Unknown names are ignored.
    public mutating func removeClass(_ className: String) {
        embeddings.removeValue(forKey: className)
    }

    /// Writes the store to `url` as JSON.
    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    /// Reads a store previously written by `save(to:)`.
    public static func load(from url: URL) throws -> EmbeddingStore {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(EmbeddingStore.self, from: data)
    }
}
