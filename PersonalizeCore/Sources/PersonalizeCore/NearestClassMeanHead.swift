import Foundation

/// The best-matching class for a vector and how similar it was (cosine, -1...1).
public struct Prediction: Equatable, Sendable {
    public let className: String
    public let similarity: Float

    public init(className: String, similarity: Float) {
        self.className = className
        self.similarity = similarity
    }
}

/// Outcome of classifying with an "unknown" threshold.
/// An `enum` with associated values: each case can carry its own data.
public enum Classification: Equatable, Sendable {
    case known(Prediction)
    /// `bestGuess` is the closest class that was still too far away (nil if nothing is taught).
    case unknown(bestGuess: Prediction?)
}

/// Nearest-class-mean classifier using cosine similarity. It needs no training:
/// each class is represented by its L2-normalized mean vector.
public struct NearestClassMeanHead: Head {
    private var classMeans: [(className: String, mean: [Float])] = []

    /// Softmax temperature used to turn cosine similarities into probabilities.
    /// Smaller values make the top class more dominant.
    public let temperature: Float

    /// An unfit head; call `fit(store:)` before predicting.
    public init(temperature: Float = 0.1) {
        self.temperature = temperature
    }

    /// Builds the head from every class in the store.
    public init(store: EmbeddingStore, temperature: Float = 0.1) {
        self.temperature = temperature
        fit(store: store)
    }

    public mutating func fit(store: EmbeddingStore) {
        classMeans = store.classNames.compactMap { name in
            guard let vectors = store.embeddings[name],
                  let mean = Self.normalizedMean(of: vectors) else { return nil }
            return (name, mean)
        }
    }

    public var classNames: [String] {
        classMeans.map(\.className)
    }

    /// Probabilities from a softmax over `cosine similarity / temperature`.
    /// Empty if no class can be compared.
    public func predict(_ vector: [Float]) -> [String: Float] {
        guard let query = Self.normalized(vector) else { return [:] }
        let candidates = classMeans.filter { $0.mean.count == query.count }
        let probabilities = softmax(candidates.map { Self.dot($0.mean, query) / temperature })
        return Dictionary(uniqueKeysWithValues: zip(candidates.map(\.className), probabilities))
    }

    /// Best class by cosine similarity, or nil if no class can be compared
    /// (empty store, or the vector has the wrong length / zero length).
    public func bestMatch(_ vector: [Float]) -> Prediction? {
        guard let query = Self.normalized(vector) else { return nil }
        var best: Prediction?
        for (name, mean) in classMeans where mean.count == query.count {
            let similarity = Self.dot(mean, query)
            if let current = best, current.similarity >= similarity { continue }
            best = Prediction(className: name, similarity: similarity)
        }
        return best
    }

    /// Like `bestMatch`, but returns `.unknown` when the best similarity is below `threshold`.
    public func classify(_ vector: [Float], threshold: Float) -> Classification {
        guard let best = bestMatch(vector) else { return .unknown(bestGuess: nil) }
        return best.similarity >= threshold ? .known(best) : .unknown(bestGuess: best)
    }

    // MARK: - Vector math

    static func dot(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
    }

    /// Scales a vector to length 1. Returns nil for empty or all-zero vectors,
    /// which have no direction.
    static func normalized(_ v: [Float]) -> [Float]? {
        let norm = dot(v, v).squareRoot()
        guard norm > 0, norm.isFinite else { return nil }
        return v.map { $0 / norm }
    }

    static func normalizedMean(of vectors: [[Float]]) -> [Float]? {
        guard let first = vectors.first, !first.isEmpty else { return nil }
        var sum = [Float](repeating: 0, count: first.count)
        for vector in vectors where vector.count == first.count {
            for i in sum.indices { sum[i] += vector[i] }
        }
        return normalized(sum)
    }
}

// MARK: - Threshold suggestion

extension NearestClassMeanHead {
    /// Suggests an unknown threshold from the taught classes' own spread.
    ///
    /// For each vector in a class with at least 2 vectors, we compare it to the mean of
    /// that class's *other* vectors (leave-one-out; otherwise a vector would trivially
    /// match a mean it helped build). The threshold is `mean - k * standard deviation`
    /// of those similarities, so most genuine examples fall above it.
    /// Returns nil when no class has 2+ vectors, since there's no spread to measure.
    public static func suggestedThreshold(from store: EmbeddingStore, stdDevs k: Float = 2) -> Float? {
        var similarities: [Float] = []
        for vectors in store.embeddings.values where vectors.count >= 2 {
            for i in vectors.indices {
                var others = vectors
                others.remove(at: i)
                guard let mean = normalizedMean(of: others),
                      let query = normalized(vectors[i]),
                      mean.count == query.count else { continue }
                similarities.append(dot(mean, query))
            }
        }
        guard !similarities.isEmpty else { return nil }
        let n = Float(similarities.count)
        let avg = similarities.reduce(0, +) / n
        let variance = similarities.reduce(0) { $0 + ($1 - avg) * ($1 - avg) } / n
        return min(1, max(-1, avg - k * variance.squareRoot()))
    }
}
