import Foundation

/// A classifier head that is fit on stored embeddings and then predicts class probabilities.
///
/// A `protocol` lists the methods a type promises to have, so the app can use any head
/// without caring which one it is. `mutating` means fitting changes the head's own state.
public protocol Head: Sendable {
    /// Fits (or refits) the head from every embedding in the store, replacing any earlier fit.
    mutating func fit(store: EmbeddingStore)

    /// Probability for each taught class (values sum to 1).
    /// Returns an empty dictionary if nothing is taught or the vector can't be compared
    /// (wrong length, or all zeros).
    func predict(_ vector: [Float]) -> [String: Float]
}

extension Head {
    /// The most probable class and its probability, or nil if `predict` returned nothing.
    /// An `extension` adds methods to a type or protocol, so every head gets this for free.
    public func bestClass(_ vector: [Float]) -> (className: String, probability: Float)? {
        // Sorting by name first makes ties deterministic (dictionary order is random).
        predict(vector)
            .sorted { $0.key < $1.key }
            .max { $0.value < $1.value }
            .map { (className: $0.key, probability: $0.value) }
    }
}

/// Softmax turns raw scores into probabilities that sum to 1.
/// Subtracting the max first avoids overflow in `exp` and doesn't change the result.
func softmax(_ scores: [Float]) -> [Float] {
    guard let maxScore = scores.max() else { return [] }
    let exps = scores.map { exp($0 - maxScore) }
    let total = exps.reduce(0, +)
    return exps.map { $0 / total }
}
