import Foundation

/// Multinomial logistic (softmax) regression fit with full-batch gradient descent.
///
/// Inputs are L2-normalized. Weights start at zero and the settings are fixed, so fitting
/// the same store twice gives identical results.
public struct LogisticHead: Head {
    /// Number of gradient descent steps; each step uses every training vector.
    public let steps: Int
    public let learningRate: Float
    /// L2 regularization strength: pulls weights toward zero to limit overfitting.
    /// Biases are not regularized.
    public let regularization: Float

    private var classNames: [String] = []
    private var weights: [[Float]] = []   // [class][feature]
    private var biases: [Float] = []      // [class]
    private var dimension = 0

    public init(steps: Int = 300, learningRate: Float = 1.0, regularization: Float = 1e-4) {
        self.steps = steps
        self.learningRate = learningRate
        self.regularization = regularization
    }

    /// Fits from every class in the store. With no usable data the head stays empty;
    /// with one class it needs no training and always answers that class with probability 1.
    public mutating func fit(store: EmbeddingStore) {
        classNames = []
        weights = []
        biases = []
        dimension = 0

        // Gather normalized inputs and a class index for each (skipping zero-length vectors).
        var inputs: [[Float]] = []
        var labels: [Int] = []
        var names: [String] = []
        for name in store.classNames {
            let vectors = (store.embeddings[name] ?? []).compactMap(NearestClassMeanHead.normalized)
            guard !vectors.isEmpty else { continue }
            names.append(name)
            inputs.append(contentsOf: vectors)
            labels.append(contentsOf: Array(repeating: names.count - 1, count: vectors.count))
        }
        guard let dim = inputs.first?.count else { return }
        classNames = names
        dimension = dim
        weights = Array(repeating: Array(repeating: 0, count: dim), count: names.count)
        biases = Array(repeating: 0, count: names.count)

        // One class: probability is always 1, so there is nothing to learn.
        guard names.count > 1 else { return }

        let sampleCount = Float(inputs.count)
        for _ in 0..<steps {
            var weightGrad = Array(repeating: Array(repeating: Float(0), count: dim), count: names.count)
            var biasGrad = Array(repeating: Float(0), count: names.count)

            // Accelerate (vDSP / BLAS matrix multiply) could replace these loops later:
            // the forward pass is one matrix product and the gradient is another.
            for (x, label) in zip(inputs, labels) {
                var probs = softmax((0..<names.count).map { logit(for: x, class: $0) })
                probs[label] -= 1   // (p - onehot) is the gradient of the loss w.r.t. the logits
                for c in 0..<names.count {
                    biasGrad[c] += probs[c]
                    for j in 0..<dim { weightGrad[c][j] += probs[c] * x[j] }
                }
            }

            for c in 0..<names.count {
                biases[c] -= learningRate * biasGrad[c] / sampleCount
                for j in 0..<dim {
                    weights[c][j] -= learningRate * (weightGrad[c][j] / sampleCount + regularization * weights[c][j])
                }
            }
        }
    }

    public func predict(_ vector: [Float]) -> [String: Float] {
        guard !classNames.isEmpty, vector.count == dimension,
              let x = NearestClassMeanHead.normalized(vector) else { return [:] }
        let probs = softmax((0..<classNames.count).map { logit(for: x, class: $0) })
        return Dictionary(uniqueKeysWithValues: zip(classNames, probs))
    }

    private func logit(for x: [Float], class c: Int) -> Float {
        NearestClassMeanHead.dot(weights[c], x) + biases[c]
    }
}
