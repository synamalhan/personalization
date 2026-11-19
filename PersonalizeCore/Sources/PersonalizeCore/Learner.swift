import Foundation

/// Which classifier head a `Learner` uses.
public enum HeadKind: String, CaseIterable, Codable, Sendable {
    case nearestClassMean
    case logistic
}

/// Result of `Learner.predict`.
public enum LearnerPrediction: Equatable, Sendable {
    /// A taught class and the head's probability for it (0...1).
    case known(className: String, confidence: Float)
    /// Too uncertain to name. `bestGuess` is the top class, if any; no confidence is claimed.
    case unknown(bestGuess: String?)
}

/// What the app talks to: stores embeddings, keeps a head fit on them, and predicts.
public struct Learner: Sendable {
    public private(set) var store: EmbeddingStore

    /// Which head is used. Changing it refits immediately.
    public var headKind: HeadKind {
        didSet { if headKind != oldValue { refit() } }
    }

    /// When true (default) the head is fit on ALL stored embeddings, so old classes are
    /// replayed. When false it is fit only on the newest class; that exists only to
    /// measure forgetting in experiments. Changing it refits immediately.
    public var replay: Bool {
        didSet { if replay != oldValue { refit() } }
    }

    /// Minimum head probability for a prediction to count as known.
    public var threshold: Float

    /// Seconds the last `teach`, `deleteClass` or refit took.
    public private(set) var lastFitSeconds: Double = 0

    private var head: any Head
    /// The class taught most recently (what the head sees when `replay` is false).
    private var newestClass: String?

    public init(headKind: HeadKind = .nearestClassMean, replay: Bool = true, threshold: Float = 0.5) {
        self.store = EmbeddingStore()
        self.headKind = headKind
        self.replay = replay
        self.threshold = threshold
        self.head = Self.makeHead(headKind)
    }

    /// Starts from a previously saved store (for example one loaded from disk).
    public init(store: EmbeddingStore, headKind: HeadKind = .nearestClassMean, replay: Bool = true, threshold: Float = 0.5) {
        self.init(headKind: headKind, replay: replay, threshold: threshold)
        self.store = store
        self.newestClass = store.classNames.last
        refit()
    }

    public var classNames: [String] { store.classNames }

    public func count(for className: String) -> Int { store.count(for: className) }

    /// Adds the vectors to the class, then refits the head.
    /// Throws (and changes nothing) if a vector has the wrong length.
    public mutating func teach(className: String, embeddings: [[Float]]) throws {
        let start = Self.now()
        try store.add(className: className, vectors: embeddings)
        newestClass = className
        fitHead()
        lastFitSeconds = Self.now() - start
    }

    /// Removes a class's embeddings, then refits the head.
    public mutating func deleteClass(_ className: String) {
        let start = Self.now()
        store.removeClass(className)
        if newestClass == className { newestClass = nil }
        fitHead()
        lastFitSeconds = Self.now() - start
    }

    /// Names the class, or `.unknown` when the top probability is below `threshold`.
    ///
    /// With fewer than two taught classes a probability is meaningless (it is always 1),
    /// so the result is `.unknown` rather than a made-up confidence.
    public func predict(_ embedding: [Float]) -> LearnerPrediction {
        let probabilities = head.predict(embedding)
        guard let best = head.bestClass(embedding) else { return .unknown(bestGuess: nil) }
        if probabilities.count < 2 || best.probability < threshold {
            return .unknown(bestGuess: best.className)
        }
        return .known(className: best.className, confidence: best.probability)
    }

    /// Raw per-class probabilities from the head, for experiments.
    public func probabilities(for embedding: [Float]) -> [String: Float] {
        head.predict(embedding)
    }

    // MARK: - Fitting

    private mutating func refit() {
        let start = Self.now()
        head = Self.makeHead(headKind)
        fitHead()
        lastFitSeconds = Self.now() - start
    }

    /// Fits on everything (replay) or only the newest class.
    private mutating func fitHead() {
        var source = store
        if !replay {
            source = EmbeddingStore()
            if let name = newestClass, let vectors = store.embeddings[name] {
                // Same dimension as the main store, so this can't fail.
                try? source.add(className: name, vectors: vectors)
            }
        }
        head.fit(store: source)
    }

    private static func makeHead(_ kind: HeadKind) -> any Head {
        switch kind {
        case .nearestClassMean: NearestClassMeanHead()
        case .logistic: LogisticHead()
        }
    }

    /// Monotonic seconds (unaffected by clock changes), used to time fits.
    private static func now() -> Double {
        ProcessInfo.processInfo.systemUptime
    }
}
