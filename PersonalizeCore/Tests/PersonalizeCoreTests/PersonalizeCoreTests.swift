import Foundation
import Testing
@testable import PersonalizeCore

// Small hand-made 2D vectors: "cat" points along x, "dog" along y.
private func makeStore() throws -> EmbeddingStore {
    var store = EmbeddingStore()
    try store.add(className: "cat", vectors: [[1, 0], [2, 0.2], [3, -0.2]])
    try store.add(className: "dog", vectors: [[0, 1], [0.2, 2], [-0.2, 3]])
    return store
}

@Suite struct EmbeddingStoreTests {
    @Test func addAndCount() throws {
        let store = try makeStore()
        #expect(store.classNames == ["cat", "dog"])
        #expect(store.count(for: "cat") == 3)
        #expect(store.count(for: "bird") == 0)
    }

    @Test func addAppendsToExistingClass() throws {
        var store = try makeStore()
        try store.add(className: "cat", vectors: [[1, 1]])
        #expect(store.count(for: "cat") == 4)
    }

    @Test func rejectsWrongDimension() throws {
        var store = try makeStore()
        #expect(throws: EmbeddingStoreError.dimensionMismatch(expected: 2, got: 3)) {
            try store.add(className: "bird", vectors: [[1, 2, 3]])
        }
        #expect(store.count(for: "bird") == 0)
    }

    @Test func removeClass() throws {
        var store = try makeStore()
        store.removeClass("cat")
        #expect(store.classNames == ["dog"])
        #expect(store.count(for: "cat") == 0)
        store.removeClass("missing") // no crash
        #expect(store.classNames == ["dog"])
    }

    @Test func deleteLeavesOtherPredictionsUnchanged() throws {
        let full = try makeStore()
        var reduced = full
        reduced.removeClass("cat")
        let before = NearestClassMeanHead(store: full).bestMatch([0.1, 1])
        let after = NearestClassMeanHead(store: reduced).bestMatch([0.1, 1])
        #expect(before == after)
        #expect(NearestClassMeanHead(store: reduced).bestMatch([1, 0])?.className == "dog")
    }

    @Test func saveLoadRoundTrip() throws {
        let store = try makeStore()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("store-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try store.save(to: url)
        let loaded = try EmbeddingStore.load(from: url)
        #expect(loaded == store)
    }

    @Test func loadMissingFileThrows() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("nope-\(UUID().uuidString).json")
        #expect(throws: (any Error).self) { try EmbeddingStore.load(from: url) }
    }
}

@Suite struct NearestClassMeanHeadTests {
    @Test func predictsClosestClass() throws {
        let head = NearestClassMeanHead(store: try makeStore())
        let cat = try #require(head.bestMatch([5, 0.1]))
        #expect(cat.className == "cat")
        #expect(cat.similarity > 0.99)
        #expect(head.bestMatch([0.1, 5])?.className == "dog")
    }

    @Test func similarityIgnoresVectorLength() throws {
        let head = NearestClassMeanHead(store: try makeStore())
        let a = try #require(head.bestMatch([1, 0]))
        let b = try #require(head.bestMatch([100, 0]))
        #expect(abs(a.similarity - b.similarity) < 1e-5)
    }

    @Test func orthogonalVectorHasZeroSimilarity() throws {
        var store = EmbeddingStore()
        try store.add(className: "x", vectors: [[1, 0]])
        let p = try #require(NearestClassMeanHead(store: store).bestMatch([0, 1]))
        #expect(abs(p.similarity) < 1e-6)
    }

    @Test func emptyStore() {
        let head = NearestClassMeanHead(store: EmbeddingStore())
        #expect(head.bestMatch([1, 0]) == nil)
        #expect(head.classify([1, 0], threshold: 0.5) == .unknown(bestGuess: nil))
        #expect(NearestClassMeanHead.suggestedThreshold(from: EmbeddingStore()) == nil)
    }

    @Test func oneClassStore() throws {
        var store = EmbeddingStore()
        try store.add(className: "cat", vectors: [[1, 0], [1, 0.1]])
        let head = NearestClassMeanHead(store: store)
        #expect(head.bestMatch([1, 0])?.className == "cat")
        #expect(head.classify([1, 0], threshold: 0.9) == .known(head.bestMatch([1, 0])!))
        // Far from the only class -> unknown, not a forced "cat".
        if case .unknown(let guess) = head.classify([0, 1], threshold: 0.9) {
            #expect(guess?.className == "cat")
        } else {
            Issue.record("expected unknown")
        }
        #expect(NearestClassMeanHead.suggestedThreshold(from: store) != nil)
    }

    @Test func oneVectorPerClassHasNoThreshold() throws {
        var store = EmbeddingStore()
        try store.add(className: "cat", vectors: [[1, 0]])
        #expect(NearestClassMeanHead.suggestedThreshold(from: store) == nil)
    }

    @Test func zeroAndWrongSizeQueriesDoNotCrash() throws {
        let head = NearestClassMeanHead(store: try makeStore())
        #expect(head.bestMatch([0, 0]) == nil)
        #expect(head.bestMatch([]) == nil)
        #expect(head.bestMatch([1, 0, 0]) == nil)
    }

    @Test func unknownBelowThreshold() throws {
        let head = NearestClassMeanHead(store: try makeStore())
        // [1,1] is ~0.707 similar to both classes.
        #expect(head.classify([1, 1], threshold: 0.9) != .known(Prediction(className: "cat", similarity: 0.7)))
        if case .unknown(let guess) = head.classify([1, 1], threshold: 0.9) {
            #expect(guess != nil)
        } else {
            Issue.record("expected unknown")
        }
        if case .known = head.classify([1, 1], threshold: 0.5) {} else {
            Issue.record("expected known")
        }
    }

    @Test func suggestedThresholdSeparatesOwnFromFar() throws {
        let store = try makeStore()
        let threshold = try #require(NearestClassMeanHead.suggestedThreshold(from: store))
        #expect(threshold > 0.5 && threshold <= 1)
        let head = NearestClassMeanHead(store: store)
        // A typical cat is accepted; a vector between the classes is rejected.
        if case .known = head.classify([2, 0.1], threshold: threshold) {} else {
            Issue.record("typical cat rejected")
        }
        if case .unknown = head.classify([1, 1], threshold: threshold) {} else {
            Issue.record("in-between vector accepted")
        }
    }
}

// Three well-separated classes in 3D.
private func makeSeparatedStore() throws -> EmbeddingStore {
    var store = EmbeddingStore()
    try store.add(className: "a", vectors: [[1, 0.1, 0], [2, 0, 0.2], [3, 0.2, -0.1]])
    try store.add(className: "b", vectors: [[0.1, 1, 0], [0, 2, 0.2], [0.2, 3, -0.1]])
    try store.add(className: "c", vectors: [[0, 0.1, 1], [0.2, 0, 2], [-0.1, 0.2, 3]])
    return store
}

@Suite struct HeadProtocolTests {
    /// Runs the same checks on any head, which is the point of the shared protocol.
    private func checkHead<H: Head>(_ makeHead: () -> H) throws {
        let store = try makeSeparatedStore()
        var head = makeHead()
        #expect(head.predict([1, 0, 0]).isEmpty) // not fit yet
        head.fit(store: store)
        for (name, vectors) in store.embeddings {
            for vector in vectors {
                let probs = head.predict(vector)
                #expect(probs.count == 3)
                #expect(abs(probs.values.reduce(0, +) - 1) < 1e-4)
                #expect(head.bestClass(vector)?.className == name)
            }
        }
    }

    @Test func nearestClassMeanConforms() throws { try checkHead { NearestClassMeanHead() } }
    @Test func logisticConforms() throws { try checkHead { LogisticHead() } }
}

@Suite struct LogisticHeadTests {
    @Test func classifiesNewPointsNearEachClass() throws {
        var head = LogisticHead()
        head.fit(store: try makeSeparatedStore())
        #expect(head.bestClass([5, 0.3, 0.1])?.className == "a")
        #expect(head.bestClass([0.1, 5, 0.3])?.className == "b")
        #expect(head.bestClass([0.3, 0.1, 5])?.className == "c")
    }

    @Test func fittingTwiceIsIdentical() throws {
        let store = try makeSeparatedStore()
        var first = LogisticHead()
        var second = LogisticHead()
        first.fit(store: store)
        second.fit(store: store)
        let query: [Float] = [1, 2, 3]
        #expect(first.predict(query) == second.predict(query))
        first.fit(store: store) // refitting the same head also resets cleanly
        #expect(first.predict(query) == second.predict(query))
    }

    @Test func settingsAreConfigurable() throws {
        let store = try makeSeparatedStore()
        var short = LogisticHead(steps: 1)
        var long = LogisticHead(steps: 300)
        short.fit(store: store)
        long.fit(store: store)
        let q: [Float] = [1, 0, 0]
        #expect(try #require(long.predict(q)["a"]) > (try #require(short.predict(q)["a"])))
    }

    @Test func zeroStepsGivesUniformProbabilities() throws {
        var head = LogisticHead(steps: 0)
        head.fit(store: try makeSeparatedStore())
        let probs = head.predict([1, 0, 0])
        for p in probs.values { #expect(abs(p - 1.0 / 3) < 1e-6) }
    }

    @Test func emptyStore() {
        var head = LogisticHead()
        head.fit(store: EmbeddingStore())
        #expect(head.predict([1, 0]).isEmpty)
        #expect(head.bestClass([1, 0]) == nil)
    }

    @Test func oneClassStore() throws {
        var store = EmbeddingStore()
        try store.add(className: "cat", vectors: [[1, 0], [1, 0.1]])
        var head = LogisticHead()
        head.fit(store: store)
        #expect(head.predict([0, 1]) == ["cat": 1])
        #expect(head.bestClass([1, 0])?.className == "cat")
    }

    @Test func badQueriesDoNotCrash() throws {
        var head = LogisticHead()
        head.fit(store: try makeSeparatedStore())
        #expect(head.predict([0, 0, 0]).isEmpty)
        #expect(head.predict([1, 0]).isEmpty)
        #expect(head.predict([]).isEmpty)
    }

    @Test func refitAfterDeletingClass() throws {
        var store = try makeSeparatedStore()
        var head = LogisticHead()
        head.fit(store: store)
        store.removeClass("c")
        head.fit(store: store)
        #expect(Set(head.predict([1, 0, 0]).keys) == ["a", "b"])
    }
}

@Suite struct LearnerTests {
    private func teachThree(_ learner: inout Learner) throws {
        let store = try makeSeparatedStore()
        for name in store.classNames {
            try learner.teach(className: name, embeddings: store.embeddings[name] ?? [])
        }
    }

    @Test(arguments: HeadKind.allCases)
    func oldClassesStillRecognizedWithReplay(kind: HeadKind) throws {
        var learner = Learner(headKind: kind, replay: true, threshold: 0.5)
        try teachThree(&learner)
        guard case .known(let a, _) = learner.predict([5, 0.2, 0]) else { Issue.record("a unknown"); return }
        guard case .known(let b, _) = learner.predict([0.1, 5, 0]) else { Issue.record("b unknown"); return }
        guard case .known(let c, let conf) = learner.predict([0, 0.1, 5]) else { Issue.record("c unknown"); return }
        #expect([a, b, c] == ["a", "b", "c"])
        #expect(conf >= 0.5 && conf <= 1)
        #expect(learner.classNames == ["a", "b", "c"])
        #expect(learner.count(for: "a") == 3)
    }

    @Test(arguments: HeadKind.allCases)
    func deleteClassRemovesItAndOthersStillWork(kind: HeadKind) throws {
        var learner = Learner(headKind: kind)
        try teachThree(&learner)
        learner.deleteClass("b")
        #expect(learner.classNames == ["a", "c"])
        #expect(learner.count(for: "b") == 0)
        #expect(learner.probabilities(for: [0, 5, 0])["b"] == nil)
        if case .known(let name, _) = learner.predict([5, 0, 0]) { #expect(name == "a") } else { Issue.record("a unknown") }
        if case .known(let name, _) = learner.predict([0, 0, 5]) { #expect(name == "c") } else { Issue.record("c unknown") }
    }

    @Test(arguments: HeadKind.allCases)
    func emptyState(kind: HeadKind) {
        var learner = Learner(headKind: kind)
        #expect(learner.predict([1, 0, 0]) == .unknown(bestGuess: nil))
        learner.deleteClass("nothing") // no crash
        #expect(learner.classNames.isEmpty)
    }

    @Test(arguments: HeadKind.allCases)
    func oneClassState(kind: HeadKind) throws {
        var learner = Learner(headKind: kind)
        try learner.teach(className: "cat", embeddings: [[1, 0], [1, 0.1]])
        // No confidence is claimed with a single class.
        #expect(learner.predict([1, 0]) == .unknown(bestGuess: "cat"))
        learner.deleteClass("cat")
        #expect(learner.predict([1, 0]) == .unknown(bestGuess: nil))
    }

    @Test func farFromEverythingIsUnknownWithHighThreshold() throws {
        var learner = Learner(headKind: .nearestClassMean, threshold: 0.99)
        try teachThree(&learner)
        // Equidistant from all three classes: probabilities near 1/3.
        if case .unknown(let guess) = learner.predict([1, 1, 1]) { #expect(guess != nil) } else { Issue.record("expected unknown") }
    }

    @Test func noReplayFitsOnlyNewestClass() throws {
        var learner = Learner(headKind: .logistic, replay: false)
        try teachThree(&learner)
        // Head only knows "c"; stored data still has all three.
        #expect(Set(learner.probabilities(for: [5, 0, 0]).keys) == ["c"])
        #expect(learner.classNames == ["a", "b", "c"])
    }

    @Test func togglingReplayRefits() throws {
        var learner = Learner(headKind: .nearestClassMean, replay: false)
        try teachThree(&learner)
        learner.replay = true
        #expect(Set(learner.probabilities(for: [5, 0, 0]).keys) == ["a", "b", "c"])
    }

    @Test func switchingHeadRefits() throws {
        var learner = Learner(headKind: .nearestClassMean)
        try teachThree(&learner)
        let before = learner.probabilities(for: [5, 0.2, 0])
        learner.headKind = .logistic
        #expect(learner.probabilities(for: [5, 0.2, 0]) != before)
        #expect(learner.probabilities(for: [5, 0.2, 0]).count == 3)
    }

    @Test func recordsFitTimeAndRejectsBadVectors() throws {
        var learner = Learner()
        #expect(learner.lastFitSeconds == 0)
        try learner.teach(className: "a", embeddings: [[1, 0]])
        #expect(learner.lastFitSeconds > 0)
        #expect(throws: EmbeddingStoreError.self) {
            try learner.teach(className: "b", embeddings: [[1, 0, 0]])
        }
        #expect(learner.classNames == ["a"])
    }

    @Test func initFromSavedStore() throws {
        let learner = Learner(store: try makeSeparatedStore(), headKind: .logistic)
        if case .known(let name, _) = learner.predict([0, 5, 0]) { #expect(name == "b") } else { Issue.record("unknown") }
    }
}
