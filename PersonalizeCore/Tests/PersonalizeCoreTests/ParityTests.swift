import Foundation
import Testing
@testable import PersonalizeCore

/// Checks the Swift heads against outputs exported from the Python reference
/// (research/export_fixture.py). Tolerance is fixed at 1e-4; do not loosen it.
private struct Fixture: Decodable {
    struct Config: Decodable {
        struct Logistic: Decodable {
            let steps: Int
            let learning_rate: Float
            let regularization: Float
        }
        struct NCM: Decodable { let temperature: Float }
        let logistic: Logistic
        let nearest_class_mean: NCM
    }
    struct Query: Decodable {
        let vector: [Float]
        let true_class: String
        let taught: Bool
    }
    struct Expected: Decodable {
        let predicted: String
        let probabilities: [String: Float]
        let similarities: [String: Float]?
    }
    struct Outputs: Decodable {
        let nearest_class_mean: [Expected]
        let logistic: [Expected]
    }
    let config: Config
    let classes: [String]
    let stored: [String: [[Float]]]
    let queries: [Query]
    let expected: Outputs

    static func load() throws -> Fixture {
        let url = try #require(Bundle.module.url(forResource: "parity", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    func makeStore() throws -> EmbeddingStore {
        var store = EmbeddingStore()
        for name in classes { try store.add(className: name, vectors: stored[name] ?? []) }
        return store
    }
}

private let tolerance: Float = 1e-4

@Suite struct ParityTests {
    @Test func fixtureIsWellFormed() throws {
        let f = try Fixture.load()
        #expect(f.classes.count == 5)
        #expect(f.queries.count == 20)
        #expect(f.expected.logistic.count == 20 && f.expected.nearest_class_mean.count == 20)
        // Defaults must equal the settings the Python reference used.
        #expect(f.config.logistic.steps == LogisticHead().steps)
        #expect(f.config.logistic.learning_rate == LogisticHead().learningRate)
        #expect(f.config.logistic.regularization == LogisticHead().regularization)
        #expect(f.config.nearest_class_mean.temperature == NearestClassMeanHead().temperature)
    }

    @Test func nearestClassMeanMatchesPython() throws {
        let f = try Fixture.load()
        let head = NearestClassMeanHead(store: try f.makeStore(), temperature: f.config.nearest_class_mean.temperature)
        for (i, (query, want)) in zip(f.queries, f.expected.nearest_class_mean).enumerated() {
            let got = head.predict(query.vector)
            #expect(head.bestClass(query.vector)?.className == want.predicted, "query \(i) class")
            for (name, p) in want.probabilities {
                let diff = abs((got[name] ?? .nan) - p)
                #expect(diff <= tolerance, "query \(i) \(name): swift \(got[name] ?? .nan) python \(p) diff \(diff)")
            }
            // Best cosine similarity must match too.
            if let best = head.bestMatch(query.vector), let sims = want.similarities {
                let diff = abs(best.similarity - (sims[best.className] ?? .nan))
                #expect(diff <= tolerance, "query \(i) similarity diff \(diff)")
            }
        }
    }

    @Test func logisticMatchesPython() throws {
        let f = try Fixture.load()
        let c = f.config.logistic
        var head = LogisticHead(steps: c.steps, learningRate: c.learning_rate, regularization: c.regularization)
        head.fit(store: try f.makeStore())
        for (i, (query, want)) in zip(f.queries, f.expected.logistic).enumerated() {
            let got = head.predict(query.vector)
            #expect(head.bestClass(query.vector)?.className == want.predicted, "query \(i) class")
            for (name, p) in want.probabilities {
                let diff = abs((got[name] ?? .nan) - p)
                #expect(diff <= tolerance, "query \(i) \(name): swift \(got[name] ?? .nan) python \(p) diff \(diff)")
            }
        }
    }
}
