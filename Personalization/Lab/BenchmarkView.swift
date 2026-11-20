import PersonalizeCore
import SwiftUI

/// Experiment 7: timings and memory on this device. Protocol: results/measurement_protocol.md.
struct BenchmarkView: View {
    nonisolated struct Row { let metric: String; let iteration: Int; let seconds: Double }

    @State private var summary: [String] = []
    @State private var csvURL: URL?
    @State private var isRunning = false
    @State private var progress = ""

    // `nonisolated` so the background timing task can read them without hopping to the main actor.
    nonisolated private static let warmups = 10
    nonisolated private static let repeats = 100

    var body: some View {
        Form {
            Section {
                Button("Run benchmark") { Task { await run() } }.disabled(isRunning)
                if isRunning { ProgressView(); Text(progress).font(.footnote) }
            } footer: {
                Text("\(Self.warmups) warm-up runs are discarded, then \(Self.repeats) timed runs each. Use a Release build on a real iPhone.")
            }
            if !summary.isEmpty {
                Section("Results (median / 95th percentile)") {
                    ForEach(summary, id: \.self) { Text($0).font(.footnote.monospaced()) }
                }
            }
            if let csvURL {
                ShareLink(item: csvURL) { Label("Export raw timings (CSV)", systemImage: "square.and.arrow.up") }
            }
        }
        .navigationTitle("Benchmark")
    }

    private func run() async {
        isRunning = true
        defer { isRunning = false }
        guard let imageURL = (Bundle.main.urls(forResourcesWithExtension: "jpg", subdirectory: nil) ?? [])
            .first(where: { $0.lastPathComponent.hasPrefix("gate_") }),
              let image = UIImage(contentsOfFile: imageURL.path) else {
            progress = "No bundled test image found."
            return
        }
        let thermalBefore = Self.thermalName
        let memoryBefore = Self.footprintMB()
        var rows: [Row] = []

        // 1. Embedding one photo (preprocessing + Core ML), per installed backbone.
        for backbone in Backbone.installed {
            progress = "Embedding with \(backbone.displayName)…"
            for i in 0..<(Self.warmups + Self.repeats) {
                guard let r = try? await EmbeddingService.shared.embed(image, backbone: backbone) else { continue }
                if i >= Self.warmups { rows.append(Row(metric: "embed_\(backbone.rawValue)", iteration: i - Self.warmups, seconds: r.seconds)) }
            }
        }

        // 2. Learner.teach for 5 and 20 photos on top of 10 existing classes x 5 examples (dimension 960).
        for kind in HeadKind.allCases {
            for photos in [5, 20] {
                progress = "Teaching \(photos) photos with \(kind.rawValue)…"
                let timings = await Task.detached(priority: .userInitiated) { () -> [Double] in
                    var base = Learner(headKind: kind)
                    var rng = SystemRandomNumberGenerator()
                    func vectors(_ n: Int) -> [[Float]] { (0..<n).map { _ in (0..<960).map { _ in Float.random(in: 0...1, using: &rng) } } }
                    for c in 0..<10 { try? base.teach(className: "base\(c)", embeddings: vectors(5)) }
                    var out: [Double] = []
                    for i in 0..<(Self.warmups + Self.repeats) {
                        var copy = base
                        try? copy.teach(className: "new", embeddings: vectors(photos))
                        if i >= Self.warmups { out.append(copy.lastFitSeconds) }
                    }
                    return out
                }.value
                for (i, s) in timings.enumerated() { rows.append(Row(metric: "teach_\(photos)_\(kind.rawValue)", iteration: i, seconds: s)) }
            }
        }

        let thermalAfter = Self.thermalName
        let memoryAfter = Self.footprintMB()
        let metrics = Array(Set(rows.map(\.metric))).sorted()
        summary = metrics.map { m in
            let v = rows.filter { $0.metric == m }.map(\.seconds).sorted()
            return String(format: "%-28@ %7.1f / %7.1f ms", m as NSString, Self.percentile(v, 0.5) * 1000, Self.percentile(v, 0.95) * 1000)
        }
        summary.append("thermal: \(thermalBefore) -> \(thermalAfter)")
        summary.append(String(format: "memory footprint: %.0f -> %.0f MB", memoryBefore, memoryAfter))
        summary.append("build: \(LabFiles.buildConfiguration), device: \(LabFiles.deviceDescription)")

        var csv = "device,build,thermal_before,thermal_after,memory_mb_before,memory_mb_after,metric,iteration,seconds\n"
        for r in rows {
            csv += "\(LabFiles.deviceDescription),\(LabFiles.buildConfiguration),\(thermalBefore),\(thermalAfter),"
            csv += String(format: "%.1f,%.1f,%@,%d,%.6f\n", memoryBefore, memoryAfter, r.metric, r.iteration, r.seconds)
        }
        csvURL = try? LabFiles.write(Data(csv.utf8), name: "benchmark_\(LabFiles.buildConfiguration).csv")
        progress = "Done."
    }

    nonisolated static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        // Nearest-rank percentile: the smallest value with at least p of the data at or below it.
        let rank = Int((Double(sorted.count) * p).rounded(.up))
        return sorted[max(0, min(sorted.count - 1, rank - 1))]
    }

    static var thermalName: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// Memory the app is using, in MB (the same number Xcode's memory gauge shows).
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : 0
    }
}
