import SwiftUI

/// Embeds the 20 bundled gate images and exports the vectors as JSON, so
/// research/compare_device.py can compare them with the Python embeddings of the same files.
struct GateCheckView: View {
    nonisolated struct Export: Codable {
        struct Item: Codable { let file: String; let embedding: [Float] }
        let backbone: String
        let device: String
        let build: String
        let seconds_per_image: Double
        let items: [Item]
    }

    @State private var status = "Not run yet."
    @State private var exportURLs: [URL] = []
    @State private var isRunning = false

    var body: some View {
        Form {
            Section {
                Button("Embed the 20 bundled images") { Task { await run() } }
                    .disabled(isRunning)
                Text(status).font(.footnote)
            } footer: {
                Text("Writes one JSON file per installed backbone to the app's Documents folder. Share it to your Mac and run research/compare_device.py.")
            }
            ForEach(exportURLs, id: \.self) { url in
                ShareLink(item: url) { Label(url.lastPathComponent, systemImage: "square.and.arrow.up") }
            }
        }
        .navigationTitle("Gate check")
    }

    private func run() async {
        isRunning = true
        defer { isRunning = false }
        let files = (Bundle.main.urls(forResourcesWithExtension: "jpg", subdirectory: nil) ?? [])
            .filter { $0.lastPathComponent.hasPrefix("gate_") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { status = "No gate images are bundled in this build."; return }
        exportURLs = []
        var lines: [String] = []
        for backbone in Backbone.installed {
            var items: [Export.Item] = []
            var total = 0.0
            for file in files {
                guard let image = UIImage(contentsOfFile: file.path) else { continue }
                do {
                    let r = try await EmbeddingService.shared.embed(image, backbone: backbone)
                    items.append(.init(file: file.lastPathComponent, embedding: r.vector))
                    total += r.seconds
                } catch { status = "Failed on \(file.lastPathComponent): \(error.localizedDescription)"; return }
            }
            let export = Export(backbone: backbone.rawValue, device: LabFiles.deviceDescription,
                                build: LabFiles.buildConfiguration, seconds_per_image: total / Double(max(items.count, 1)),
                                items: items)
            do {
                let data = try JSONEncoder().encode(export)
                exportURLs.append(try LabFiles.write(data, name: "gate_device_\(backbone.rawValue).json"))
                let head = items.first?.embedding.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ") ?? ""
                lines.append("\(backbone.displayName): \(items.count) images, \(String(format: "%.0f", export.seconds_per_image * 1000)) ms each. First vector starts [\(head), …]")
            } catch { status = "Could not write the file: \(error.localizedDescription)"; return }
        }
        status = lines.joined(separator: "\n")
    }
}
