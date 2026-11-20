import PhotosUI
import SwiftUI

/// Experiment 8 tool: import photos for a class with a "day" label, embed them, store only the
/// vectors, and export class + day + embedding. Photos are never saved or exported.
struct RealPhotoView: View {
    nonisolated struct Item: Codable { let `class`: String; let day: String; let embedding: [Float] }
    nonisolated struct Export: Codable { let backbone: String; let items: [Item] }

    @State private var className = ""
    @State private var day = ""
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var items: [Item] = RealPhotoView.load()
    @State private var isWorking = false
    @State private var message: String?

    private static var fileURL: URL { AppModel.documentsURL.appendingPathComponent("real_photo_embeddings.json") }
    private var canImport: Bool { !className.trimmingCharacters(in: .whitespaces).isEmpty && !day.trimmingCharacters(in: .whitespaces).isEmpty && !pickerItems.isEmpty && !isWorking }

    var body: some View {
        Form {
            Section {
                Explainer(title: "What is this for?", color: Theme.sky) {
                    Text("The study's last question is whether the app still works on your own photos taken on different days and in different light. Pick a class name and a day label such as day1, choose photos, and I'll store only their numbers.")
                    Text("Your photos are never saved or exported. On the laptop, a script trains on some days and tests on the others.")
                }
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets(top: 4, leading: 4, bottom: 4, trailing: 4))
            }
            Section("Import") {
                TextField("Class (for example: mug)", text: $className).textInputAutocapitalization(.never)
                TextField("Day label (for example: day1)", text: $day).textInputAutocapitalization(.never)
                PhotosPicker(selection: $pickerItems, matching: .images) {
                    Label("Choose photos (\(pickerItems.count))", systemImage: "photo.on.rectangle")
                }
                Button("Embed and store") { Task { await importPhotos() } }.disabled(!canImport)
                if isWorking { ProgressView() }
                if let message { Text(message).font(.footnote) }
            }
            Section("Stored (vectors only)") {
                ForEach(summaryLines, id: \.self) { Text($0).font(.footnote) }
                if items.isEmpty { Text("Nothing stored yet.").foregroundStyle(.secondary) }
            }
            if !items.isEmpty {
                Section {
                    if let url = try? exportFile() { ShareLink(item: url) { Label("Export JSON", systemImage: "square.and.arrow.up") } }
                    Button("Delete all stored vectors", role: .destructive) { items = []; Self.save([]) }
                }
            }
        }
        .scrollContentBackground(.hidden)
        .background(ScreenBackground(color: Theme.grape))
        .navigationTitle("Real-photo test")
    }

    private var summaryLines: [String] {
        let counts = Dictionary(grouping: items, by: { "\($0.class) / \($0.day)" }).mapValues(\.count)
        return counts.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value) photos" }
    }

    private func importPhotos() async {
        isWorking = true
        defer { isWorking = false }
        let name = className.trimmingCharacters(in: .whitespaces), dayLabel = day.trimmingCharacters(in: .whitespaces)
        var added = 0
        for picked in pickerItems {
            guard let data = try? await picked.loadTransferable(type: Data.self), let image = ImageLoading.image(from: data),
                  let r = try? await EmbeddingService.shared.embed(image) else { continue }
            items.append(Item(class: name, day: dayLabel, embedding: r.vector))
            added += 1
        }
        pickerItems = []
        Self.save(items)
        message = "Stored \(added) vector\(added == 1 ? "" : "s"). The photos themselves were discarded."
    }

    private func exportFile() throws -> URL {
        let backbone = Backbone(rawValue: AppSettings.backboneRaw)?.rawValue ?? AppSettings.defaultBackbone
        return try LabFiles.write(JSONEncoder().encode(Export(backbone: backbone, items: items)), name: "real_photos_export.json")
    }

    private static func load() -> [Item] {
        (try? JSONDecoder().decode([Item].self, from: Data(contentsOf: fileURL))) ?? []
    }
    private static func save(_ items: [Item]) {
        try? JSONEncoder().encode(items).write(to: fileURL, options: .atomic)
    }
}
