import PhotosUI
import SwiftUI
import PersonalizeCore

/// Teach a new class from 3 to 20 photos.
struct TeachView: View {
    @Environment(AppModel.self) private var model
    @AppStorage(AppSettings.deletePhotosKey) private var deletePhotos = true

    @State private var className = ""
    // `PhotosPickerItem` is a handle to a chosen photo; the picture itself is loaded later, one at a time.
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var capturedPhotos: [UIImage] = []
    @State private var showCamera = false
    @State private var isWorking = false
    @State private var progress = ""
    @State private var errorMessage: String?
    @State private var summary: String?

    private static let minPhotos = 3
    private static let maxPhotos = 20

    private var photoCount: Int { pickerItems.count + capturedPhotos.count }
    private var trimmedName: String { className.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var canTeach: Bool { !trimmedName.isEmpty && photoCount >= Self.minPhotos && photoCount <= Self.maxPhotos && !isWorking && !model.isBusy }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("For example: my mug", text: $className)
                        .textInputAutocapitalization(.never)
                } header: { Text("Class name") } footer: {
                    if model.classNames.contains(trimmedName) {
                        Text("“\(trimmedName)” already exists. These photos will be added to it. Names are case-sensitive.")
                    }
                }
                Section {
                    PhotosPicker(selection: $pickerItems, maxSelectionCount: Self.maxPhotos - capturedPhotos.count, matching: .images) {
                        Label("Choose photos", systemImage: "photo.on.rectangle")
                    }
                    if UIImagePickerController.isSourceTypeAvailable(.camera) {
                        Button { showCamera = true } label: { Label("Take a photo", systemImage: "camera") }
                            .disabled(photoCount >= Self.maxPhotos)
                    }
                    Text("\(photoCount) photo\(photoCount == 1 ? "" : "s") selected (\(Self.minPhotos) to \(Self.maxPhotos) needed)")
                        .font(.footnote).foregroundStyle(.secondary)
                    if photoCount > 0 {
                        Button("Clear photos", role: .destructive) { pickerItems = []; capturedPhotos = [] }
                    }
                } header: { Text("Photos") } footer: {
                    Text(deletePhotos ? "Photos are deleted from memory right after they are turned into numbers."
                                      : "Photos will be kept in the app's Documents folder (change this in Settings).")
                }
                Section {
                    Button {
                        Task { await teach() }
                    } label: {
                        HStack {
                            Text("Teach")
                            if isWorking { Spacer(); ProgressView() }
                        }
                    }
                    .disabled(!canTeach)
                    if isWorking { Text(progress).font(.footnote).foregroundStyle(.secondary) }
                    if let summary { Text(summary).font(.footnote) }
                    if let errorMessage { Text(errorMessage).font(.footnote).foregroundStyle(.red) }
                }
            }
            .navigationTitle("Teach")
            .fullScreenCover(isPresented: $showCamera) {
                CameraPicker { image in
                    if let image { capturedPhotos.append(image) }
                }
                .ignoresSafeArea()
            }
        }
    }

    private func teach() async {
        guard canTeach else { return }
        isWorking = true
        errorMessage = nil
        summary = nil
        defer { isWorking = false }
        let name = trimmedName
        let keep = !deletePhotos
        let items = pickerItems, captured = capturedPhotos
        var vectors: [[Float]] = []
        var embedSeconds = 0.0
        var unreadable = 0

        func embedOne(_ image: UIImage, jpeg: Data?) async throws {
            progress = "Embedding photo \(vectors.count + 1) of \(items.count + captured.count)…"
            let result = try await EmbeddingService.shared.embed(image)
            vectors.append(result.vector)
            embedSeconds += result.seconds
            if keep, let data = jpeg ?? image.jpegData(compressionQuality: 0.9) { Self.keepPhoto(data, className: name) }
        }

        do {
            for item in items {
                // Each photo is loaded, embedded, and released before the next one.
                guard let data = try? await item.loadTransferable(type: Data.self), let image = ImageLoading.image(from: data) else {
                    unreadable += 1
                    continue
                }
                try await embedOne(image, jpeg: data)
            }
            for image in captured { try await embedOne(image, jpeg: nil) }
        } catch {
            errorMessage = "Something went wrong while reading the photos: \(error.localizedDescription)"
            return
        }

        guard vectors.count >= Self.minPhotos else {
            errorMessage = "Only \(vectors.count) photo\(vectors.count == 1 ? "" : "s") could be read. Please choose at least \(Self.minPhotos) readable photos."
            return
        }
        progress = "Updating the classifier…"
        do {
            try await model.teach(className: name, embeddings: vectors)
        } catch {
            errorMessage = error.localizedDescription
            return
        }
        pickerItems = []
        capturedPhotos = []
        className = ""
        let skipped = unreadable > 0 ? " (\(unreadable) unreadable photo\(unreadable == 1 ? "" : "s") skipped)" : ""
        summary = String(format: "Learned “%@” from %d photos in %.2f s (embedding %.2f s, refit %.3f s)%@.",
                         name, vectors.count, embedSeconds + model.learner.lastFitSeconds, embedSeconds,
                         model.learner.lastFitSeconds, skipped)
    }

    private static func keepPhoto(_ data: Data, className: String) {
        let dir = AppModel.photoDirectory(for: className)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? data.write(to: dir.appendingPathComponent("\(UUID().uuidString).jpg"))
    }
}

/// Wraps UIKit's camera screen so SwiftUI can show it. A `UIViewControllerRepresentable`
/// is the bridge between UIKit view controllers and SwiftUI.
struct CameraPicker: UIViewControllerRepresentable {
    let onFinish: (UIImage?) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            parent.onFinish(info[.originalImage] as? UIImage)
            parent.dismiss()
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.onFinish(nil)
            parent.dismiss()
        }
    }
}
