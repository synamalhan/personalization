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
    @State private var showLiveCapture = false
    @State private var isWorking = false
    @State private var progress = ""
    @State private var errorMessage: String?
    @State private var summary: String?
    @State private var lastTiming: (photos: Int, embed: Double, fit: Double)?
    @State private var seenThumbs: [UIImage] = []

    private static let minPhotos = 3
    private static let maxPhotos = 20

    private var photoCount: Int { pickerItems.count + capturedPhotos.count }
    private var trimmedName: String { className.trimmingCharacters(in: .whitespacesAndNewlines) }
    private var canTeach: Bool { !trimmedName.isEmpty && photoCount >= Self.minPhotos && photoCount <= Self.maxPhotos && !isWorking && !model.isBusy }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    ScreenHeader(title: "Teach me something", subtitle: "Pick a name, show me a few photos, and I'll remember what it looks like.", color: Theme.tomato)

                    StickerCard(tint: Theme.sunflower) {
                        VStack(alignment: .leading, spacing: 10) {
                            Label("1. What is it called?", systemImage: "pencil.line").font(.headline.weight(.heavy))
                            TextField("my mug, grandma's cat, the red bike…", text: $className)
                                .textInputAutocapitalization(.never)
                                .padding(12)
                                .background(Theme.paper, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Theme.ink, lineWidth: 2))
                            if model.classNames.contains(trimmedName) {
                                Text("I already know “\(trimmedName)”, so these photos will be added to it. Names are case-sensitive.")
                                    .font(.footnote).foregroundStyle(Theme.ink.opacity(0.7))
                            }
                        }
                    }

                    StickerCard(tint: Theme.sky) {
                        VStack(alignment: .leading, spacing: 14) {
                            Label("2. Show me the photos", systemImage: "photo.stack").font(.headline.weight(.heavy))
                            PhotosPicker(selection: $pickerItems, maxSelectionCount: Self.maxPhotos - capturedPhotos.count, matching: .images) {
                                Label("Choose from my photos", systemImage: "photo.on.rectangle.angled")
                            }
                            .buttonStyle(StickerButtonStyle(color: Theme.sky))
                            Button { showLiveCapture = true } label: { Label("Teach with the live camera", systemImage: "video.fill") }
                                .buttonStyle(StickerButtonStyle(color: Theme.mint))
                                .disabled(photoCount >= Self.maxPhotos)
                            Text("Best for recognizing live: I'll learn from the same kind of frames I'll see later.")
                                .font(.caption).foregroundStyle(Theme.ink.opacity(0.65))
                            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                                Button { showCamera = true } label: { Label("Take a photo now", systemImage: "camera.fill") }
                                    .buttonStyle(StickerButtonStyle(color: Theme.bubblegum))
                                    .disabled(photoCount >= Self.maxPhotos)
                            }
                            photoDots
                            if photoCount > 0 {
                                Button("Start over with the photos", role: .destructive) { pickerItems = []; capturedPhotos = [] }
                                    .font(.footnote.weight(.semibold))
                            }
                        }
                    }

                    VStack(spacing: 14) {
                        Button { Task { await teach() } } label: {
                            HStack(spacing: 10) {
                                if isWorking { ProgressView().tint(Theme.onAccent) }
                                Text(isWorking ? "Learning…" : "Teach it!")
                            }
                        }
                        .buttonStyle(StickerButtonStyle(color: Theme.tomato))
                        .disabled(!canTeach)
                        if !canTeach && !isWorking { Text(helperText).font(.footnote).foregroundStyle(Theme.ink.opacity(0.65)) }
                        if isWorking { Text(progress).font(.footnote.weight(.semibold)).foregroundStyle(Theme.ink.opacity(0.75)) }
                    }

                    if let summary { resultCard(summary) }
                    if let errorMessage {
                        StickerCard(tint: Theme.tomato) {
                            Label { Text(errorMessage).font(.callout) } icon: { Image(systemName: "exclamationmark.bubble.fill").foregroundStyle(Theme.tomato) }
                        }
                    }

                    VStack(spacing: 14) {
                        Explainer(title: "What happens to my photos?", color: Theme.mint, icon: "lock.fill") {
                            Text("Each photo goes through a small image model that runs on this phone. It turns the picture into a list of about a thousand numbers (called an embedding) that describes what's in it.")
                            Text(deletePhotos ? "Right after that, the photo is thrown away. Only the numbers are kept, so nothing can be turned back into your picture."
                                              : "You've chosen to keep photos, so a copy is also saved inside this app's own folder. You can change this in Settings.")
                            Text("Nothing is ever uploaded. The app has no internet code at all.")
                        }
                        Explainer(title: "Why teach with the live camera?", color: Theme.mint, icon: "video.fill") {
                            Text("Photos from your library are sharp, well lit stills. Frames from the live camera are smaller, a touch blurrier, and lit differently. If I only learn from stills, the live view can look unfamiliar.")
                            Text("Teaching from live frames means I practice on exactly what I'll be shown later. Slowly turn the thing and move it closer and farther while I grab the frames, so I see it from many sides.")
                        }
                        Explainer(title: "Why 3 to 20 photos?", color: Theme.sky, icon: "number.circle.fill") {
                            Text("One photo can be a fluke: a weird angle, bad light. Three or more lets me average out the quirks and find what stays the same.")
                            Text("After about 10 photos you mostly get diminishing returns. In my tests on flowers, going from 1 to 5 photos helped a lot, while 10 to 20 changed little. Photos from different angles and lighting beat twenty near-identical shots.")
                        }
                        Explainer(title: "What is an “embedding”?", color: Theme.bubblegum, icon: "point.3.connected.trianglepath.dotted") {
                            Text("Think of it as coordinates. Similar things land close together: two photos of your mug end up near each other, and a photo of a banana ends up far away.")
                            Text("Teaching just stores those coordinates under your name. To recognize something new, I check which taught group it's closest to.")
                        }
                        Explainer(title: "What does “Teach it!” actually do?", color: Theme.grape, icon: "gearshape.2.fill") {
                            Text("It adds your new numbers to everything I already know, then re-tunes the classifier from all of them. That re-tuning is why old things stay recognized when you add new ones.")
                            Text("The timings it shows afterwards are real: how long the photos took to read and how long the re-tune took.")
                        }
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
            .background(ScreenBackground(color: Theme.tomato))
            .toolbar(.hidden, for: .navigationBar)
            .fullScreenCover(isPresented: $showLiveCapture) {
                LiveCaptureView(remaining: Self.maxPhotos - photoCount) { frames in
                    capturedPhotos.append(contentsOf: frames)
                }
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraPicker { image in
                    if let image { capturedPhotos.append(image) }
                }
                .ignoresSafeArea()
            }
        }
    }

    /// Twenty dots: the first three have a thicker outline (the minimum), and they fill in as photos are added.
    private var photoDots: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 3) {
                ForEach(0..<Self.maxPhotos, id: \.self) { i in
                    Circle()
                        .fill(i < photoCount ? Theme.tomato : Color.clear)
                        .overlay(Circle().stroke(Theme.ink.opacity(i < Self.minPhotos ? 1 : 0.3), lineWidth: i < Self.minPhotos ? 2 : 1.5))
                        .aspectRatio(1, contentMode: .fit)
                }
            }
            Text(photoCount == 0 ? "No photos yet. I need at least \(Self.minPhotos)." :
                    photoCount < Self.minPhotos ? "\(photoCount) so far. \(Self.minPhotos - photoCount) more to go." :
                    "\(photoCount) photo\(photoCount == 1 ? "" : "s") ready.")
                .font(.footnote.weight(.semibold))
                .foregroundStyle(Theme.ink.opacity(0.75))
        }
        .animation(.snappy, value: photoCount)
    }

    private var helperText: String {
        if trimmedName.isEmpty { return "Give it a name first." }
        if photoCount < Self.minPhotos { return "Add at least \(Self.minPhotos) photos." }
        if photoCount > Self.maxPhotos { return "That's too many, keep it to \(Self.maxPhotos)." }
        return ""
    }

    private func resultCard(_ text: String) -> some View {
        StickerCard(tint: Theme.mint, tilt: -1) {
            VStack(alignment: .leading, spacing: 12) {
                Label("Got it!", systemImage: "party.popper.fill").font(.title3.weight(.heavy))
                Text(text).font(.callout)
                if !seenThumbs.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("This is what I looked at").font(.caption.weight(.heavy)).opacity(0.65)
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(seenThumbs.indices, id: \.self) { i in
                                    Image(uiImage: seenThumbs[i]).resizable().scaledToFill()
                                        .frame(width: 64, height: 64)
                                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                                        .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Theme.ink, lineWidth: 2))
                                }
                            }
                            .padding(2)
                        }
                        Text("If a square shows the wrong thing (your hand, the table…), retake that photo with the subject clearer.")
                            .font(.caption2).opacity(0.6)
                    }
                }
                if let timing = lastTiming {
                    HStack(spacing: 10) {
                        StatChip(label: "photos", value: "\(timing.photos)", color: Theme.sunflower)
                        StatChip(label: "reading", value: String(format: "%.2fs", timing.embed), color: Theme.sky)
                        StatChip(label: "re-tune", value: String(format: "%.3fs", timing.fit), color: Theme.bubblegum)
                    }
                }
            }
        }
    }

    private func teach() async {
        guard canTeach else { return }
        isWorking = true
        errorMessage = nil
        summary = nil
        lastTiming = nil
        seenThumbs = []
        defer { isWorking = false }
        let name = trimmedName
        let keep = !deletePhotos
        let items = pickerItems, captured = capturedPhotos
        var vectors: [[Float]] = []
        var embedSeconds = 0.0
        var unreadable = 0

        func embedOne(_ image: UIImage, jpeg: Data?) async throws {
            progress = "Embedding photo \(vectors.count + 1) of \(items.count + captured.count)…"
            let result = try await EmbeddingService.shared.embed(image, backbone: model.backbone)
            vectors.append(result.vector)
            if let seen = result.seen { seenThumbs.append(seen) }
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
        let skipped = unreadable > 0 ? " I had to skip \(unreadable) photo\(unreadable == 1 ? "" : "s") I couldn't read." : ""
        summary = "I now know “\(name)”, learned from \(vectors.count) photos.\(skipped)"
        lastTiming = (vectors.count, embedSeconds, model.learner.lastFitSeconds)
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
