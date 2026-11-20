import PhotosUI
import SwiftUI
import PersonalizeCore

/// Live camera recognition: about twice a second the newest frame is embedded and classified.
struct RecognizeView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.openURL) private var openURL

    @State private var camera = CameraController()
    @State private var availability: CameraController.Availability?
    @State private var result: LearnerPrediction?
    @State private var embedSeconds: Double?
    @State private var isAnalyzing = false
    @State private var pickedPhoto: PhotosPickerItem?
    @State private var showHelp = false

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                if availability == .ready {
                    CameraPreview(session: camera.session).ignoresSafeArea()
                } else {
                    ScreenBackground(color: Theme.sky)
                    unavailableView
                }
                resultCard.padding(.horizontal, 20).padding(.bottom, 12)
            }
            .navigationTitle("Recognize")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showHelp = true } label: { Image(systemName: "questionmark.circle.fill") }
                }
                // Lets you test without a camera (the simulator has none).
                ToolbarItem(placement: .topBarTrailing) {
                    PhotosPicker(selection: $pickedPhoto, matching: .images) { Image(systemName: "photo.fill") }
                }
            }
            .sheet(isPresented: $showHelp) { helpSheet }
            .onChange(of: pickedPhoto) { _, item in
                guard let item else { return }
                Task {
                    if let data = try? await item.loadTransferable(type: Data.self), let image = ImageLoading.image(from: data) {
                        await analyze { try await EmbeddingService.shared.embed(image) }
                    }
                    pickedPhoto = nil
                }
            }
        }
        .task(id: scenePhase) {
            // Runs while this view exists; restarts when the app goes to the background and back.
            guard scenePhase == .active else { camera.stop(); return }
            availability = await camera.prepare()
            if availability == .ready { camera.start() }
            while !Task.isCancelled && availability == .ready {
                try? await Task.sleep(for: .milliseconds(500))
                // A frame is skipped (not queued) if the previous one is still being processed.
                if let frame = camera.latestFrame() {
                    await analyze { try await EmbeddingService.shared.embed(frame: frame) }
                }
            }
        }
        .onDisappear { camera.stop() }
    }

    private func analyze(_ embed: () async throws -> EmbeddingResult) async {
        guard !isAnalyzing, !model.classNames.isEmpty else { return }
        isAnalyzing = true
        defer { isAnalyzing = false }
        if let r = try? await embed() {
            withAnimation(.snappy) { result = model.predict(r.vector) }
            embedSeconds = r.seconds
        }
    }

    // MARK: - Pieces

    @ViewBuilder private var unavailableView: some View {
        VStack(spacing: 14) {
            Text(availability == .denied ? "🙈" : availability == .unavailable ? "📷" : "⏳").font(.system(size: 64))
            switch availability {
            case .denied:
                Text("I can't see anything").font(.title2.weight(.black))
                Text("Camera access is switched off. Turn it on in Settings and I'll start looking. Nothing you see is ever saved or sent anywhere.")
                    .multilineTextAlignment(.center).foregroundStyle(Theme.ink.opacity(0.75))
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                }.buttonStyle(StickerButtonStyle(color: Theme.sky)).frame(maxWidth: 260)
            case .unavailable:
                Text("No camera here").font(.title2.weight(.black))
                Text("This device has no camera I can use. Tap the photo button at the top to try me on a picture instead.")
                    .multilineTextAlignment(.center).foregroundStyle(Theme.ink.opacity(0.75))
            default:
                Text("Warming up the camera…").font(.title3.weight(.bold))
                ProgressView()
            }
        }
        .foregroundStyle(Theme.ink)
        .padding(28)
        .padding(.bottom, 120)
    }

    @ViewBuilder private var resultCard: some View {
        StickerCard(tint: cardColor) {
            VStack(alignment: .leading, spacing: 10) {
                if model.classNames.isEmpty {
                    Label("Nothing taught yet", systemImage: "graduationcap.fill").font(.title3.weight(.heavy))
                    Text("Head to the Teach tab and show me something first.").font(.callout)
                } else if model.classNames.count < 2 {
                    Label("Teach me one more", systemImage: "plus.circle.fill").font(.title3.weight(.heavy))
                    Text("With only one thing taught, I'd say it every time and have no way to know if I'm wrong. Add a second thing to compare against.").font(.callout)
                } else {
                    switch result {
                    case .known(let name, let confidence):
                        HStack(spacing: 12) {
                            ClassAvatar(name: name, size: 52)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(name).font(.title.weight(.black)).lineLimit(2).minimumScaleFactor(0.6)
                                Text(String(format: "%.0f%% sure", confidence * 100)).font(.subheadline.weight(.bold))
                            }
                        }
                        ConfidenceMeter(value: confidence, color: ClassAvatar.color(for: name))
                    case .unknown(let guess):
                        HStack(spacing: 12) {
                            Text("🤔").font(.system(size: 44))
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Hmm, don't know that one").font(.title3.weight(.black))
                                if let guess { Text("Closest is “\(guess)”, but I'm not sure enough.").font(.footnote) }
                            }
                        }
                    case nil:
                        Label("Point me at something", systemImage: "viewfinder").font(.title3.weight(.heavy))
                    }
                    if let embedSeconds {
                        Text(String(format: "Looked at it in %.0f ms", embedSeconds * 1000))
                            .font(.caption2.weight(.semibold)).opacity(0.6)
                    }
                }
            }
            .foregroundStyle(Theme.ink)
        }
    }

    private var cardColor: Color {
        if case .known(let name, _) = result, model.classNames.count >= 2 { return ClassAvatar.color(for: name) }
        return Theme.sky
    }

    private var helpSheet: some View {
        ScrollView {
            VStack(spacing: 16) {
                ScreenHeader(title: "How I decide", subtitle: "A few things worth knowing before you trust me.", color: Theme.sky)
                Explainer(title: "What do the percentages mean?", color: Theme.sky, icon: "percent") {
                    Text("It's how strongly I favor my top answer over the other things you taught me. 90% means I'm leaning very hard one way; 40% means I'm torn between a few.")
                    Text("It's not a guarantee. A confident answer can still be wrong, which is why there's an “I don't know” option.")
                }
                Explainer(title: "When do I say “don't know”?", color: Theme.tomato, icon: "questionmark.bubble.fill") {
                    Text("When my best guess falls below your unknown threshold (Settings → Unknown threshold). Raise it and I'll admit uncertainty more often; lower it and I'll guess more.")
                    Text("Something you've never taught me can still look similar to something you have, so a good threshold depends on what you teach.")
                }
                Explainer(title: "Why does it need two things taught?", color: Theme.mint, icon: "2.circle.fill") {
                    Text("Recognizing is choosing between options. With one option, I'd always pick it with 100% “confidence”, which would be meaningless, so I wait until there's something to compare against.")
                }
                Explainer(title: "What gets looked at, and where does it go?", color: Theme.grape, icon: "lock.shield.fill") {
                    Text("About twice a second I grab the newest camera frame, shrink and crop it to a small square, and turn it into numbers on this phone. If I'm still busy with the last frame, the new one is dropped rather than queued.")
                    Text("Frames are never saved or sent anywhere.")
                }
                Explainer(title: "No camera? Try a photo", color: Theme.bubblegum, icon: "photo.fill") {
                    Text("The picture button in the top corner lets you test me on any photo from your library. That's the way to try the app in the simulator.")
                }
            }
            .padding(20)
        }
        .background(ScreenBackground(color: Theme.sky))
        .presentationDetents([.large])
    }
}
