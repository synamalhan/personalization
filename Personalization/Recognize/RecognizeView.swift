import PhotosUI
import SwiftUI
import PersonalizeCore

/// What the app thought of one photo you picked, kept on screen until you move on.
struct PhotoResult {
    let image: UIImage
    let prediction: LearnerPrediction?
    let scores: [Score]
    let seconds: Double?
    var seen: UIImage? = nil
}

/// One class and how much I lean toward it (0...1).
struct Score: Identifiable {
    let name: String
    let value: Float
    var id: String { name }
}

/// Live camera recognition: about twice a second the newest frame is embedded and classified.
/// Picking a photo switches to a photo view that stays put until you choose what to do next.
struct RecognizeView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.openURL) private var openURL

    @State private var camera = CameraController()
    @State private var availability: CameraController.Availability?
    @State private var result: LearnerPrediction?
    @State private var liveScores: [Score] = []
    @State private var embedSeconds: Double?
    @State private var liveSeen: UIImage?
    /// Probabilities from the last few frames; averaged so one odd frame can't flip the answer.
    @State private var recent: [[String: Float]] = []
    private static let smoothingFrames = 4
    @State private var isAnalyzing = false
    @State private var pickedPhoto: PhotosPickerItem?
    @State private var photo: PhotoResult?
    @State private var isLoadingPhoto = false
    @State private var showHelp = false
    @AppStorage(AppSettings.focusSubjectKey) private var focusSubject = true

    var body: some View {
        NavigationStack {
            Group {
                if let photo {
                    photoView(photo)
                } else {
                    liveView
                }
            }
            .navigationTitle("Recognize")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showHelp = true } label: { Image(systemName: "questionmark.circle.fill") }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    PhotosPicker(selection: $pickedPhoto, matching: .images) { Image(systemName: "photo.fill") }
                }
            }
            .sheet(isPresented: $showHelp) { helpSheet }
            .onChange(of: model.classNames) { _, _ in recent = [] }
            .onChange(of: model.backbone) { _, _ in recent = []; result = nil; liveScores = [] }
            .onChange(of: pickedPhoto) { _, item in
                guard let item else { return }
                Task { await usePhoto(item) }
            }
        }
        .task(id: scenePhase) {
            // Runs while this view exists; restarts when the app goes to the background and back.
            guard scenePhase == .active else { camera.stop(); return }
            availability = await camera.prepare()
            if availability == .ready && photo == nil { camera.start() }
            while !Task.isCancelled && availability == .ready {
                try? await Task.sleep(for: .milliseconds(500))
                // Paused while a photo is on screen. A frame is skipped (not queued) if the last one is still running.
                if photo == nil, let frame = camera.latestFrame() {
                    await analyze { try await EmbeddingService.shared.embed(frame: frame, backbone: model.backbone) } update: { r, probs in
                        recent.append(probs)
                        if recent.count > Self.smoothingFrames { recent.removeFirst() }
                        let averaged = Self.average(recent)
                        withAnimation(.snappy) { result = decide(averaged); liveScores = topScores(averaged) }
                        embedSeconds = r.seconds
                        liveSeen = r.seen
                    }
                }
            }
        }
        .onDisappear { camera.stop() }
    }

    // MARK: - Live camera

    private var liveView: some View {
        ZStack(alignment: .bottom) {
            if availability == .ready {
                CameraPreview(session: camera.session).ignoresSafeArea()
            } else {
                ScreenBackground(color: Theme.sky)
                unavailableView
            }
            ScrollView {
                resultCard(prediction: result, scores: liveScores, seconds: embedSeconds, seen: liveSeen)
                    .padding(.horizontal, 20)
            }
            .scrollBounceBehavior(.basedOnSize)
            .frame(maxHeight: 340)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.bottom, 8)
        }
    }

    // MARK: - Photo mode

    private func photoView(_ photo: PhotoResult) -> some View {
        ScrollView {
            VStack(spacing: 18) {
                Image(uiImage: photo.image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 320)
                    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous).stroke(Theme.ink, lineWidth: 2.5))
                    .background(RoundedRectangle(cornerRadius: 20, style: .continuous).fill(Theme.sky).offset(x: 5, y: 6))
                    .padding(.horizontal, 6)
                    .padding(.top, 8)

                resultCard(prediction: photo.prediction, scores: photo.scores, seconds: photo.seconds, seen: photo.seen)

                Text("Want me to look at something else?")
                    .font(.system(.callout, design: .serif)).italic().foregroundStyle(Theme.ink.opacity(0.7))
                    .padding(.top, 4)

                PhotosPicker(selection: $pickedPhoto, matching: .images) {
                    Label("Try another photo", systemImage: "photo.on.rectangle.angled")
                }
                .buttonStyle(StickerButtonStyle(color: Theme.sky))

                Button {
                    withAnimation(.snappy) { self.photo = nil; result = nil; liveScores = []; recent = [] }
                    if availability == .ready { camera.start() }
                } label: {
                    Label(availability == .ready ? "Back to the camera" : "Done", systemImage: availability == .ready ? "camera.fill" : "checkmark")
                }
                .buttonStyle(StickerButtonStyle(color: Theme.bubblegum))
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .background(ScreenBackground(color: Theme.sky))
    }

    /// Embeds the picked photo and shows it with the answer. It stays until you pick another or go back.
    private func usePhoto(_ item: PhotosPickerItem) async {
        defer { pickedPhoto = nil }
        guard let data = try? await item.loadTransferable(type: Data.self),
              let image = ImageLoading.image(from: data) else { return }
        camera.stop()
        isLoadingPhoto = true
        defer { isLoadingPhoto = false }
        if model.classNames.isEmpty {
            withAnimation(.snappy) { photo = PhotoResult(image: image, prediction: nil, scores: [], seconds: nil) }
            return
        }
        await analyze { try await EmbeddingService.shared.embed(image, backbone: model.backbone) } update: { r, probs in
            withAnimation(.snappy) { photo = PhotoResult(image: image, prediction: decide(probs), scores: topScores(probs), seconds: r.seconds, seen: r.seen) }
        }
        if photo == nil {   // embedding failed: still show the photo, with a note
            photo = PhotoResult(image: image, prediction: nil, scores: [], seconds: nil)
        }
    }

    private func analyze(_ embed: () async throws -> EmbeddingResult,
                         update: (EmbeddingResult, [String: Float]) -> Void) async {
        guard !isAnalyzing, !model.classNames.isEmpty else { return }
        isAnalyzing = true
        defer { isAnalyzing = false }
        guard let r = try? await embed() else { return }
        update(r, model.learner.probabilities(for: r.vector))
    }

    /// Same rule as `Learner.predict`, applied to (possibly averaged) probabilities.
    private func decide(_ probs: [String: Float]) -> LearnerPrediction {
        let ranked = probs.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value }
        guard let best = ranked.first else { return .unknown(bestGuess: nil) }
        if probs.count < 2 || best.value < model.learner.threshold { return .unknown(bestGuess: best.key) }
        return .known(className: best.key, confidence: best.value)
    }

    private func topScores(_ probs: [String: Float]) -> [Score] {
        probs.map { Score(name: $0.key, value: $0.value) }.sorted { $0.value > $1.value }.prefix(3).map { $0 }
    }

    /// Mean probability per class over the recent frames.
    private static func average(_ frames: [[String: Float]]) -> [String: Float] {
        var sum: [String: Float] = [:]
        for frame in frames { for (name, p) in frame { sum[name, default: 0] += p } }
        return sum.mapValues { $0 / Float(frames.count) }
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
        .padding(.bottom, 200)
    }

    private func resultCard(prediction: LearnerPrediction?, scores: [Score], seconds: Double?, seen: UIImage?) -> some View {
        StickerCard(tint: cardColor(prediction)) {
            VStack(alignment: .leading, spacing: 10) {
                if isLoadingPhoto {
                    HStack { ProgressView(); Text("Looking at your photo…").font(.headline) }
                } else if model.classNames.isEmpty {
                    Label("Nothing taught yet", systemImage: "graduationcap.fill").font(.title3.weight(.heavy))
                    Text("Head to the Teach tab and show me something first.").font(.callout)
                } else if model.classNames.count < 2 {
                    Label("Teach me one more", systemImage: "plus.circle.fill").font(.title3.weight(.heavy))
                    Text("With only one thing taught, I'd say it every time and have no way to know if I'm wrong. Add a second thing to compare against.").font(.callout)
                } else {
                    switch prediction {
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
                        Label(photo == nil ? "Point me at something" : "I couldn't read that photo",
                              systemImage: photo == nil ? "viewfinder" : "exclamationmark.triangle.fill").font(.title3.weight(.heavy))
                    }
                    if scores.count > 1 { scoreRows(scores) }
                    if model.needsReteaching(currentFocus: focusSubject) {
                        Text("Heads up: some classes were taught with different settings. Re-teach them for better answers.")
                            .font(.caption.weight(.semibold)).foregroundStyle(Theme.tomato)
                    }
                    HStack(spacing: 10) {
                        if let seen {
                            Image(uiImage: seen).resizable().scaledToFill()
                                .frame(width: 52, height: 52)
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).stroke(Theme.ink, lineWidth: 2))
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            if seen != nil { Text("What I looked at").font(.caption.weight(.heavy)).opacity(0.65) }
                            if let seconds {
                                Text(String(format: "%.0f ms", seconds * 1000)).font(.caption2.weight(.semibold)).opacity(0.6)
                            }
                        }
                    }
                }
            }
            .foregroundStyle(Theme.ink)
        }
    }

    /// How close the runner-ups were. When two bars are nearly equal, I was really torn between them.
    private func scoreRows(_ scores: [Score]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("How torn was I?").font(.caption.weight(.heavy)).opacity(0.65)
            ForEach(scores) { s in
                HStack(spacing: 8) {
                    Text(s.name).font(.caption.weight(.semibold)).lineLimit(1).frame(width: 90, alignment: .leading)
                    GeometryReader { geo in
                        Capsule().fill(Theme.ink.opacity(0.08))
                            .overlay(alignment: .leading) {
                                Capsule().fill(ClassAvatar.color(for: s.name)).frame(width: max(6, geo.size.width * CGFloat(s.value)))
                            }
                            .overlay(Capsule().stroke(Theme.ink, lineWidth: 1.2))
                    }
                    .frame(height: 10)
                    Text(String(format: "%.0f%%", s.value * 100)).font(.caption.weight(.bold)).frame(width: 38, alignment: .trailing)
                }
            }
        }
    }

    private func cardColor(_ prediction: LearnerPrediction?) -> Color {
        if case .known(let name, _) = prediction, model.classNames.count >= 2 { return ClassAvatar.color(for: name) }
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
                Explainer(title: "Why can't I tell two things apart?", color: Theme.tomato, icon: "square.on.square") {
                    Text("I look at the whole picture, not just the object. If two things were photographed in the same place, with the same background and light, their numbers end up very close, and the “How torn was I?” bars show it.")
                    Text("The “Ignore the background” switch in Settings makes me cut the subject out first, which helps a lot with this. Also teach each thing with photos where it's the clear subject, on different backgrounds and from different angles.")
                }
                Explainer(title: "When do I say “don't know”?", color: Theme.grape, icon: "questionmark.bubble.fill") {
                    Text("When my best guess falls below your unknown threshold (Settings → Unknown threshold). Raise it and I'll admit uncertainty more often; lower it and I'll guess more.")
                    Text("Something you've never taught me can still look similar to something you have, so a good threshold depends on what you teach.")
                }
                Explainer(title: "Why does it need two things taught?", color: Theme.mint, icon: "2.circle.fill") {
                    Text("Recognizing is choosing between options. With one option, I'd always pick it with 100% “confidence”, which would be meaningless, so I wait until there's something to compare against.")
                }
                Explainer(title: "What gets looked at, and where does it go?", color: Theme.sunflower, icon: "lock.shield.fill") {
                    Text("About twice a second I grab the newest camera frame, shrink and crop it to a small square, and turn it into numbers on this phone. If I'm still busy with the last frame, the new one is dropped rather than queued.")
                    Text("Frames and photos are never saved or sent anywhere.")
                }
                Explainer(title: "Trying a photo", color: Theme.bubblegum, icon: "photo.fill") {
                    Text("The picture button in the top corner lets you test me on any photo. The photo stays on screen with my answer until you pick another one or go back to the camera.")
                }
            }
            .padding(20)
        }
        .background(ScreenBackground(color: Theme.sky))
        .presentationDetents([.large])
    }
}
