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
    @State private var isVisible = false
    @State private var pickedPhoto: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                if availability == .ready {
                    CameraPreview(session: camera.session).ignoresSafeArea()
                } else {
                    unavailableView
                }
                resultCard.padding()
            }
            .navigationTitle("Recognize")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                // Lets you test without a camera (the simulator has none).
                ToolbarItem(placement: .topBarTrailing) {
                    PhotosPicker(selection: $pickedPhoto, matching: .images) { Image(systemName: "photo") }
                }
            }
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
            isVisible = true
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
        .onDisappear { camera.stop(); isVisible = false }
    }

    private func analyze(_ embed: () async throws -> EmbeddingResult) async {
        guard !isAnalyzing, !model.classNames.isEmpty else { return }
        isAnalyzing = true
        defer { isAnalyzing = false }
        if let r = try? await embed() {
            result = model.predict(r.vector)
            embedSeconds = r.seconds
        }
    }

    @ViewBuilder private var unavailableView: some View {
        VStack(spacing: 12) {
            Image(systemName: "camera.slash").font(.largeTitle)
            switch availability {
            case .denied:
                Text("Camera access is off.").font(.headline)
                Text("Turn it on in Settings to recognize things live. Photos never leave your device.")
                    .multilineTextAlignment(.center).foregroundStyle(.secondary)
                Button("Open Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) { openURL(url) }
                }.buttonStyle(.borderedProminent)
            case .unavailable:
                Text("No camera available.").font(.headline)
                Text("Use the photo button at the top to test with a picture instead.").foregroundStyle(.secondary)
            default:
                ProgressView()
            }
        }.padding()
    }

    @ViewBuilder private var resultCard: some View {
        VStack(spacing: 4) {
            if model.classNames.isEmpty {
                Text("Nothing taught yet").font(.headline)
                Text("Go to the Teach tab and add a class first.").font(.footnote)
            } else if model.classNames.count < 2 {
                Text("Teach at least two classes").font(.headline)
                Text("With one class the app cannot tell if it is right.").font(.footnote)
            } else {
                switch result {
                case .known(let name, let confidence):
                    Text(name).font(.title.bold())
                    Text(String(format: "%.0f%% confident", confidence * 100)).font(.subheadline)
                case .unknown:
                    Text("Unknown").font(.title.bold())
                    Text("Too far from anything you taught.").font(.footnote)
                case nil:
                    Text("Point the camera at something").font(.headline)
                }
                if let embedSeconds {
                    Text(String(format: "%.0f ms per frame", embedSeconds * 1000)).font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}
