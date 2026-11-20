import SwiftUI
import UIKit

/// Grabs a burst of frames from the live camera so a class can be taught from exactly what Recognize will see.
struct LiveCaptureView: View {
    /// How many more photos Teach can accept (the limit is 20 in total).
    let remaining: Int
    let onFinish: ([UIImage]) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var camera = CameraController()
    @State private var availability: CameraController.Availability?
    @State private var captured: [UIImage] = []
    @State private var isCapturing = false

    private var goal: Int { min(10, remaining) }

    var body: some View {
        ZStack {
            if availability == .ready {
                CameraPreview(session: camera.session).ignoresSafeArea()
            } else {
                ScreenBackground(color: Theme.mint)
                VStack(spacing: 12) {
                    Text(availability == .denied ? "🙈" : availability == .unavailable ? "📷" : "⏳").font(.system(size: 56))
                    Text(availability == .denied ? "Camera access is off. Turn it on in Settings."
                         : availability == .unavailable ? "No camera available here."
                         : "Warming up the camera…")
                        .multilineTextAlignment(.center).font(.headline).foregroundStyle(Theme.ink)
                }
                .padding(30)
            }

            VStack {
                StickerCard(tint: Theme.mint) {
                    Text(isCapturing ? "Keep going: slowly turn it, move closer and farther…"
                                     : "Hold the thing in view. When you tap, I'll grab \(goal) frames over a few seconds, so move it around a little: different angles and distances.")
                        .font(.footnote.weight(.semibold)).foregroundStyle(Theme.ink)
                }
                .padding(.horizontal, 20).padding(.top, 8)

                Spacer()

                VStack(spacing: 12) {
                    if isCapturing {
                        HStack(spacing: 4) {
                            ForEach(0..<goal, id: \.self) { i in
                                Circle().fill(i < captured.count ? Theme.mint : Color.clear)
                                    .overlay(Circle().stroke(Theme.ink, lineWidth: 2))
                                    .aspectRatio(1, contentMode: .fit)
                            }
                        }
                        .padding(.horizontal, 30)
                    }
                    Button { Task { await capture() } } label: {
                        Label(isCapturing ? "Capturing \(captured.count) of \(goal)…" : "Capture \(goal) frames", systemImage: "camera.aperture")
                    }
                    .buttonStyle(StickerButtonStyle(color: Theme.mint))
                    .disabled(availability != .ready || isCapturing || goal == 0)
                    .padding(.horizontal, 24)

                    Button("Cancel") { dismiss() }
                        .font(.subheadline.weight(.bold))
                        .padding(.vertical, 6).padding(.horizontal, 16)
                        .background(Theme.card, in: Capsule())
                        .overlay(Capsule().stroke(Theme.ink, lineWidth: 2))
                        .foregroundStyle(Theme.ink)
                        .disabled(isCapturing)
                }
                .padding(.bottom, 24)
            }
        }
        .task {
            availability = await camera.prepare()
            if availability == .ready { camera.start() }
        }
        .onDisappear { camera.stop() }
    }

    private func capture() async {
        isCapturing = true
        captured = []
        let haptic = UIImpactFeedbackGenerator(style: .light)
        for _ in 0..<goal {
            try? await Task.sleep(for: .milliseconds(450))
            if let image = camera.snapshot() {
                captured.append(image)
                haptic.impactOccurred()
            }
        }
        onFinish(captured)
        dismiss()
    }
}
