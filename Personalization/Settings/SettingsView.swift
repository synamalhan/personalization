import SwiftUI
import PersonalizeCore

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    // `@AppStorage` reads and writes UserDefaults and refreshes the view when the value changes.
    @AppStorage(AppSettings.thresholdKey) private var threshold = AppSettings.defaultThreshold
    @AppStorage(AppSettings.headKey) private var head = AppSettings.defaultHead
    @AppStorage(AppSettings.backboneKey) private var backbone = AppSettings.defaultBackbone
    @AppStorage(AppSettings.deletePhotosKey) private var deletePhotos = true

    private var strictness: (String, Color) {
        switch threshold {
        case ..<0.3: ("Easygoing: I'll take a guess at almost anything", Theme.mint)
        case ..<0.6: ("Balanced: I guess when I'm fairly sure", Theme.sunflower)
        case ..<0.85: ("Careful: I want to be quite sure", Theme.bubblegum)
        default: ("Very strict: I'll say “don't know” a lot", Theme.tomato)
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 22) {
                    ScreenHeader(title: "Tweak me", subtitle: "Everything here is optional. The defaults work fine.", color: Theme.grape)

                    // Unknown threshold
                    StickerCard(tint: Theme.tomato) {
                        VStack(alignment: .leading, spacing: 12) {
                            Label("How sure before I answer?", systemImage: "hand.raised.fill").font(.headline.weight(.heavy))
                            Slider(value: $threshold, in: 0...1, step: 0.01).tint(strictness.1)
                            HStack(spacing: 8) {
                                Text(String(format: "%.0f%%", threshold * 100))
                                    .font(.system(.title3, design: .rounded).weight(.black))
                                    .foregroundStyle(Theme.onAccent)
                                    .padding(.horizontal, 10).padding(.vertical, 4)
                                    .background(strictness.1, in: Capsule())
                                    .overlay(Capsule().stroke(Theme.ink, lineWidth: 2))
                                Text(strictness.0).font(.footnote.weight(.semibold)).foregroundStyle(Theme.ink.opacity(0.8))
                            }
                        }
                    }
                    Explainer(title: "Unknown threshold, in detail", color: Theme.tomato, icon: "questionmark.diamond.fill") {
                        Text("Every time I look at something, I give each thing you taught me a probability. If my best one is below this line, I say “don't know” instead of guessing.")
                        Text("Too low and I'll confidently mislabel things you never taught me. Too high and I'll refuse even things I've seen. In my experiments, detecting unfamiliar things worked much better with one of the two image models than the other, so it's worth trying the camera on a few things you haven't taught.")
                    }

                    // Head choice
                    StickerCard(tint: Theme.sky) {
                        VStack(alignment: .leading, spacing: 12) {
                            Label("How I tell things apart", systemImage: "brain.head.profile").font(.headline.weight(.heavy))
                            ChoiceCard(title: "Nearest average", detail: "Compares to the average of each thing", icon: "scope", color: Theme.sky,
                                       isSelected: head == HeadKind.nearestClassMean.rawValue) { head = HeadKind.nearestClassMean.rawValue }
                            ChoiceCard(title: "Logistic regression", detail: "Learns a boundary between things", icon: "chart.line.uptrend.xyaxis", color: Theme.grape,
                                       isSelected: head == HeadKind.logistic.rawValue) { head = HeadKind.logistic.rawValue }
                        }
                    }
                    Explainer(title: "Which one should I pick?", color: Theme.sky, icon: "arrow.triangle.branch") {
                        Text("Nearest average: for each thing I keep the average of its photos, then answer with whichever average your new photo is closest to. It's instant and needs no training.")
                        Text("Logistic regression: I run a few hundred small adjustment steps over all your photos to learn where the boundaries between things are. It takes a moment longer when you teach, and in my flower tests it came out slightly ahead with the lighter image model.")
                        Text("With the bigger image model the two were practically tied, so Nearest average is a fine default.")
                    }

                    // Backbone
                    StickerCard(tint: Theme.bubblegum) {
                        VStack(alignment: .leading, spacing: 10) {
                            Label("Which eyes to use", systemImage: "eye.fill").font(.headline.weight(.heavy))
                            Picker("Image model", selection: $backbone) {
                                ForEach(Backbone.allCases) { b in
                                    Text(b.displayName + (b.isInstalled ? "" : " (not installed)")).tag(b.rawValue)
                                }
                            }
                            .pickerStyle(.menu)
                            .disabled(Backbone.installed.count < 2)
                            if Backbone.installed.count < 2 {
                                Text("Only one image model is installed in this build, so there's nothing to switch yet.")
                                    .font(.footnote).foregroundStyle(Theme.ink.opacity(0.7))
                            }
                        }
                    }
                    Explainer(title: "What's the image model?", color: Theme.bubblegum, icon: "camera.aperture") {
                        Text("It's the part that turns a photo into numbers. I ship with MobileNetV3, a small, fast one made for phones. DINOv2-small is a bigger alternative that did better at spotting unfamiliar things in my tests, but isn't bundled in this build.")
                        Text("Switching models changes the numbers completely, so anything you've taught with one can't be used with the other. You'd need to start fresh.")
                    }

                    // Photos
                    StickerCard(tint: Theme.mint) {
                        VStack(alignment: .leading, spacing: 8) {
                            Toggle(isOn: $deletePhotos) {
                                Label("Throw photos away after reading them", systemImage: "trash.slash.fill").font(.headline.weight(.heavy))
                            }
                            .tint(Theme.mint)
                            Text(deletePhotos ? "Only the numbers are kept." : "Photos are also saved in this app's private folder.")
                                .font(.footnote).foregroundStyle(Theme.ink.opacity(0.7))
                        }
                    }

                    // Privacy
                    StickerCard(tint: Theme.mint, tilt: -0.8) {
                        VStack(alignment: .leading, spacing: 8) {
                            Label("Private by design", systemImage: "lock.shield.fill").font(.headline.weight(.heavy))
                            Text("Everything happens on this phone. Nothing leaves it.").font(.callout)
                        }
                    }
                    Explainer(title: "Exactly what's stored, and what isn't", color: Theme.mint, icon: "checkmark.shield.fill") {
                        Text("Stored on your phone: the names you give things, and a list of numbers for each photo. Optionally, the photos themselves if you turn that on above.")
                        Text("Never sent anywhere: the app has no networking code, no analytics, no accounts, and no cloud models. The only permission it asks for is the camera.")
                        Text("The Lab tools below can write files you choose to share, but only when you tap Share.")
                    }

                    // Lab
                    NavigationLink { LabView() } label: {
                        StickerCard(tint: Theme.grape) {
                            HStack(spacing: 12) {
                                Image(systemName: "flask.fill").font(.title2).foregroundStyle(Theme.grape)
                                    .shadow(color: Theme.ink, radius: 0, x: 1, y: 1)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("The Lab").font(.headline.weight(.heavy))
                                    Text("Benchmarks and checks for the curious").font(.caption).opacity(0.7)
                                }
                                Spacer()
                                Image(systemName: "chevron.right").font(.footnote.weight(.bold))
                            }
                            .foregroundStyle(Theme.ink)
                        }
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
            .background(ScreenBackground(color: Theme.grape))
            .toolbar(.hidden, for: .navigationBar)
            .onChange(of: threshold) { _, _ in Task { await model.applySettings() } }
            .onChange(of: head) { _, _ in Task { await model.applySettings() } }
        }
    }
}
