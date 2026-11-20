import SwiftUI

/// Tools for the experiments: not part of everyday use, so they live behind Settings.
struct LabView: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                ScreenHeader(title: "The Lab", subtitle: "Where the app gets measured. Skip it unless you're curious.", color: Theme.grape)
                labLink("Gate check", "Does the phone see photos the same way my laptop does?", "checkmark.seal.fill", Theme.mint) { GateCheckView() }
                labLink("Benchmark", "How fast is it, and how much memory does it use?", "stopwatch.fill", Theme.sunflower) { BenchmarkView() }
                labLink("Real-photo test", "Store vectors from your own photos, tagged by day, to test on the laptop.", "calendar.badge.checkmark", Theme.sky) { RealPhotoView() }
                Explainer(title: "Why does this exist?", color: Theme.grape, icon: "flask.fill") {
                    Text("This app is part of a small study: how many photos does a new thing need, does adding things make old ones worse, and can it say “I don't know”. The offline numbers come from a laptop. These tools check that the phone agrees with the laptop and measure what it costs on a real device.")
                    Text("Nothing here sends data anywhere. Files are only written when you tap Share.")
                }
            }
            .padding(20)
        }
        .background(ScreenBackground(color: Theme.grape))
        .navigationTitle("Lab")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func labLink<D: View>(_ title: String, _ detail: String, _ icon: String, _ color: Color, @ViewBuilder destination: () -> D) -> some View {
        NavigationLink(destination: destination) {
            StickerCard(tint: color) {
                HStack(spacing: 12) {
                    Image(systemName: icon).font(.title2).foregroundStyle(color).shadow(color: Theme.ink, radius: 0, x: 1, y: 1)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(title).font(.headline.weight(.heavy))
                        Text(detail).font(.caption).opacity(0.7).multilineTextAlignment(.leading)
                    }
                    Spacer()
                    Image(systemName: "chevron.right").font(.footnote.weight(.bold))
                }
                .foregroundStyle(Theme.ink)
            }
        }
        .buttonStyle(.plain)
    }
}

/// Small helpers shared by the Lab screens.
nonisolated enum LabFiles {
    /// Writes `data` into Documents (easy to find from a Mac) and returns that URL for sharing.
    static func write(_ data: Data, name: String) throws -> URL {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(name)
        try data.write(to: url, options: .atomic)
        return url
    }

    static var deviceDescription: String {
        var info = utsname()
        uname(&info)
        let model = withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        return model
    }

    static var buildConfiguration: String {
        #if DEBUG
        "debug"
        #else
        "release"
        #endif
    }
}
