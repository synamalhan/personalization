import SwiftUI

/// The tab bar. Every tab has its own color, and the bar takes on the color of the tab you are in.
struct ContentView: View {
    enum Page: Hashable { case teach, recognize, library, settings }
    @State private var page = Page.teach

    private var tint: Color {
        switch page {
        case .teach: Theme.tomato
        case .recognize: Theme.sky
        case .library: Theme.mint
        case .settings: Theme.grape
        }
    }

    var body: some View {
        TabView(selection: $page) {
            Tab("Teach", systemImage: "wand.and.stars", value: Page.teach) { TeachView() }
            Tab("Recognize", systemImage: "eye.circle.fill", value: Page.recognize) { RecognizeView() }
            Tab("Library", systemImage: "square.stack.3d.up.fill", value: Page.library) { LibraryView() }
            Tab("Settings", systemImage: "slider.horizontal.3", value: Page.settings) { SettingsView() }
        }
        .tint(tint)
        .fontDesign(.rounded)
        .animation(.snappy, value: page)
    }
}

#Preview {
    ContentView().environment(AppModel(storeURL: FileManager.default.temporaryDirectory.appendingPathComponent("preview.json")))
}
