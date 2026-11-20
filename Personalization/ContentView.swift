import SwiftUI

/// The tab bar. Each tab lives in its own folder.
struct ContentView: View {
    var body: some View {
        TabView {
            Tab("Teach", systemImage: "plus.circle") { TeachView() }
            Tab("Recognize", systemImage: "camera.viewfinder") { RecognizeView() }
            Tab("Library", systemImage: "books.vertical") { LibraryView() }
            Tab("Settings", systemImage: "gearshape") { SettingsView() }
        }
    }
}

#Preview {
    ContentView().environment(AppModel(storeURL: FileManager.default.temporaryDirectory.appendingPathComponent("preview.json")))
}
