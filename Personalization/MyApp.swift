import SwiftUI
import PersonalizeCore

@main struct MyApp: App {
    // `@State` keeps one AppModel alive for the whole app; `.environment` hands it to every view below.
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(model)
        }
    }
}
