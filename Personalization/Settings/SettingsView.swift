import SwiftUI
import PersonalizeCore

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    // `@AppStorage` reads and writes UserDefaults and refreshes the view when the value changes.
    @AppStorage(AppSettings.thresholdKey) private var threshold = AppSettings.defaultThreshold
    @AppStorage(AppSettings.headKey) private var head = AppSettings.defaultHead
    @AppStorage(AppSettings.backboneKey) private var backbone = AppSettings.defaultBackbone
    @AppStorage(AppSettings.deletePhotosKey) private var deletePhotos = true

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading) {
                        Slider(value: $threshold, in: 0...1, step: 0.01)
                        Text(String(format: "Unknown below %.0f%% confidence", threshold * 100))
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                } header: { Text("Unknown threshold") } footer: {
                    Text("A prediction less confident than this is shown as “Unknown”.")
                }
                Section("Classifier") {
                    Picker("Head", selection: $head) {
                        Text("Nearest class mean").tag(HeadKind.nearestClassMean.rawValue)
                        Text("Logistic regression").tag(HeadKind.logistic.rawValue)
                    }
                    Picker("Backbone", selection: $backbone) {
                        ForEach(Backbone.allCases) { b in
                            Text(b.displayName + (b.isInstalled ? "" : " (not installed)")).tag(b.rawValue)
                        }
                    }
                    .disabled(Backbone.installed.count < 2)
                }
                Section {
                    Toggle("Delete photos right after embedding", isOn: $deletePhotos)
                } footer: {
                    Text("On (recommended): only the numbers are kept. Off: photos are also saved in the app's Documents folder.")
                }
                Section("Privacy") {
                    Label("Everything happens on this device.", systemImage: "lock.shield")
                    Text("The app stores class names and embedding vectors (lists of numbers). It has no network code, no analytics, and nothing leaves your device.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Section("Tools") {
                    NavigationLink("Lab (benchmark, checks, real-photo test)") { LabView() }
                }
            }
            .navigationTitle("Settings")
            .onChange(of: threshold) { _, _ in Task { await model.applySettings() } }
            .onChange(of: head) { _, _ in Task { await model.applySettings() } }
        }
    }
}
