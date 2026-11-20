import SwiftUI

/// Tools for the experiments: not part of everyday use, so they live behind Settings.
struct LabView: View {
    var body: some View {
        List {
            NavigationLink("Gate check (phone vs laptop)") { GateCheckView() }
            NavigationLink("Benchmark (experiment 7)") { BenchmarkView() }
            NavigationLink("Real-photo test (experiment 8)") { RealPhotoView() }
        }
        .navigationTitle("Lab")
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
