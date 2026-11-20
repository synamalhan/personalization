import SwiftUI

/// Every taught class and how many examples it has. Deleting removes its embeddings and refits.
struct LibraryView: View {
    @Environment(AppModel.self) private var model
    @State private var classToForget: String?

    private var biggest: Int { max(1, model.classNames.map { model.count(for: $0) }.max() ?? 1) }

    var body: some View {
        NavigationStack {
            List {
                ScreenHeader(title: "What I know", subtitle: model.classNames.isEmpty ? "Nothing yet. Let's fix that." : "\(model.classNames.count) thing\(model.classNames.count == 1 ? "" : "s") in my head. Swipe one to forget it.", color: Theme.mint)
                    .plainRow()

                if model.classNames.isEmpty {
                    StickerCard(tint: Theme.sunflower, tilt: 1) {
                        VStack(spacing: 10) {
                            Text("🌱").font(.system(size: 56))
                            Text("My library is empty").font(.title3.weight(.heavy))
                            Text("Everything you teach me shows up here, with how many examples I've seen of each.")
                                .multilineTextAlignment(.center).font(.callout).foregroundStyle(Theme.ink.opacity(0.75))
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .plainRow()
                } else {
                    ForEach(model.classNames, id: \.self) { name in
                        classRow(name)
                            .plainRow()
                            .swipeActions(edge: .trailing) {
                                Button { classToForget = name } label: { Label("Forget", systemImage: "trash.fill") }
                                    .tint(Theme.tomato)
                            }
                    }
                }

                VStack(spacing: 14) {
                    Explainer(title: "What exactly do you store?", color: Theme.mint, icon: "externaldrive.fill") {
                        Text("For each thing: its name, and one short list of numbers per photo you showed me. That's all. A single entry is a few kilobytes, a lot less than a photo.")
                        Text("It lives in this app's private folder on your phone and is saved after every change.")
                    }
                    Explainer(title: "What happens when I forget something?", color: Theme.tomato, icon: "trash.fill") {
                        Text("Its numbers are deleted for good and the classifier is re-tuned from everything that's left, so the others stay sharp. If you chose to keep photos in Settings, that class's saved photos are deleted too.")
                        Text("There's no undo, but you can always teach it again.")
                    }
                    Explainer(title: "What does the bar mean?", color: Theme.sky, icon: "chart.bar.fill") {
                        Text("It compares how many example photos each thing has. Things with more examples are usually recognized more reliably, but past 10 or so the difference gets small.")
                    }
                }
                .plainRow()
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
            .background(ScreenBackground(color: Theme.mint))
            .toolbar(.hidden, for: .navigationBar)
            .confirmationDialog("Forget “\(classToForget ?? "")”?", isPresented: Binding(get: { classToForget != nil }, set: { if !$0 { classToForget = nil } }), titleVisibility: .visible) {
                Button("Forget it for good", role: .destructive) {
                    if let name = classToForget { Task { await model.deleteClass(name) } }
                    classToForget = nil
                }
                Button("Keep it", role: .cancel) { classToForget = nil }
            } message: {
                Text("I'll delete its examples and re-tune. This can't be undone.")
            }
        }
    }

    private func classRow(_ name: String) -> some View {
        let count = model.count(for: name)
        let color = ClassAvatar.color(for: name)
        return StickerCard(tint: color) {
            HStack(spacing: 14) {
                ClassAvatar(name: name)
                VStack(alignment: .leading, spacing: 6) {
                    Text(name).font(.title3.weight(.heavy)).foregroundStyle(Theme.ink).lineLimit(1)
                    GeometryReader { geo in
                        Capsule().fill(Theme.ink.opacity(0.08))
                            .overlay(alignment: .leading) {
                                Capsule().fill(color).frame(width: max(14, geo.size.width * CGFloat(count) / CGFloat(biggest)))
                            }
                            .overlay(Capsule().stroke(Theme.ink, lineWidth: 1.5))
                    }
                    .frame(height: 12)
                    Text("\(count) example\(count == 1 ? "" : "s")").font(.caption.weight(.semibold)).foregroundStyle(Theme.ink.opacity(0.7))
                }
                Button { classToForget = name } label: { Image(systemName: "trash").font(.headline).foregroundStyle(Theme.ink.opacity(0.6)) }
                    .buttonStyle(.plain)
            }
        }
    }
}

private extension View {
    /// Strips the default list-row look so our own cards show through.
    func plainRow() -> some View {
        self.listRowBackground(Color.clear)
            .listRowSeparator(.hidden)
            .listRowInsets(EdgeInsets(top: 6, leading: 20, bottom: 6, trailing: 20))
    }
}
