import SwiftUI

/// Every taught class and how many examples it has. Deleting removes its embeddings and refits.
struct LibraryView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        NavigationStack {
            Group {
                if model.classNames.isEmpty {
                    ContentUnavailableView("Nothing taught yet", systemImage: "books.vertical",
                                           description: Text("Classes you teach will show up here."))
                } else {
                    List {
                        ForEach(model.classNames, id: \.self) { name in
                            HStack {
                                Text(name)
                                Spacer()
                                Text("\(model.count(for: name)) examples").foregroundStyle(.secondary)
                            }
                            .swipeActions {
                                Button("Delete", role: .destructive) { Task { await model.deleteClass(name) } }
                            }
                        }
                        .onDelete { offsets in
                            let names = offsets.map { model.classNames[$0] }
                            Task { for name in names { await model.deleteClass(name) } }
                        }
                    }
                }
            }
            .navigationTitle("Library")
            .toolbar { if !model.classNames.isEmpty { EditButton() } }
        }
    }
}
