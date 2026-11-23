import SwiftUI

enum SessionRoute: Hashable {
    case preparing(bundleName: String)
    case live
}

enum CatalogFamilyScope: String, CaseIterable, Hashable, Identifiable {
    case all
    case fingerprint
    case cnn
    case tcn
    case cnn3d
    case legacy

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: return "Todas"
        case .fingerprint: return ModelFamily.fingerprint.title
        case .cnn: return ModelFamily.cnn.title
        case .tcn: return ModelFamily.tcn.title
        case .cnn3d: return ModelFamily.cnn3d.title
        case .legacy: return ModelFamily.legacy.title
        }
    }

    var family: ModelFamily? {
        switch self {
        case .all: return nil
        case .fingerprint: return .fingerprint
        case .cnn: return .cnn
        case .tcn: return .tcn
        case .cnn3d: return .cnn3d
        case .legacy: return .legacy
        }
    }
}

enum ModelCatalog {
    static func sections(
        models: [LSMModel],
        query: String,
        scope: CatalogFamilyScope
    ) -> [(set: SignSet, models: [LSMModel])] {
        let foldedQuery = folded(query).trimmingCharacters(in: .whitespacesAndNewlines)
        let filtered = models.filter { model in
            if let family = scope.family, model.family != family { return false }
            guard !foldedQuery.isEmpty else { return true }
            let haystack = folded([
                model.name,
                model.description,
                model.family.title,
                model.signSet.title,
                model.signSet.detail
            ].joined(separator: " "))
            return haystack.contains(foldedQuery)
        }
        return SignSet.allCases.compactMap { set in
            let rows = filtered
                .filter { $0.signSet == set }
                .sorted { $0.family.sortIndex < $1.family.sortIndex }
            return rows.isEmpty ? nil : (set, rows)
        }
    }

    private static func folded(_ text: String) -> String {
        text.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
    }
}

struct ModelBrowserView: View {
    @ObservedObject var pipeline: LandmarkPipeline
    var initialQuery: String = ""
    var initialPendingBundleName: String? = nil
    let onConfirm: (LSMModel) -> Void
    let onReturnToCamera: () -> Void

    @State private var query = ""
    @State private var scope: CatalogFamilyScope = .all
    @State private var pendingModel: LSMModel?
    @AccessibilityFocusState private var focusedBundleName: String?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let sections = ModelCatalog.sections(
            models: AVAILABLE_MODELS,
            query: query,
            scope: scope
        )
        List {
            if sections.isEmpty {
                ContentUnavailableView(
                    "Sin modelos",
                    systemImage: "magnifyingglass",
                    description: Text("No hay modelos que coincidan con la búsqueda.")
                )
            } else {
                ForEach(sections, id: \.set) { section in
                    Section {
                        ForEach(section.models) { model in
                            modelRow(model)
                        }
                    } header: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(section.set.title)
                            Text(section.set.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .textCase(nil)
                        .accessibilityElement(children: .combine)
                    }
                }
            }
        }
        .navigationTitle("Modelos")
        .searchable(
            text: $query,
            placement: .navigationBarDrawer(displayMode: .always),
            prompt: "Buscar modelo"
        )
        .searchScopes($scope) {
            ForEach(CatalogFamilyScope.allCases) { item in
                Text(item.title).tag(item)
            }
        }
        .toolbar {
            if pipeline.activeModel != nil {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cámara", action: onReturnToCamera)
                }
            }
        }
        .overlay {
            if let model = pendingModel {
                ModelConfirmCenter(
                    model: model,
                    onConfirm: {
                        onConfirm(model)
                        setPending(nil)
                    },
                    onCancel: { dismissConfirm(returningFocusTo: model.bundleName) }
                )
                .transition(.opacity)
            }
        }
        .onAppear {
            if query.isEmpty, !initialQuery.isEmpty {
                query = initialQuery
            }
            if pendingModel == nil, let name = initialPendingBundleName {
                pendingModel = AVAILABLE_MODELS.first { $0.bundleName == name }
            }
        }
    }

    private func presentConfirm(_ model: LSMModel) {
        setPending(model)
    }

    private func dismissConfirm(returningFocusTo bundleName: String) {
        setPending(nil)
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(100))
            focusedBundleName = bundleName
        }
    }

    private func setPending(_ model: LSMModel?) {
        if reduceMotion {
            pendingModel = model
        } else {
            withAnimation(.easeInOut(duration: 0.2)) {
                pendingModel = model
            }
        }
    }

    private func modelRow(_ model: LSMModel) -> some View {
        Button {
            presentConfirm(model)
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.family.title)
                    .font(.headline)
                Text(model.name)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text(model.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(model.family.title). \(model.signSet.title). \(model.description)")
        .accessibilityHint("Pide confirmación antes de cargar")
        .accessibilityFocused($focusedBundleName, equals: model.bundleName)
    }
}

private struct ModelConfirmCenter: View {
    let model: LSMModel
    let onConfirm: () -> Void
    let onCancel: () -> Void

    @AccessibilityFocusState private var dialogFocused: Bool

    var body: some View {
        ZStack {
            Color.black.opacity(0.45)
                .ignoresSafeArea()
                .onTapGesture(perform: onCancel)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 14) {
                Text("¿Cargar este modelo?")
                    .font(.title3.weight(.semibold))
                Text(model.family.title)
                    .font(.headline)
                Text(model.signSet.title)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text(model.description)
                    .font(.subheadline)
                HStack(spacing: 12) {
                    Button("Cancelar", action: onCancel)
                        .buttonStyle(.bordered)
                    Button("Cargar", action: onConfirm)
                        .buttonStyle(.borderedProminent)
                }
                .frame(maxWidth: .infinity, alignment: .trailing)
                .padding(.top, 4)
            }
            .padding(24)
            .frame(maxWidth: 360)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .padding(.horizontal, 24)
            .accessibilityElement(children: .contain)
            .accessibilityAddTraits(.isModal)
            .accessibilityFocused($dialogFocused)
        }
        .onAppear { dialogFocused = true }
    }
}

#Preview("Catálogo") {
    NavigationStack {
        ModelBrowserView(
            pipeline: LandmarkPipeline(),
            onConfirm: { _ in },
            onReturnToCamera: {}
        )
    }
}

#Preview("Confirmación") {
    NavigationStack {
        ModelBrowserView(
            pipeline: LandmarkPipeline(),
            initialPendingBundleName: "runs_m4_min7_cnn_lsmoutput",
            onConfirm: { _ in },
            onReturnToCamera: {}
        )
    }
}

#Preview("Sin resultados") {
    NavigationStack {
        ModelBrowserView(
            pipeline: LandmarkPipeline(),
            initialQuery: "zzzz",
            onConfirm: { _ in },
            onReturnToCamera: {}
        )
    }
}
