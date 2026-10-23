import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var processor = BatchProcessor()

    var body: some View {
        HSplitView {
            // ── Left panel: controls + log ─────────────────────────────────
            VStack(alignment: .leading, spacing: 0) {
                headerView
                Divider()
                folderSelectionView
                Divider()
                progressView
                Divider()
                logView
            }
            .frame(minWidth: 340, maxWidth: 400)

            // ── Right panel: video list ────────────────────────────────────
            videoListView
        }
        .frame(minWidth: 800, minHeight: 600)
    }

    // MARK: - Header

    var headerView: some View {
        HStack(spacing: 10) {
            Image(systemName: "hand.raised.fill")
                .font(.title2)
                .foregroundStyle(.teal)
            VStack(alignment: .leading, spacing: 1) {
                Text("LSM Extractor")
                    .font(.headline)
                Text("Vision Framework • macOS")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.bar)
    }

    // MARK: - Folder Selection

    var folderSelectionView: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Carpetas")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)

            FolderPickerRow(
                label: "Manifiesto (dataset_manifest.json)",
                systemImage: "doc.badge.gearshape",
                color: .orange,
                url: processor.manifestURL,
                placeholder: "Seleccionar manifiesto (opcional)..."
            ) {
                if let url = pickFile(title: "Seleccionar dataset_manifest.json", ext: "json") {
                    processor.loadManifest(from: url)
                }
            }

            FolderPickerRow(
                label: "Dataset",
                systemImage: "folder.fill",
                color: .blue,
                url: processor.inputURL,
                placeholder: "Seleccionar carpeta LSMVideos..."
            ) {
                if let url = pickFolder(title: "Seleccionar dataset") {
                    processor.loadDataset(from: url)
                }
            }

            FolderPickerRow(
                label: "Salida",
                systemImage: "folder.badge.plus",
                color: .green,
                url: processor.outputURL,
                placeholder: "Seleccionar carpeta LSMVideosOutput..."
            ) {
                if let url = pickFolder(title: "Seleccionar carpeta de salida") {
                    processor.outputURL = url
                    if processor.inputURL != nil {
                        processor.loadDataset(from: processor.inputURL!)
                    }
                }
            }

            Toggle(isOn: $processor.generateOverlay) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Generar video overlay")
                        .font(.subheadline)
                    Text("Desactivar para extraer solo JSON (más rápido)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.switch)
            .disabled(processor.isRunning)

            // Action buttons
            HStack(spacing: 8) {
                Button {
                    processor.startProcessing()
                } label: {
                    Label("Procesar", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!processor.canStart)

                if processor.isRunning {
                    Button {
                        processor.cancelProcessing()
                    } label: {
                        Label("Cancelar", systemImage: "stop.fill")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .tint(.red)
                }
            }

            if processor.isDone, let report = processor.report {
                HStack {
                    Image(systemName: report.errors > 0 ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .foregroundStyle(report.errors > 0 ? .orange : .green)
                    Text(processor.summaryText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Abrir salida") {
                        if let url = processor.outputURL {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                }
                .padding(8)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding(16)
    }

    // MARK: - Progress

    var progressView: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Progreso")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(processor.progress * 100))%")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: processor.progress)
                .progressViewStyle(.linear)
                .tint(.teal)

            if processor.isRunning {
                let done = processor.videos.filter { $0.status == .done || $0.status == .error }.count
                Text("\(done) de \(processor.videos.count) videos")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    // MARK: - Log

    var logView: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Log")
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 16)
                .padding(.top, 10)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(processor.log.enumerated()), id: \.offset) { i, line in
                            Text(line)
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(logColor(for: line))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(i)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                }
                .background(.quaternary.opacity(0.5))
                .onChange(of: processor.log.count) { _, _ in
                    if let last = processor.log.indices.last {
                        withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                    }
                }
            }
        }
        .frame(maxHeight: .infinity)
    }

    // MARK: - Video List

    var videoListView: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Videos")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
                Spacer()
                if !processor.videos.isEmpty {
                    let done  = processor.videos.filter { $0.status == .done }.count
                    let error = processor.videos.filter { $0.status == .error }.count
                    HStack(spacing: 8) {
                        if done > 0 {
                            Label("\(done)", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                                .font(.caption)
                        }
                        if error > 0 {
                            Label("\(error)", systemImage: "xmark.circle.fill")
                                .foregroundStyle(.red)
                                .font(.caption)
                        }
                        Text("\(processor.videos.count) total")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.bar)

            Divider()

            if processor.videos.isEmpty {
                emptyStateView
            } else {
                List(processor.videos) { item in
                    VideoRow(item: item)
                }
                .listStyle(.plain)
            }
        }
    }

    var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: "film.stack")
                .font(.system(size: 48))
                .foregroundStyle(.quaternary)
            Text("Selecciona la carpeta raíz del dataset")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text("Estructura esperada: category/word/*.mp4\nEjemplo: Familia/Abuela/Abuela1.mp4")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Helpers

    func pickFolder(title: String) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        return url
    }

    func pickFile(title: String, ext: String) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        return url
    }

    func logColor(for line: String) -> Color {
        if line.contains("❌") || line.contains("Error") { return .red }
        if line.contains("✅") || line.contains("✓")    { return .green }
        if line.contains("▶")                           { return .primary }
        if line.contains("🚀") || line.contains("📂")  { return .teal }
        return .secondary
    }
}

// MARK: - Subviews

struct FolderPickerRow: View {
    let label: String
    let systemImage: String
    let color: Color
    let url: URL?
    let placeholder: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                    .foregroundStyle(color)
                    .frame(width: 20)
                VStack(alignment: .leading, spacing: 1) {
                    Text(label)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(url?.lastPathComponent ?? placeholder)
                        .font(.subheadline)
                        .foregroundStyle(url == nil ? .tertiary : .primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }
}

struct VideoRow: View {
    let item: VideoItem

    var body: some View {
        HStack(spacing: 10) {
            // Status icon
            statusIcon
                .frame(width: 18)

            // Info
            VStack(alignment: .leading, spacing: 1) {
                Text(item.fileName)
                    .font(.subheadline)
                    .lineLimit(1)
                HStack(spacing: 4) {
                    Text(item.categoryName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("·")
                        .foregroundStyle(.tertiary)
                    Text(item.wordFolder)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if item.framesProcessed > 0 {
                        Text("·")
                            .foregroundStyle(.tertiary)
                        Text("\(item.framesProcessed) frames")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
            }

            Spacer()

            // Status badge
            statusBadge
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    var statusIcon: some View {
        switch item.status {
        case .pending:
            Image(systemName: "circle")
                .foregroundStyle(.tertiary)
        case .processing:
            ProgressView()
                .scaleEffect(0.7)
                .controlSize(.small)
        case .done:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .error:
            Image(systemName: "xmark.circle.fill")
                .foregroundStyle(.red)
        case .skipped:
            Image(systemName: "slash.circle")
                .foregroundStyle(.yellow)
        }
    }

    @ViewBuilder
    var statusBadge: some View {
        switch item.status {
        case .pending:
            EmptyView()
        case .processing:
            Text("Procesando")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.teal.opacity(0.15))
                .foregroundStyle(.teal)
                .clipShape(Capsule())
        case .done:
            EmptyView()
        case .error:
            Text("Error")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.red.opacity(0.15))
                .foregroundStyle(.red)
                .clipShape(Capsule())
        case .skipped:
            Text("Skipped")
                .font(.caption2)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.red.opacity(0.15))
                .foregroundStyle(.red)
                .clipShape(Capsule())
        }
    }
}

#Preview {
    ContentView()
}
