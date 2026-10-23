import SwiftUI
import UIKit

// MARK: - Otra sign search (two-step: search → confirm)

struct OtraSignSearchSheet: View {
    let labels: [String]
    let onConfirm: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var pendingSelection: String?

    private var filteredLabels: [String] {
        let sorted = labels.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return sorted }
        return sorted.filter { $0.localizedCaseInsensitiveContains(q) }
    }

    var body: some View {
        NavigationStack {
            Group {
                if let pending = pendingSelection {
                    confirmView(sign: pending)
                } else {
                    searchView
                }
            }
            .navigationTitle("Otra seña")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancelar") { dismiss() }
                }
            }
        }
    }

    private var searchView: some View {
        VStack(spacing: 0) {
            Text("Busca la seña correcta entre las \(labels.count) clases del modelo activo.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
                .padding(.top, 8)

            List(filteredLabels, id: \.self) { label in
                Button {
                    pendingSelection = label
                } label: {
                    Text(label)
                        .foregroundStyle(.primary)
                }
            }
            .listStyle(.plain)
            .searchable(text: $query, prompt: "Buscar seña…")
        }
    }

    private func confirmView(sign: String) -> some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "hand.raised.fill")
                .font(.system(size: 48))
                .foregroundStyle(.blue)

            VStack(spacing: 8) {
                Text("¿Confirmar seña?")
                    .font(.headline)
                Text(sign)
                    .font(.title2.bold())
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }

            Text("Revisa antes de confirmar para evitar selecciones accidentales.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)

            Spacer()

            VStack(spacing: 12) {
                Button {
                    onConfirm(sign)
                    dismiss()
                } label: {
                    Text("Confirmar")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                Button("Volver a buscar") {
                    pendingSelection = nil
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
    }
}

// MARK: - Grading option row

struct GradingOptionButton: View {
    let rank: Int
    let label: String
    let confidence: Float
    let isTop: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Text("\(rank)")
                    .font(.caption.bold())
                    .foregroundStyle(isTop ? .yellow : .white.opacity(0.5))
                    .frame(width: 18)

                Text(label)
                    .font(isTop ? .body.weight(.semibold) : .subheadline)
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)

                Spacer()

                Text(String(format: "%.1f%%", confidence * 100))
                    .font(.caption.bold().monospacedDigit())
                    .foregroundStyle(isTop ? .yellow : .white.opacity(0.65))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(isTop ? Color.yellow.opacity(0.12) : Color.white.opacity(0.06),
                        in: RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(isTop ? Color.yellow.opacity(0.35) : Color.clear, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Export preparing overlay

struct ExportPreparingOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                ProgressView()
                    .scaleEffect(1.4)
                    .tint(.white)

                Text("Preparando CSV para exportar")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
            }
            .padding(28)
            .frame(maxWidth: 300)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .strokeBorder(.white.opacity(0.15), lineWidth: 1)
            )
            .padding(.horizontal, 24)
        }
        .allowsHitTesting(true)
    }
}

// MARK: - Share sheet

/// Presents `UIActivityViewController` from a host VC (reliable with file URLs in Documents).
struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIViewController {
        let host = UIViewController()
        host.view.backgroundColor = .clear
        return host
    }

    func updateUIViewController(_ host: UIViewController, context: Context) {
        guard !context.coordinator.didPresent else { return }
        context.coordinator.didPresent = true

        DispatchQueue.main.async {
            guard host.presentedViewController == nil else { return }
            let activity = UIActivityViewController(activityItems: [url], applicationActivities: nil)
            activity.completionWithItemsHandler = { _, _, _, _ in
                dismiss()
            }
            if let popover = activity.popoverPresentationController {
                popover.sourceView = host.view
                popover.sourceRect = CGRect(
                    x: host.view.bounds.midX,
                    y: host.view.bounds.midY,
                    width: 0,
                    height: 0
                )
                popover.permittedArrowDirections = []
            }
            host.present(activity, animated: true)
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator {
        var didPresent = false
    }
}
