import CoreData
import SwiftUI

// MARK: - 洞察卡片视图

struct InsightCardView: View {
    let insight: Insight

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: insight.iconName)
                    .foregroundColor(Color.Design.accentCyan)
                    .font(.title2)

                Text(insight.title)
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.softWhite)

                Spacer()
            }

            Text(insight.description)
                .font(Typography.body)
                .foregroundColor(Color.Design.mutedGray)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .background(Color.Design.cardBackground)
        .cornerRadius(CornerRadius.md)
        .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}

// MARK: - 洞察列表视图

struct InsightsView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var insights: [Insight] = []
    @State private var isLoading = true

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                if isLoading {
                    ProgressView("正在分析...")
                        .padding()
                } else if insights.isEmpty {
                    emptyStateView
                } else {
                    ForEach(insights.indices, id: \.self) { index in
                        InsightCardView(insight: insights[index])
                    }
                }
            }
            .padding()
        }
        .navigationTitle("洞察")
        .navigationBarTitleDisplayMode(.large)
        .task {
            await loadInsights()
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.bar.doc.horizontal")
                .font(.system(size: 60))
                .foregroundColor(.gray)

            Text("暂无洞察")
                .font(.title3)
                .foregroundColor(.secondary)

            Text("至少记录7天目标后，我们将为你生成个性化洞察")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding(.top, 80)
    }

    private func loadInsights() async {
        // 异步执行洞察生成（避免阻塞UI）
        await Task {
            let engine = InsightEngine(viewContext: context)
            let generatedInsights = await engine.generateInsights(for: 30)

            await MainActor.run {
                insights = generatedInsights
                isLoading = false
            }
        }.value
    }
}

// MARK: - 目标详情视图（包含分类显示和纠正功能）

// MARK: - 预览

#Preview("Insight Card") {
    InsightCardView(
        insight: Insight(
            type: .balance,
            title: "Balance Check",
            description: "You have a good balance.",
            priority: 1,
            actionSuggestion: nil
        )
    )
    .padding()
}

#Preview("Insights View") {
    NavigationView {
        InsightsView()
            .environment(\.managedObjectContext, DataController.shared.container.viewContext)
    }
}
