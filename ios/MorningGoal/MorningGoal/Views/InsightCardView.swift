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
            let engine = InsightEngine(context: context)
            let generatedInsights = engine.generateInsights(for: 30)

            await MainActor.run {
                insights = generatedInsights
                isLoading = false
            }
        }.value
    }
}

// MARK: - 目标详情视图（包含分类显示和纠正功能）

struct GoalDetailView: View {
    @Environment(\.managedObjectContext) private var context
    @ObservedObject var entry: GoalEntry

    @State private var showCategoryPicker = false
    @State private var showSentimentPicker = false

    private let categories = ["工作", "健康", "家庭", "个人发展", "财务", "学习", "社交", "休闲"]
    private let sentiments = ["积极", "中性", "消极"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // 目标文本
                VStack(alignment: .leading, spacing: 8) {
                    Text("目标")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Text(entry.goalText)
                        .font(.title3)
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)

                Divider()

                // 主题分类
                VStack(alignment: .leading, spacing: 12) {
                    Text("主题分类")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack {
                        if let category = entry.effectiveCategory {
                            Text(category)
                                .font(.body)
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.blue)
                                .cornerRadius(8)
                        } else {
                            Text("未分类")
                                .font(.body)
                                .foregroundColor(.secondary)
                        }

                        if entry.categoryConfidence > 0 {
                            Text("\(Int(entry.categoryConfidence * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Button {
                            showCategoryPicker.toggle()
                        } label: {
                            Label("修改", systemImage: "pencil")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                    }

                    if showCategoryPicker {
                        CategoryPickerView(
                            selected: Binding(
                                get: { entry.categoryUserCorrected ?? entry.category ?? categories[0] },
                                set: { correctCategory($0) }
                            ),
                            categories: categories
                        )
                        .padding(.top, 8)
                    }
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.05), radius: 2)

                // 情感倾向
                VStack(alignment: .leading, spacing: 12) {
                    Text("情感倾向")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack {
                        if let sentiment = entry.effectiveSentiment {
                            Text(sentiment)
                                .font(.body)
                                .foregroundColor(.white)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(sentimentColor(sentiment))
                                .cornerRadius(8)
                        } else {
                            Text("未分析")
                                .font(.body)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Button {
                            showSentimentPicker.toggle()
                        } label: {
                            Label("修改", systemImage: "pencil")
                                .font(.caption)
                        }
                        .buttonStyle(.bordered)
                    }

                    if showSentimentPicker {
                        SentimentPickerView(
                            selected: Binding(
                                get: { entry.sentimentUserCorrected ?? entry.sentiment ?? sentiments[1] },
                                set: { correctSentiment($0) }
                            ),
                            sentiments: sentiments
                        )
                        .padding(.top, 8)
                    }
                }
                .padding()
                .background(Color(.systemBackground))
                .cornerRadius(12)
                .shadow(color: Color.black.opacity(0.05), radius: 2)

                // 分析时间
                if let analyzedAt = entry.analyzedAt {
                    HStack {
                        Image(systemName: "clock")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        Text("分析于 \(analyzedAt, style: .relative)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                }

                Spacer()
            }
            .padding()
        }
        .navigationTitle("目标详情")
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - 纠正逻辑

    private func correctCategory(_ newCategory: String) {
        entry.categoryUserCorrected = newCategory
        entry.correctedAt = Date()
        entry.isTrainingSample = true

        do {
            try context.save()

            // 触发模型更新调度
            ModelUpdateScheduler.shared.scheduleModelUpdate()

            print("✅ 用户纠正主题: \(newCategory)")

            withAnimation {
                showCategoryPicker = false
            }
        } catch {
            print("❌ 保存纠正失败: \(error)")
        }
    }

    private func correctSentiment(_ newSentiment: String) {
        entry.sentimentUserCorrected = newSentiment
        entry.correctedAt = Date()
        entry.isTrainingSample = true

        do {
            try context.save()

            ModelUpdateScheduler.shared.scheduleModelUpdate()

            print("✅ 用户纠正情感: \(newSentiment)")

            withAnimation {
                showSentimentPicker = false
            }
        } catch {
            print("❌ 保存纠正失败: \(error)")
        }
    }

    private func sentimentColor(_ sentiment: String) -> Color {
        switch sentiment {
        case "积极": return .green
        case "消极": return .red
        default: return .gray
        }
    }
}

// MARK: - 分类选择器

struct CategoryPickerView: View {
    @Binding var selected: String
    let categories: [String]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(categories, id: \.self) { category in
                    Button {
                        selected = category
                    } label: {
                        Text(category)
                            .font(.subheadline)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(selected == category ? Color.blue : Color.gray.opacity(0.2))
                            .foregroundColor(selected == category ? .white : .primary)
                            .cornerRadius(20)
                    }
                }
            }
        }
    }
}

// MARK: - 情感选择器

struct SentimentPickerView: View {
    @Binding var selected: String
    let sentiments: [String]

    var body: some View {
        HStack(spacing: 16) {
            ForEach(sentiments, id: \.self) { sentiment in
                Button {
                    selected = sentiment
                } label: {
                    Text(sentiment)
                        .font(.subheadline)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .frame(maxWidth: .infinity)
                        .background(selected == sentiment ? sentimentColor(sentiment) : Color.gray.opacity(0.2))
                        .foregroundColor(selected == sentiment ? .white : .primary)
                        .cornerRadius(20)
                }
            }
        }
    }

    private func sentimentColor(_ sentiment: String) -> Color {
        switch sentiment {
        case "积极": return .green
        case "消极": return .red
        default: return .gray
        }
    }
}

// MARK: - 预览

#Preview("Insight Card") {
    InsightCardView(insight: .categoryDistribution(topCategory: "工作", percentage: 40))
        .padding()
}

#Preview("Insights View") {
    NavigationView {
        InsightsView()
            .environment(\.managedObjectContext, DataController.shared.container.viewContext)
    }
}
