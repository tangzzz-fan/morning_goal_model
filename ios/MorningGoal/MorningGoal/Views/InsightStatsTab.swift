//
//  InsightStatsTab.swift
//  MorningGoal
//
//  洞察统计Tab - 展示目标数据的聚合统计、分布和趋势
//

import Charts
import CoreData
import SwiftUI

struct InsightStatsTab: View {
    // ViewModel
    @StateObject private var viewModel: InsightViewModel

    init(context: NSManagedObjectContext) {
        _viewModel = StateObject(wrappedValue: InsightViewModel(context: context))
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.Design.deepIndigo.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: Spacing.lg) {
                        // 周期选择器
                        periodPicker

                        if viewModel.isLoading {
                            loadingView
                        } else if let error = viewModel.errorMessage {
                            errorView(error)
                        } else {
                            // 快速统计卡片
                            if let stats = viewModel.stats {
                                quickStatsSection(stats)
                            }

                            // 智能洞察
                            if !viewModel.insights.isEmpty {
                                insightsSection
                            }

                            // 主题分布
                            if !viewModel.topicDistribution.isEmpty {
                                topicDistributionSection
                            }

                            // 情感趋势图表
                            if let trend = viewModel.sentimentTrend, !trend.dataPoints.isEmpty {
                                sentimentTrendSection(trend)
                            }

                            // 目标数量趋势
                            if !viewModel.goalCountTrend.isEmpty {
                                goalCountTrendSection
                            }

                            // 星期分布
                            if !viewModel.weekdayDistribution.isEmpty {
                                weekdayDistributionSection
                            }

                            // 置信度统计
                            if let stats = viewModel.stats {
                                confidenceSection(stats.averageConfidence)
                            }
                        }

                        Spacer(minLength: 100)
                    }
                    .padding(.horizontal)
                    .padding(.top, Spacing.md)
                }
            }
            .navigationTitle(LocalizedStringKey("stats_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { viewModel.loadData() }) {
                        Image(systemName: "arrow.clockwise")
                            .foregroundColor(Color.Design.sunriseGold)
                    }
                }
            }
            .onAppear {
                viewModel.loadData()
            }
        }
    }

    // MARK: - 智能洞察

    private var insightsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            sectionHeader("stats_section_smart_insights", icon: "sparkles")

            ForEach(viewModel.insights, id: \.id) { insight in
                InsightCardView(insight: insight)
            }
        }
    }

    // MARK: - 周期选择器

    private var periodPicker: some View {
        Picker(LocalizedStringKey("stats_period_picker"), selection: $viewModel.selectedPeriod) {
            ForEach(AggregationPeriod.allCases, id: \.self) { period in
                Text(period.displayName).tag(period)
            }
        }
        .pickerStyle(.segmented)
        .padding(.horizontal)
    }

    // MARK: - 加载视图

    private var loadingView: some View {
        VStack(spacing: Spacing.md) {
            ProgressView()
                .tint(Color.Design.sunriseGold)
            Text(LocalizedStringKey("stats_loading"))
                .font(Typography.caption)
                .foregroundColor(Color.Design.mutedGray)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Spacing.xl)
    }

    // MARK: - 错误视图

    private func errorView(_ message: String) -> some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40))
                .foregroundColor(.orange)
            Text(message)
                .font(Typography.body)
                .foregroundColor(Color.Design.softWhite)
                .multilineTextAlignment(.center)
            Button(LocalizedStringKey("stats_error_retry")) {
                viewModel.loadData()
            }
            .buttonStyle(SunriseGoldButtonStyle())
        }
        .padding()
    }

    // MARK: - 快速统计卡片

    private func quickStatsSection(_ stats: AggregatedStats) -> some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            sectionHeader("stats_section_overview", icon: "chart.bar.fill")

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: Spacing.sm) {
                StatCard(
                    title: "stats_total_goals",
                    value: "\(stats.totalGoals)",
                    icon: "target",
                    color: Color.Design.sunriseGold
                )

                StatCard(
                    title: "stats_analyzed",
                    value: "\(stats.analyzedGoals)",
                    icon: "brain.head.profile",
                    color: .cyan
                )

                StatCard(
                    title: "stats_corrected",
                    value: "\(stats.correctedGoals)",
                    icon: "pencil.and.outline",
                    color: .green
                )
            }
        }
    }

    // MARK: - 主题分布

    private var topicDistributionSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            sectionHeader("stats_section_topic_dist", icon: "tag.fill")

            VStack(spacing: Spacing.xs) {
                ForEach(viewModel.topicDistribution.prefix(8)) { item in
                    HStack {
                        if let icon = item.icon {
                            Image(systemName: icon)
                                .foregroundColor(Color.Design.sunriseGold)
                                .frame(width: 24)
                        }

                        Text(item.label)
                            .font(Typography.body)
                            .foregroundColor(Color.Design.softWhite)

                        Spacer()

                        Text("\(item.count)")
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.mutedGray)

                        // 进度条
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Color.Design.darkIndigo)
                                    .frame(height: 6)

                                RoundedRectangle(cornerRadius: 2)
                                    .fill(Color.Design.sunriseGold)
                                    .frame(width: geo.size.width * item.percentage / 100, height: 6)
                            }
                        }
                        .frame(width: 80, height: 6)

                        Text(String(format: "%.0f%%", item.percentage))
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.sunriseGold)
                            .frame(width: 40, alignment: .trailing)
                    }
                    .padding(.vertical, 4)
                }
            }
            .padding()
            .background(Color.Design.darkIndigo.opacity(0.5))
            .cornerRadius(CornerRadius.md)
        }
    }

    // MARK: - 情感趋势图表

    private func sentimentTrendSection(_ trend: SentimentTrendResult) -> some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                sectionHeader("stats_section_sentiment_trend", icon: "heart.fill")
                Spacer()
                Text(String(format: NSLocalizedString("stats_avg_sentiment", comment: ""), trend.averageSentiment))
                    .font(Typography.caption)
                    .foregroundColor(sentimentColor(trend.averageSentiment))
            }

            // 使用 Swift Charts - 过滤有数据的点
            let filteredPoints = trend.dataPoints.filter { $0.value != 0 }
            Chart(filteredPoints) { point in
                LineMark(
                    x: .value(LocalizedStringKey("stats_chart_date"), point.date),
                    y: .value(LocalizedStringKey("stats_chart_sentiment"), point.value)
                )
                .foregroundStyle(Color.Design.sunriseGold)

                PointMark(
                    x: .value(LocalizedStringKey("stats_chart_date"), point.date),
                    y: .value(LocalizedStringKey("stats_chart_sentiment"), point.value)
                )
                .foregroundStyle(sentimentColor(point.value))
            }
            .chartYScale(domain: -1 ... 1)
            .chartYAxis {
                AxisMarks(values: [-1, 0, 1]) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let labelValue = value.as(Double.self) {
                            Text(labelValue == 1 ? LocalizedStringKey("stats_sentiment_positive") :
                                (labelValue == -1 ? LocalizedStringKey("stats_sentiment_negative") : LocalizedStringKey("stats_sentiment_neutral")))
                                .font(.system(size: 10))
                                .foregroundColor(Color.Design.mutedGray)
                        }
                    }
                }
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { _ in
                    AxisGridLine()
                    AxisValueLabel(format: .dateTime.month().day())
                }
            }
            .frame(height: 200)
            .padding()
            .background(Color.Design.darkIndigo.opacity(0.5))
            .cornerRadius(CornerRadius.md)
        }
    }

    // MARK: - 目标数量趋势

    private var goalCountTrendSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            sectionHeader("stats_section_goal_trend", icon: "chart.line.uptrend.xyaxis")

            Chart(viewModel.goalCountTrend) { point in
                BarMark(
                    x: .value(LocalizedStringKey("stats_chart_date"), point.date),
                    y: .value(LocalizedStringKey("stats_chart_count"), point.count)
                )
                .foregroundStyle(
                    point.value != 0 ? Color.Design.sunriseGold : Color.Design.mutedGray.opacity(0.3)
                )
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { _ in
                    AxisGridLine()
                    AxisValueLabel(format: .dateTime.month().day())
                }
            }
            .frame(height: 150)
            .padding()
            .background(Color.Design.darkIndigo.opacity(0.5))
            .cornerRadius(CornerRadius.md)
        }
    }

    // MARK: - 星期分布

    private var weekdayDistributionSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            sectionHeader("stats_section_weekday_dist", icon: "calendar")

            HStack(spacing: 4) {
                ForEach(viewModel.weekdayDistribution) { item in
                    let itemCount = item.count
                    VStack(spacing: 4) {
                        Text("\(itemCount)")
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.sunriseGold)

                        RoundedRectangle(cornerRadius: 4)
                            .fill(itemCount > 0 ? Color.Design.sunriseGold : Color.Design.darkIndigo)
                            .frame(height: max(10, CGFloat(itemCount) * 10))

                        Text(String(item.label.suffix(1)))
                            .font(.system(size: 10))
                            .foregroundColor(Color.Design.mutedGray)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(height: 120)
            .padding()
            .background(Color.Design.darkIndigo.opacity(0.5))
            .cornerRadius(CornerRadius.md)
        }
    }

    // MARK: - 置信度统计

    private func confidenceSection(_ confidence: DimensionConfidence) -> some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                sectionHeader("stats_section_confidence", icon: "gauge.with.dots.needle.bottom.50percent")
                Spacer()
                Text(String(format: NSLocalizedString("stats_confidence_overall", comment: ""), confidence.overall * 100))
                    .font(Typography.caption)
                    .foregroundColor(confidenceColor(confidence.overall))
            }

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: Spacing.sm) {
                ConfidenceRow(label: "dimension_topic", value: confidence.topic)
                ConfidenceRow(label: "dimension_sentiment", value: confidence.sentiment)
                ConfidenceRow(label: "dimension_urgency", value: confidence.urgency)
                ConfidenceRow(label: "dimension_timeframe", value: confidence.timeFrame)
                ConfidenceRow(label: "dimension_action_type", value: confidence.actionType)
                ConfidenceRow(label: "dimension_difficulty", value: confidence.difficulty)
                ConfidenceRow(label: "dimension_specificity", value: confidence.specificity)
            }
            .padding()
            .background(Color.Design.darkIndigo.opacity(0.5))
            .cornerRadius(CornerRadius.md)
        }
    }

    // MARK: - 辅助视图

    private func sectionHeader(_ title: String, icon: String) -> some View {
        HStack(spacing: Spacing.xs) {
            Image(systemName: icon)
                .foregroundColor(Color.Design.sunriseGold)
            Text(LocalizedStringKey(title))
                .font(Typography.headline)
                .foregroundColor(Color.Design.softWhite)
        }
    }

    private func sentimentColor(_ value: Double) -> Color {
        if value > 0.3 { return .green }
        if value < -0.3 { return .red }
        return .gray
    }

    private func confidenceColor(_ value: Double) -> Color {
        if value >= 0.8 { return .green }
        if value >= 0.6 { return .orange }
        return .red
    }

    // MARK: - 辅助组件

    private struct StatCard: View {
        let title: String
        let value: String
        let icon: String
        let color: Color

        var body: some View {
            VStack(spacing: Spacing.xs) {
                Image(systemName: icon)
                    .font(.system(size: 20))
                    .foregroundColor(color)

                Text(value)
                    .font(Typography.title)
                    .foregroundColor(Color.Design.softWhite)

                Text(LocalizedStringKey(title))
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.mutedGray)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, Spacing.md)
            .background(Color.Design.darkIndigo.opacity(0.5))
            .cornerRadius(CornerRadius.md)
        }
    }

    private struct ConfidenceRow: View {
        let label: String
        let value: Double

        var body: some View {
            HStack {
                Text(LocalizedStringKey(label))
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.softWhite)

                Spacer()

                // 进度条
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(Color.Design.deepIndigo)
                            .frame(height: 4)

                        RoundedRectangle(cornerRadius: 2)
                            .fill(confidenceColor)
                            .frame(width: geo.size.width * value, height: 4)
                    }
                }
                .frame(width: 60, height: 4)

                Text(String(format: "%.0f%%", value * 100))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundColor(confidenceColor)
                    .frame(width: 36, alignment: .trailing)
            }
        }

        private var confidenceColor: Color {
            if value >= 0.8 { return .green }
            if value >= 0.6 { return .orange }
            return .red
        }
    }
}

#Preview {
    InsightStatsTab(context: DataController.preview.container.viewContext)
}
