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
            .navigationTitle("洞察统计")
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
            sectionHeader("智能洞察", icon: "sparkles")

            ForEach(viewModel.insights, id: \.id) { insight in
                InsightCardView(insight: insight)
            }
        }
    }

    // MARK: - 周期选择器

    private var periodPicker: some View {
        Picker("周期", selection: $viewModel.selectedPeriod) {
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
            Text("正在加载统计数据...")
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
            Button("重试") {
                viewModel.loadData()
            }
            .buttonStyle(SunriseGoldButtonStyle())
        }
        .padding()
    }

    // MARK: - 快速统计卡片

    private func quickStatsSection(_ stats: AggregatedStats) -> some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            sectionHeader("概览", icon: "chart.bar.fill")

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: Spacing.sm) {
                StatCard(
                    title: "总目标",
                    value: "\(stats.totalGoals)",
                    icon: "target",
                    color: Color.Design.sunriseGold
                )

                StatCard(
                    title: "已分析",
                    value: "\(stats.analyzedGoals)",
                    icon: "brain.head.profile",
                    color: .cyan
                )

                StatCard(
                    title: "已纠正",
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
            sectionHeader("主题分布", icon: "tag.fill")

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
                sectionHeader("情感趋势", icon: "heart.fill")
                Spacer()
                Text(String(format: "平均: %.2f", trend.averageSentiment))
                    .font(Typography.caption)
                    .foregroundColor(sentimentColor(trend.averageSentiment))
            }

            // 使用 Swift Charts - 过滤有数据的点
            let filteredPoints = trend.dataPoints.filter { $0.value != 0 }
            Chart(filteredPoints) { point in
                LineMark(
                    x: .value("日期", point.date),
                    y: .value("情感", point.value)
                )
                .foregroundStyle(Color.Design.sunriseGold)

                PointMark(
                    x: .value("日期", point.date),
                    y: .value("情感", point.value)
                )
                .foregroundStyle(sentimentColor(point.value))
            }
            .chartYScale(domain: -1 ... 1)
            .chartYAxis {
                AxisMarks(values: [-1, 0, 1]) { value in
                    AxisGridLine()
                    AxisValueLabel {
                        if let labelValue = value.as(Double.self) {
                            Text(labelValue == 1 ? "积极" : (labelValue == -1 ? "消极" : "中性"))
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
            sectionHeader("目标数量趋势", icon: "chart.line.uptrend.xyaxis")

            Chart(viewModel.goalCountTrend) { point in
                BarMark(
                    x: .value("日期", point.date),
                    y: .value("数量", point.count)
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
            sectionHeader("星期分布", icon: "calendar")

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
                sectionHeader("模型置信度", icon: "gauge.with.dots.needle.bottom.50percent")
                Spacer()
                Text(String(format: "总体: %.0f%%", confidence.overall * 100))
                    .font(Typography.caption)
                    .foregroundColor(confidenceColor(confidence.overall))
            }

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: Spacing.sm) {
                ConfidenceRow(label: "主题", value: confidence.topic)
                ConfidenceRow(label: "情感", value: confidence.sentiment)
                ConfidenceRow(label: "紧急度", value: confidence.urgency)
                ConfidenceRow(label: "时间范围", value: confidence.timeFrame)
                ConfidenceRow(label: "行动类型", value: confidence.actionType)
                ConfidenceRow(label: "难度", value: confidence.difficulty)
                ConfidenceRow(label: "具体程度", value: confidence.specificity)
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
            Text(title)
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

                Text(title)
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
                Text(label)
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
