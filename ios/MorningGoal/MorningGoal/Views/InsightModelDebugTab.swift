//
//  InsightModelDebugTab.swift
//  MorningGoal
//
//  洞察模型调试Tab - 用于测试7个洞察分类器，支持用户纠正和设备端训练
//

import CoreML
import SwiftUI

/// 洞察模型调试视图
@MainActor
struct InsightModelDebugTab: View {
    // MARK: - State

    @StateObject private var modelManager: InsightModelManagerWrapper = .init()
    @StateObject private var updateManager = InsightUpdateManager()

    @State private var inputText: String = ""
    @State private var analysisResult: InsightAnalysisResult?
    @State private var isAnalyzing: Bool = false
    @State private var errorMessage: String?
    @State private var lastEmbedding: MLMultiArray?

    // 纠正 UI 状态
    @State private var showCorrectionUI: Bool = false
    @State private var selectedTopicIndex: Int = 0
    @State private var selectedSentimentIndex: Int = 0
    @State private var selectedUrgencyIndex: Int = 0
    @State private var selectedTimeframeIndex: Int = 0
    @State private var selectedActiontypeIndex: Int = 0
    @State private var selectedDifficultyIndex: Int = 0
    @State private var selectedSpecificityIndex: Int = 0

    // 预设测试用例
    private let presets: [(label: String, text: String)] = [
        ("紧急工作", "必须今天完成这个项目报告，明天就要提交"),
        ("长期学习", "这个月开始系统学习 Swift 编程语言"),
        ("运动健身", "晚上去健身房跑步30分钟，然后做力量训练"),
        ("模糊目标", "希望变得更好"),
        ("具体目标", "下周一前阅读《深入理解Swift》第3-5章并做笔记"),
        ("社交活动", "周末约朋友一起去咖啡厅聊天"),
        ("创意项目", "开始设计新的App图标和UI界面"),
        ("困难任务", "完成复杂的机器学习算法优化")
    ]

    // 标签数组
    private let topicLabels = [
        "工作", "健康", "家庭", "个人发展", "理财", "社交", "家务", "学习",
        "睡眠", "饮食", "心态", "娱乐", "出行", "职业发展", "沟通", "育儿"
    ]
    private let sentimentLabels = ["消极", "中性", "积极"]
    private let urgencyLabels = ["低", "中", "高"]
    private let timeframeLabels = ["今天", "本周", "本月", "长期"]
    private let actiontypeLabels = ["学习", "运动", "工作", "生活", "社交"]
    private let difficultyLabels = ["简单", "中等", "困难"]
    private let specificityLabels = ["模糊", "一般", "具体"]

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ZStack {
                Color.Design.deepIndigo.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: Spacing.lg) {
                        statusCard
                        inputSection
                        presetsSection

                        if let result = analysisResult {
                            resultSection(result)
                        }

                        if showCorrectionUI {
                            correctionSection
                        }

                        trainingSection

                        if let error = errorMessage {
                            errorCard(error)
                        }

                        modelInfoSection

                        Spacer(minLength: 100)
                    }
                    .padding()
                }
            }
            .navigationTitle("洞察模型调试")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                Task {
                    await modelManager.loadModels()
                }
            }
        }
    }

    // MARK: - Status Card

    private var statusCard: some View {
        HStack(spacing: Spacing.md) {
            Circle()
                .fill(modelManager.isInitialized ? Color.green : Color.orange)
                .frame(width: 12, height: 12)

            VStack(alignment: .leading, spacing: 2) {
                Text(modelManager.isInitialized ? "模型就绪" : "正在加载...")
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.softWhite)

                Text(modelManager.statusMessage)
                    .font(.system(size: 11))
                    .foregroundColor(Color.Design.mutedGray)
            }

            Spacer()

            if modelManager.isInitialized {
                Text("\(modelManager.loadedClassifiers.count)/7")
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.sunriseGold)
            } else {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: Color.Design.softWhite))
            }
        }
        .padding()
        .background(Color.Design.darkIndigo.opacity(0.6))
        .cornerRadius(CornerRadius.md)
    }

    // MARK: - Input Section

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Label("输入目标文本", systemImage: "text.alignleft")
                .font(Typography.caption)
                .foregroundColor(Color.Design.softWhite)

            TextEditor(text: $inputText)
                .frame(height: 100)
                .padding(Spacing.sm)
                .background(Color.Design.darkIndigo.opacity(0.4))
                .cornerRadius(CornerRadius.sm)
                .foregroundColor(Color.Design.softWhite)
                .overlay(
                    RoundedRectangle(cornerRadius: CornerRadius.sm)
                        .stroke(Color.Design.mutedGray.opacity(0.3), lineWidth: 1)
                )
                .scrollContentBackground(.hidden)

            Button(action: runAnalysis) {
                HStack {
                    if isAnalyzing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: "wand.and.stars")
                    }
                    Text(isAnalyzing ? "分析中..." : (updateManager.isTraining ? "训练中..." : "运行分析"))
                        .fontWeight(.medium)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(
                    modelManager.isInitialized && !inputText.isEmpty && !updateManager.isTraining
                        ? Color.Design.sunriseGold
                        : Color.gray
                )
                .foregroundColor(.white)
                .cornerRadius(CornerRadius.md)
            }
            .disabled(!modelManager.isInitialized || inputText.isEmpty || isAnalyzing || updateManager.isTraining)
        }
    }

    // MARK: - Presets Section

    private var presetsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Label("快速测试", systemImage: "bolt.fill")
                .font(Typography.caption)
                .foregroundColor(Color.Design.softWhite)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: Spacing.sm) {
                ForEach(presets, id: \.label) { preset in
                    Button {
                        inputText = preset.text
                    } label: {
                        Text(preset.label)
                            .font(.system(size: 12))
                            .padding(.horizontal, Spacing.sm)
                            .padding(.vertical, Spacing.xs)
                            .background(Color.Design.darkIndigo.opacity(0.6))
                            .foregroundColor(Color.Design.softWhite)
                            .cornerRadius(CornerRadius.sm)
                    }
                }
            }
        }
    }

    // MARK: - Result Section

    private func resultSection(_ result: InsightAnalysisResult) -> some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            HStack {
                Label("分析结果 (7维度)", systemImage: "chart.bar.fill")
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.sunriseGold)

                Spacer()

                VStack(alignment: .trailing, spacing: 2) {
                    Text("总耗时: \(String(format: "%.1f", result.inferenceTimeMs)) ms")
                        .font(.system(size: 10))
                        .foregroundColor(Color.Design.mutedGray)
                    Text("特征提取: \(String(format: "%.1f", result.featureExtractionTimeMs)) ms")
                        .font(.system(size: 10))
                        .foregroundColor(Color.Design.mutedGray)
                }
            }

            // 使用 Grid 布局显示 7 个维度
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                ForEach(result.allDimensions, id: \.dimension) { dim in
                    resultCard(dim)
                }
            }

            // 纠正按钮
            Button {
                initializeCorrectionSelections(from: result)
                showCorrectionUI.toggle()
            } label: {
                HStack {
                    Image(systemName: showCorrectionUI ? "xmark.circle" : "pencil.circle")
                    Text(showCorrectionUI ? "取消纠正" : "预测错误？点击纠正")
                }
                .font(Typography.caption)
                .foregroundColor(Color.Design.sunriseGold)
            }
        }
        .padding()
        .background(Color.Design.darkIndigo.opacity(0.4))
        .cornerRadius(CornerRadius.md)
    }

    private func resultCard(_ dim: DimensionResult) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(InsightLabels.dimensionDisplayName(dim.dimension))
                .font(.caption)
                .foregroundColor(Color.Design.mutedGray)

            Text(dim.label)
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(colorForDimension(dim.dimension))

            Text("\(Int(dim.confidence * 100))%")
                .font(.caption2)
                .foregroundColor(colorForConfidence(dim.confidence))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(colorForDimension(dim.dimension).opacity(0.1))
        .cornerRadius(10)
    }

    // MARK: - Correction Section

    private var correctionSection: some View {
        VStack(alignment: .leading, spacing: Spacing.md) {
            Label("提供正确标签", systemImage: "tag.fill")
                .font(Typography.headline)
                .foregroundColor(.orange)

            // 7 个维度的选择器
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: Spacing.sm) {
                pickerView(title: "主题", selection: $selectedTopicIndex, labels: topicLabels)
                pickerView(title: "情感", selection: $selectedSentimentIndex, labels: sentimentLabels)
                pickerView(title: "紧急度", selection: $selectedUrgencyIndex, labels: urgencyLabels)
                pickerView(title: "时间范围", selection: $selectedTimeframeIndex, labels: timeframeLabels)
                pickerView(title: "行动类型", selection: $selectedActiontypeIndex, labels: actiontypeLabels)
                pickerView(title: "难度", selection: $selectedDifficultyIndex, labels: difficultyLabels)
                pickerView(title: "具体程度", selection: $selectedSpecificityIndex, labels: specificityLabels)
            }

            Button(action: addTrainingSample) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("添加到训练集")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.orange)
                .foregroundColor(.white)
                .cornerRadius(CornerRadius.md)
            }
            .disabled(lastEmbedding == nil)
        }
        .padding()
        .background(Color.orange.opacity(0.1))
        .cornerRadius(CornerRadius.md)
    }

    private func pickerView(title: String, selection: Binding<Int>, labels: [String]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(Color.Design.mutedGray)

            Menu {
                ForEach(0 ..< labels.count, id: \.self) { index in
                    Button(labels[index]) {
                        selection.wrappedValue = index
                    }
                }
            } label: {
                HStack {
                    Text(labels[selection.wrappedValue])
                        .font(.system(size: 13))
                        .foregroundColor(Color.Design.softWhite)
                    Spacer()
                    Image(systemName: "chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(Color.Design.mutedGray)
                }
                .padding(8)
                .background(Color.Design.darkIndigo.opacity(0.6))
                .cornerRadius(8)
            }
        }
    }

    // MARK: - Training Section

    private var trainingSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                Label("设备端训练", systemImage: "cpu.fill")
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.softWhite)

                Spacer()

                Text("\(updateManager.pendingSampleCount) 个样本")
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.mutedGray)
            }

            if updateManager.isTraining {
                VStack(spacing: 8) {
                    ProgressView(value: Double(updateManager.trainingProgress))
                        .tint(Color.Design.sunriseGold)

                    Text(updateManager.trainingMessage)
                        .font(.caption)
                        .foregroundColor(Color.Design.mutedGray)
                }
            }

            HStack(spacing: Spacing.sm) {
                Button(action: trainModels) {
                    HStack {
                        Image(systemName: "brain")
                        Text("训练 7 个模型")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        updateManager.pendingSampleCount > 0 && !updateManager.isTraining
                            ? Color.orange
                            : Color.gray
                    )
                    .foregroundColor(.white)
                    .cornerRadius(CornerRadius.md)
                }
                .disabled(updateManager.pendingSampleCount == 0 || updateManager.isTraining)

                Button {
                    updateManager.clearSamples()
                } label: {
                    Image(systemName: "trash")
                        .padding()
                        .background(Color.Design.darkIndigo.opacity(0.6))
                        .foregroundColor(Color.Design.mutedGray)
                        .cornerRadius(CornerRadius.md)
                }
                .disabled(updateManager.pendingSampleCount == 0)
            }

            if let lastDate = updateManager.lastTrainingDate {
                Text("上次训练: \(lastDate.formatted(date: .abbreviated, time: .shortened))")
                    .font(.system(size: 10))
                    .foregroundColor(Color.Design.mutedGray)
            }
        }
        .padding()
        .background(Color.Design.darkIndigo.opacity(0.4))
        .cornerRadius(CornerRadius.md)
    }

    // MARK: - Error Card

    private func errorCard(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            Text(message)
                .font(Typography.caption)
                .foregroundColor(.red)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(CornerRadius.md)
    }

    // MARK: - Model Info Section

    private var modelInfoSection: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            Label("模型状态", systemImage: "info.circle.fill")
                .font(Typography.caption)
                .foregroundColor(Color.Design.softWhite)

            let versions = modelManager.getModelVersions()

            VStack(spacing: Spacing.xs) {
                ForEach(Array(versions.sorted(by: { $0.key < $1.key })), id: \.key) { key, value in
                    HStack {
                        Image(systemName: InsightLabels.dimensionIcon(key))
                            .foregroundColor(colorForDimension(key))
                            .frame(width: 20)

                        Text(InsightLabels.dimensionDisplayName(key))
                            .font(.system(size: 11))
                            .foregroundColor(Color.Design.mutedGray)

                        Spacer()

                        statusBadge(value)
                    }
                }
            }
            .padding(Spacing.sm)
            .background(Color.Design.darkIndigo.opacity(0.4))
            .cornerRadius(CornerRadius.sm)

            Button(action: resetModels) {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                    Text("重置所有模型")
                }
                .font(Typography.caption)
                .foregroundColor(Color.Design.mutedGray)
            }
        }
    }

    private func statusBadge(_ status: String) -> some View {
        let color: Color = {
            switch status {
            case "loaded", "bundled": return .green
            case "updated": return .blue
            case "not_loaded": return .red
            default: return .gray
            }
        }()

        let text: String = {
            switch status {
            case "loaded": return "已加载"
            case "bundled": return "初始版本"
            case "updated": return "已更新"
            case "not_loaded": return "未加载"
            default: return status
            }
        }()

        return Text(text)
            .font(.system(size: 10, weight: .medium))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.2))
            .foregroundColor(color)
            .cornerRadius(4)
    }

    // MARK: - Helper Methods

    private func colorForDimension(_ dimension: String) -> Color {
        switch dimension {
        case "topic": return .cyan
        case "sentiment": return .pink
        case "urgency": return .red
        case "timeFrame": return .blue
        case "actionType": return .purple
        case "difficulty": return .orange
        case "specificity": return .green
        case "featureExtractor": return .indigo
        default: return .gray
        }
    }

    private func colorForConfidence(_ confidence: Double) -> Color {
        if confidence > 0.7 { return .green }
        if confidence > 0.4 { return .yellow }
        return .red
    }

    private func initializeCorrectionSelections(from result: InsightAnalysisResult) {
        // 根据预测结果初始化选择
        if let topic = result.topic {
            selectedTopicIndex = topicLabels.firstIndex(of: topic.label) ?? 0
        }
        if let sentiment = result.sentiment {
            selectedSentimentIndex = sentimentLabels.firstIndex(of: sentiment.label) ?? 0
        }
        if let urgency = result.urgency {
            selectedUrgencyIndex = urgencyLabels.firstIndex(of: urgency.label) ?? 0
        }
        if let timeframe = result.timeFrame {
            selectedTimeframeIndex = timeframeLabels.firstIndex(of: timeframe.label) ?? 0
        }
        if let actiontype = result.actionType {
            selectedActiontypeIndex = actiontypeLabels.firstIndex(of: actiontype.label) ?? 0
        }
        if let difficulty = result.difficulty {
            selectedDifficultyIndex = difficultyLabels.firstIndex(of: difficulty.label) ?? 0
        }
        if let specificity = result.specificity {
            selectedSpecificityIndex = specificityLabels.firstIndex(of: specificity.label) ?? 0
        }
    }

    // MARK: - Actions

    private func runAnalysis() {
        guard !inputText.isEmpty, modelManager.isInitialized else { return }

        isAnalyzing = true
        errorMessage = nil

        Task {
            do {
                // 获取嵌入向量用于后续训练
                lastEmbedding = try modelManager.getEmbeddingForTraining(text: inputText)

                let result = try await modelManager.analyze(text: inputText)

                await MainActor.run {
                    self.analysisResult = result
                    self.isAnalyzing = false
                    self.showCorrectionUI = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "分析失败: \(error.localizedDescription)"
                    self.isAnalyzing = false
                }
            }
        }
    }

    private func addTrainingSample() {
        guard let embedding = lastEmbedding else { return }

        updateManager.addTrainingSample(
            embedding: embedding,
            text: inputText,
            topicLabel: selectedTopicIndex,
            sentimentLabel: selectedSentimentIndex,
            urgencyLabel: selectedUrgencyIndex,
            timeframeLabel: selectedTimeframeIndex,
            actiontypeLabel: selectedActiontypeIndex,
            difficultyLabel: selectedDifficultyIndex,
            specificityLabel: selectedSpecificityIndex
        )

        showCorrectionUI = false

        // 显示成功提示
        errorMessage = nil
    }

    private func trainModels() {
        Task {
            do {
                try await updateManager.updateAllModels()
                // 重新加载更新后的分类器（不是重置）
                await modelManager.reloadModelsAfterTraining()
            } catch {
                errorMessage = "训练失败: \(error.localizedDescription)"
            }
        }
    }

    private func resetModels() {
        do {
            try modelManager.resetAllClassifiers()
        } catch {
            errorMessage = "重置失败: \(error.localizedDescription)"
        }
    }
}
