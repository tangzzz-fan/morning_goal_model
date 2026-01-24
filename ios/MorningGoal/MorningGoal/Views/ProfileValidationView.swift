import CoreML
import SwiftUI

struct ProfileValidationView: View {
    @State private var inputText: String = ""
    @State private var analysisResult: AnalysisResult?
    @State private var inferenceTime: Double = 0
    @State private var isAnalyzing: Bool = false
    @State private var errorMessage: String?

    // Service instance (lazy loaded) - using the new updatable model architecture
    @State private var service: UpdatableCoreMLGoalAnalysisService?
    @State private var modelVersion: String = "加载中..."

    // Correction state
    @State private var showCorrection: Bool = false
    @State private var selectedCorrectCategory: String = "工作"
    @State private var selectedCorrectSentiment: String = "中性"
    @State private var isUpdating: Bool = false
    @State private var updateSuccess: Bool = false

    // Preset test cases
    let presets = [
        ("工作", "今天完成了所有代码审查，效率很高"),
        ("健康", "晚上去健身房跑步30分钟"),
        ("家庭", "周末陪孩子去公园玩"),
        ("学习", "阅读《Swift进阶》两章"),
        ("财务", "整理上个月的账单支出"),
        ("长文本", "今天虽然很忙，但是完成了很多重要的事情，包括项目汇报和团队同步，感觉非常有成就感，希望明天继续保持。"),
        ("负面", "今天什么都没做成，感觉很糟糕")
    ]

    var body: some View {
        NavigationStack {
            ZStack {
                Color.Design.deepIndigo.ignoresSafeArea()

                ScrollView {
                    VStack(spacing: Spacing.lg) {
                        // Header
                        VStack(spacing: Spacing.xs) {
                            Text("模型验证")
                                .font(Typography.headline)
                                .foregroundColor(Color.Design.sunriseGold)
                            Text("Model: FeatureExtractor + k-NN (Updatable)")
                                .font(Typography.caption)
                                .foregroundColor(Color.Design.mutedGray)
                        }
                        .padding(.top, Spacing.md)

                        // Input Section
                        VStack(alignment: .leading, spacing: Spacing.sm) {
                            Text("输入测试")
                                .font(Typography.caption)
                                .foregroundColor(Color.Design.softWhite)

                            TextField("输入目标文本...", text: $inputText)
                                .padding()
                                .background(Color.Design.darkIndigo)
                                .cornerRadius(CornerRadius.md)
                                .foregroundColor(.white)
                                .overlay(
                                    RoundedRectangle(cornerRadius: CornerRadius.md)
                                        .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                                )
                                .submitLabel(.go)
                                .onSubmit {
                                    runAnalysis(text: inputText)
                                }

                            Button(action: { runAnalysis(text: inputText) }) {
                                if isAnalyzing {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Text("运行分析")
                                        .font(Typography.body)
                                        .fontWeight(.medium)
                                }
                            }
                            .buttonStyle(SunriseGoldButtonStyle())
                            .disabled(inputText.isEmpty || isAnalyzing)
                        }
                        .padding(.horizontal)

                        // Presets
                        VStack(alignment: .leading, spacing: Spacing.sm) {
                            Text("快速测试")
                                .font(Typography.caption)
                                .foregroundColor(Color.Design.softWhite)

                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: Spacing.sm) {
                                ForEach(presets, id: \.0) { category, text in
                                    Button(category) {
                                        inputText = text
                                        runAnalysis(text: text)
                                    }
                                    .font(Typography.caption)
                                    .padding(.vertical, 8)
                                    .padding(.horizontal, 12)
                                    .background(Color.Design.darkIndigo)
                                    .cornerRadius(CornerRadius.sm)
                                    .foregroundColor(Color.Design.softWhite)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: CornerRadius.sm)
                                            .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                                    )
                                }
                            }
                        }
                        .padding(.horizontal)

                        // Results and Correction
                        if let result = analysisResult {
                            VStack(alignment: .leading, spacing: Spacing.md) {
                                Text("分析结果")
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.softWhite)

                                HStack(spacing: Spacing.md) {
                                    ResultCard(
                                        title: "分类",
                                        value: result.category,
                                        confidence: result.categoryConfidence,
                                        color: .blue
                                    )

                                    ResultCard(
                                        title: "情感",
                                        value: result.sentiment,
                                        confidence: result.sentimentScore,
                                        color: result.sentiment == "积极" ? .green : (result.sentiment == "消极" ? .red : .gray)
                                    )
                                }

                                HStack {
                                    Text("⏱️ 推理耗时:")
                                        .foregroundColor(Color.Design.mutedGray)
                                    Text("\(String(format: "%.2f", inferenceTime)) ms")
                                        .foregroundColor(Color.Design.sunriseGold)
                                        .bold()
                                    Spacer()
                                    Button(action: { showCorrection.toggle() }) {
                                        Label(showCorrection ? "取消纠错" : "对此分类有异议？", systemImage: "pencil.and.outline")
                                            .font(Typography.caption)
                                            .foregroundColor(Color.Design.sunriseGold)
                                    }
                                }
                                .font(Typography.caption)

                                // Correction UI
                                if showCorrection {
                                    VStack(alignment: .leading, spacing: Spacing.sm) {
                                        Divider().background(Color.Design.sunriseGold.opacity(0.3))

                                        Text("模型纠错 & 端侧训练")
                                            .font(Typography.caption)
                                            .foregroundColor(Color.Design.sunriseGold)

                                        Text("选择正确类别并更新模型：")
                                            .font(.system(size: 12))
                                            .foregroundColor(Color.Design.mutedGray)

                                        Picker("正确类别", selection: $selectedCorrectCategory) {
                                            if let categories = service?.categories {
                                                ForEach(categories, id: \.self) { cat in
                                                    Text(cat).tag(cat)
                                                }
                                            }
                                        }
                                        .pickerStyle(.menu)
                                        .tint(Color.Design.sunriseGold)

                                        Picker("正确情感", selection: $selectedCorrectSentiment) {
                                            if let sentiments = service?.sentiments {
                                                ForEach(sentiments, id: \.self) { sen in
                                                    Text(sen).tag(sen)
                                                }
                                            }
                                        }
                                        .pickerStyle(.menu)
                                        .tint(Color.Design.sunriseGold)
                                        .padding(.bottom, 4)

                                        Button(action: { runUpdate() }) {
                                            HStack {
                                                if isUpdating {
                                                    ProgressView().tint(.white)
                                                    Text("训练中...")
                                                } else {
                                                    Image(systemName: "arrow.up.circle.fill")
                                                    Text("更新模型并训练")
                                                }
                                            }
                                            .frame(maxWidth: .infinity)
                                        }
                                        .buttonStyle(SunriseGoldButtonStyle())
                                        .disabled(isUpdating)

                                        if updateSuccess {
                                            Text("✅ 模型已更新！再次运行分析以验证效果。")
                                                .font(.system(size: 12))
                                                .foregroundColor(.green)
                                        }
                                    }
                                    .padding(.top, 8)
                                }
                            }
                            .padding()
                            .background(Color.Design.darkIndigo.opacity(0.5))
                            .cornerRadius(CornerRadius.md)
                            .padding(.horizontal)
                            .transition(.scale.combined(with: .opacity))
                        }

                        if let error = errorMessage {
                            Text(error)
                                .foregroundColor(.red)
                                .font(Typography.caption)
                                .padding()
                        }

                        // Model Management Footer
                        VStack(spacing: Spacing.sm) {
                            Divider().background(Color.Design.mutedGray.opacity(0.3))
                            HStack {
                                Text("当前权重版本:")
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.mutedGray)
                                Text(modelVersion)
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.sunriseGold)
                                    .monospaced()
                                Spacer()
                                Button("重置模型") {
                                    try? service?.resetModel()
                                    updateModelStatus()
                                }
                                .font(Typography.caption)
                                .foregroundColor(.red.opacity(0.8))
                            }
                            .padding(.top, 4)
                        }
                        .padding()
                        .padding(.bottom, Spacing.xl)

                        Spacer()
                    }
                }
            }
            .navigationBarHidden(true)
            .onAppear {
                initializeService()
            }
        }
    }

    private func initializeService() {
        if service == nil {
            do {
                service = try UpdatableCoreMLGoalAnalysisService()
                updateModelStatus()
            } catch {
                errorMessage = "模型加载失败: \(error.localizedDescription)"
            }
        }
    }

    private func updateModelStatus() {
        guard let service = service else { return }
        modelVersion = service.getModelVersion()
        if let firstCat = service.categories.first {
            selectedCorrectCategory = firstCat
        }
    }

    private func runAnalysis(text: String) {
        guard !text.isEmpty, let service = service else { return }

        isAnalyzing = true
        errorMessage = nil
        updateSuccess = false
        // 不自动隐藏纠错，方便对比

        Task {
            do {
                let start = CFAbsoluteTimeGetCurrent()
                let result = try await service.analyzeGoal(text)
                let end = CFAbsoluteTimeGetCurrent()

                await MainActor.run {
                    self.analysisResult = result
                    self.inferenceTime = (end - start) * 1000
                    self.isAnalyzing = false
                    // 默认纠错类别设为预测的相反（如果没偏见的话）或者保持
                    if !showCorrection {
                        selectedCorrectCategory = result.category
                        selectedCorrectSentiment = result.sentiment
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "分析出错: \(error.localizedDescription)"
                    self.isAnalyzing = false
                    self.analysisResult = nil
                }
            }
        }
    }

    private func runUpdate() {
        guard let service = service, !inputText.isEmpty else { return }

        isUpdating = true
        updateSuccess = false

        let sample = TrainingSample(
            text: inputText,
            correctCategory: selectedCorrectCategory,
            correctSentiment: selectedCorrectSentiment
        )

        Task {
            do {
                try await service.updateModel(with: [sample])

                // 给一点训练感延迟
                try await Task.sleep(nanoseconds: 1_000_000_000)

                await MainActor.run {
                    isUpdating = false
                    updateSuccess = true
                    updateModelStatus()

                    // 1.5秒后自动关闭纠错界面
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        withAnimation {
                            showCorrection = false
                            updateSuccess = false
                        }
                    }
                }
            } catch {
                await MainActor.run {
                    errorMessage = "模型更新失败: \(error.localizedDescription)"
                    isUpdating = false
                }
            }
        }
    }
}

#Preview {
    ProfileValidationView()
}
