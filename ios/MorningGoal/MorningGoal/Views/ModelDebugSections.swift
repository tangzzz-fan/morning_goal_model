import CoreData
import SwiftUI

struct ModelInfoSectionView: View {
    let modelInfo: String
    let correctionCount: Int
    let refresh: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "cpu.fill")
                    .foregroundColor(.blue)
                Text("模型信息")
                    .font(.headline)
                Spacer()
                Button(action: refresh) {
                    Image(systemName: "arrow.clockwise")
                        .foregroundColor(.blue)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                InfoRow(label: "模型状态", value: modelInfo.isEmpty ? "加载中..." : "已加载")
                InfoRow(label: "待训练样本", value: "\(correctionCount) 个")
                InfoRow(label: "训练阈值", value: "10 个样本")
                InfoRow(label: "训练间隔", value: "24 小时")
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(12)
        }
    }
}

struct PredictionTestSectionView: View {
    @Binding var inputText: String
    let isAnalyzing: Bool
    let errorMessage: String?
    let runPrediction: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "wand.and.stars")
                    .foregroundColor(.purple)
                Text("预测测试")
                    .font(.headline)
            }

            VStack(spacing: 12) {
                TextEditor(text: $inputText)
                    .frame(height: 100)
                    .padding(8)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    )

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        QuickFillButton(text: "今天要完成项目开发工作", action: { inputText = "今天要完成项目开发工作" })
                        QuickFillButton(text: "晚上去健身房锻炼", action: { inputText = "晚上去健身房锻炼" })
                        QuickFillButton(text: "和家人一起吃饭", action: { inputText = "和家人一起吃饭" })
                    }
                }

                Button(action: runPrediction, label: {
                    HStack {
                        if isAnalyzing {
                            ProgressView()
                                .progressViewStyle(.circular)
                                .tint(.white)
                        } else {
                            Image(systemName: "play.circle.fill")
                        }
                        Text(isAnalyzing ? "分析中..." : "运行预测")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(inputText.isEmpty ? Color.gray : Color.purple)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                })
                .disabled(inputText.isEmpty || isAnalyzing)
            }

            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.top, 4)
            }
        }
    }
}

struct PredictionResultSectionView: View {
    let result: GoalAnalysisResult
    let categories: [String]
    let sentiments: [String]
    @Binding var selectedCategories: Set<String>
    @Binding var correctedSentiment: String
    let saveCorrection: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(.green)
                Text("预测结果")
                    .font(.headline)
            }

            VStack(spacing: 12) {
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
                    color: .orange
                )

                if let topCategories = result.topCategories, topCategories.count > 1 {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Image(systemName: "chart.bar.doc.horizontal")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("Top-\(topCategories.count) 分类概率")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        ForEach(Array(topCategories.enumerated()), id: \ .offset) { index, item in
                            HStack(spacing: 8) {
                                Text("\(index + 1).")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .frame(width: 20, alignment: .leading)

                                Text(item.name)
                                    .font(.caption)
                                    .fontWeight(index == 0 ? .semibold : .regular)

                                Spacer()

                                Text("\(Int(item.confidence * 100))%")
                                    .font(.caption)
                                    .foregroundColor(index == 0 ? .blue : .secondary)

                                GeometryReader { geometry in
                                    ZStack(alignment: .leading) {
                                        Rectangle()
                                            .fill(Color.gray.opacity(0.2))
                                            .frame(height: 3)

                                        Rectangle()
                                            .fill(index == 0 ? Color.blue : Color.gray)
                                            .frame(width: geometry.size.width * item.confidence, height: 3)
                                    }
                                }
                                .frame(width: 60, height: 3)
                            }
                        }
                    }
                    .padding()
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(8)
                }

                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "hand.tap.fill")
                            .font(.caption)
                            .foregroundColor(.orange)
                        Text("如果预测不准确,可以纠正:")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("正确分类")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("(可多选)")
                                .font(.caption2)
                                .foregroundColor(.orange)
                            Spacer()
                            if !selectedCategories.isEmpty {
                                Button("清除") {
                                    selectedCategories.removeAll()
                                }
                                .font(.caption2)
                                .foregroundColor(.red)
                            }
                        }

                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 70))], spacing: 8) {
                            ForEach(categories, id: \ .self) { category in
                                CategoryChip(
                                    category: category,
                                    isSelected: selectedCategories.contains(category),
                                    action: {
                                        if selectedCategories.contains(category) {
                                            selectedCategories.remove(category)
                                        } else {
                                            selectedCategories.insert(category)
                                        }
                                    }
                                )
                            }
                        }
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        Text("正确情感")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        HStack(spacing: 8) {
                            ForEach(sentiments, id: \ .self) { sentiment in
                                Button(action: { correctedSentiment = sentiment }, label: {
                                    Text(sentiment)
                                        .font(.caption)
                                        .padding(.horizontal, 16)
                                        .padding(.vertical, 8)
                                        .background(correctedSentiment == sentiment ? Color.orange : Color.gray.opacity(0.2))
                                        .foregroundColor(correctedSentiment == sentiment ? .white : .primary)
                                        .cornerRadius(8)
                                })
                            }
                        }
                    }

                    Button(action: saveCorrection, label: {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                            Text("保存纠正")
                            if !selectedCategories.isEmpty {
                                Text("(\(selectedCategories.count) 个分类)")
                                    .font(.caption)
                            }
                        }
                        .font(.subheadline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(selectedCategories.isEmpty ? Color.gray : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    })
                    .disabled(selectedCategories.isEmpty)
                }
                .padding()
                .background(Color.yellow.opacity(0.1))
                .cornerRadius(8)
            }
        }
    }
}

struct ExistingGoalsSectionView: View {
    @Binding var showExistingGoals: Bool
    let existingGoals: FetchedResults<GoalEntry>
    @Binding var inputText: String
    let isBatchAnalyzing: Bool
    let batchProgress: Double
    let analyzedCount: Int
    let batchAnalyzeGoals: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "list.bullet.clipboard")
                    .foregroundColor(.purple)
                Text("已有目标数据")
                    .font(.headline)
                Spacer()
                Button(action: { showExistingGoals.toggle() }, label: {
                    Image(systemName: showExistingGoals ? "chevron.up" : "chevron.down")
                        .foregroundColor(.gray)
                })
            }

            if showExistingGoals {
                VStack(spacing: 12) {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("总目标数")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(existingGoals.count)")
                                .font(.title2)
                                .fontWeight(.bold)
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 4) {
                            Text("已分析")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("\(analyzedCount)")
                                .font(.title2)
                                .fontWeight(.bold)
                                .foregroundColor(.green)
                        }
                    }
                    .padding()
                    .background(Color.purple.opacity(0.1))
                    .cornerRadius(8)

                    Button(action: batchAnalyzeGoals, label: {
                        HStack {
                            if isBatchAnalyzing {
                                ProgressView()
                                    .progressViewStyle(.circular)
                                    .tint(.white)
                            } else {
                                Image(systemName: "wand.and.stars")
                            }
                            Text(isBatchAnalyzing ? "分析中... \(Int(batchProgress * 100))%" : "批量分析所有目标")
                        }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isBatchAnalyzing ? Color.gray : Color.purple)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    })
                    .disabled(isBatchAnalyzing || existingGoals.isEmpty)

                    if isBatchAnalyzing {
                        ProgressView(value: batchProgress)
                            .progressViewStyle(.linear)
                    }

                    if !existingGoals.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("最近的目标 (点击填充到测试框)")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            ForEach(existingGoals.prefix(5)) { goal in
                                Button(action: {
                                    inputText = goal.goalText
                                }, label: {
                                    HStack(spacing: 8) {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(goal.goalText)
                                                .font(.caption)
                                                .lineLimit(2)
                                                .foregroundColor(.primary)

                                            HStack(spacing: 8) {
                                                Text(goal.dateString)
                                                    .font(.caption2)
                                                    .foregroundColor(.secondary)

                                                if let category = goal.category {
                                                    Text(category)
                                                        .font(.caption2)
                                                        .padding(.horizontal, 6)
                                                        .padding(.vertical, 2)
                                                        .background(Color.blue.opacity(0.2))
                                                        .cornerRadius(4)
                                                }
                                            }
                                        }

                                        Spacer()

                                        Image(systemName: "arrow.right.circle")
                                            .foregroundColor(.purple)
                                    }
                                    .padding(8)
                                    .background(Color.gray.opacity(0.1))
                                    .cornerRadius(8)
                                })
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
            }
        }
    }
}

struct TrainingControlSectionView: View {
    @Binding var showTrainingSection: Bool
    let correctionCount: Int
    let isTraining: Bool
    let trainingCount: Int
    let trainingLog: [String]
    let triggerTraining: () -> Void
    let clearTrainingLog: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(.red)
                Text("训练控制")
                    .font(.headline)
                Spacer()
                Button(action: { showTrainingSection.toggle() }, label: {
                    Image(systemName: showTrainingSection ? "chevron.up" : "chevron.down")
                        .foregroundColor(.gray)
                })
            }

            if showTrainingSection {
                VStack(spacing: 12) {
                    Text("当前有 \(correctionCount) 个待训练样本")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    if correctionCount < 10 {
                        Text("⚠️ 样本数量不足 10 个,建议先积累更多样本")
                            .font(.caption)
                            .foregroundColor(.orange)
                            .padding(.vertical, 4)
                    }

                    Button(action: triggerTraining) {
                        HStack {
                            if isTraining {
                                ProgressView()
                                    .progressViewStyle(.circular)
                                    .tint(.white)
                            } else {
                                Image(systemName: "bolt.fill")
                            }
                            Text(isTraining ? "训练中..." : "立即训练")
                        }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(isTraining ? Color.gray : Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(isTraining || correctionCount == 0)

                    if trainingCount > 0 {
                        Text("已完成 \(trainingCount) 次训练")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    if !trainingLog.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("训练日志")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Button("清除") {
                                    clearTrainingLog()
                                }
                                .font(.caption2)
                                .foregroundColor(.red)
                            }

                            ScrollView {
                                VStack(alignment: .leading, spacing: 4) {
                                    ForEach(trainingLog, id: \ .self) { log in
                                        Text(log)
                                            .font(.system(.caption2, design: .monospaced))
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(8)
                                .background(Color.black.opacity(0.05))
                                .cornerRadius(6)
                            }
                            .frame(maxHeight: 150)
                        }
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(12)
            }
        }
    }
}
