import CoreData
import SwiftUI

struct TodayInputView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var text: String = ""
    @FocusState private var focused: Bool
    @State private var streakTrigger = false
    @State private var saveStatus: SaveStatus = .idle
    @State private var onThisDayEntry: GoalEntry?
    @State private var showOnThisDay = false
    @State private var selectedEntryForCard: GoalEntry?
    @State private var titleText: String = ""

    // Completion callback for dismissal
    let onComplete: () -> Void

    enum SaveStatus {
        case idle, saving, saved
    }

    private func load() {
        let req = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        req.predicate = NSPredicate(format: "dateString == %@", GoalEntry.todayString())
        req.fetchLimit = 1
        if let entry = try? context.fetch(req).first {
            text = entry.goalText
        }
    }

    /// 用户点击键盘确认按钮时保存
    private func saveGoal() {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }

        saveStatus = .saving

        let req = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        req.predicate = NSPredicate(format: "dateString == %@", GoalEntry.todayString())
        req.fetchLimit = 1

        // 保存或更新条目
        let entry = (try? context.fetch(req).first) ?? GoalEntry(context: context)
        entry.dateString = GoalEntry.todayString()
        entry.goalText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        entry.lastUpdated = Date()

        do {
            try context.save()
            saveStatus = .saved
            streakTrigger.toggle()
            NotificationService.shared.clearBadgeAndCancelToday()

            // 提供触觉反馈
            UINotificationFeedbackGenerator().notificationOccurred(.success)

            // 保存后收起键盘
            focused = false

            // 延迟后调用完成回调以关闭视图
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                onComplete()
            }

            // 偶发展示“历史上的今天”卡片
            if Int.random(in: 0 ..< 10) < 3 { // 约30% 触发概率
                loadOnThisDayEntries()
            }

            // 2秒后重置状态指示器
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if saveStatus == .saved {
                    saveStatus = .idle
                }
            }
        } catch {
            saveStatus = .idle
        }
    }

    /// 加载历史上的今天数据
    private func loadOnThisDayEntries() {
        let entries = MockDataService.fetchOnThisDayEntries(in: context)

        print("📅 加载历史上的今天数据：共 \(entries.count) 条")

        // 随机选择一条展示
        if let randomEntry = entries.randomElement() {
            print("  ✅ 随机选中: \(randomEntry.dateString) - \(randomEntry.goalText)")
            onThisDayEntry = randomEntry
            // 延迟展示，添加动画效果
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
                    showOnThisDay = true
                }
            }
        }
    }

    /// 随机选择标题文案
    private func randomizeTitleText() {
        let titleKeys = [
            "today_goal_title_1",
            "today_goal_title_2",
            "today_goal_title_3",
            "today_goal_title_4",
            "today_goal_title_5",
            "today_goal_title_6"
        ]
        let randomKey = titleKeys.randomElement() ?? "today_goal_title_1"
        titleText = NSLocalizedString(randomKey, comment: "")
    }

    private var currentDateText: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy年MM月dd日"
        return formatter.string(from: Date())
    }

    var body: some View {
        ZStack {
            // 背景层 - 用于捕获点击手势以释放键盘焦点
            Color.Design.deepIndigo
                .ignoresSafeArea()
                .onTapGesture {
                    focused = false
                }

            VStack(alignment: .leading, spacing: Spacing.lg) {
                // 日期显示
                Text(currentDateText)
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.mutedGray)
                    .padding(.top, Spacing.xl)

                // 标题 - 随机文案
                Text(titleText)
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.softWhite)
                    .onTapGesture(count: 2) {
                        // 调试机制：双击标题强制加载历史上的今天
                        loadOnThisDayEntries()
                        let generator = UIImpactFeedbackGenerator(style: .medium)
                        generator.impactOccurred()
                    }

                // 文本输入区域 - 单行输入，按回车键保存
                TextField(LocalizedStringKey("today_goal_placeholder"), text: $text)
                    .font(Typography.body)
                    .foregroundColor(Color.Design.softWhite)
                    .focused($focused)
                    .lineLimit(1)
                    .textFieldStyle(.plain)
                    .padding(Spacing.md)
                    .background(
                        RoundedRectangle(cornerRadius: CornerRadius.md)
                            .fill(Color.Design.darkIndigo.opacity(0.5))
                            .overlay(
                                RoundedRectangle(cornerRadius: CornerRadius.md)
                                    .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                            )
                    )
                    .submitLabel(.done)
                    .onSubmit {
                        saveGoal()
                    }

                // 保存状态指示器和字数统计
                HStack {
                    // 保存状态指示
                    if saveStatus == .saved {
                        HStack(spacing: 4) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundColor(Color.Design.sunriseGold)

                            Text(LocalizedStringKey("today_goal_saved"))
                                .font(Typography.caption)
                                .foregroundColor(Color.Design.sunriseGold)
                        }
                        .transition(.scale.combined(with: .opacity))
                    } else if saveStatus == .saving {
                        HStack(spacing: 4) {
                            ProgressView()
                                .scaleEffect(0.7)
                                .tint(Color.Design.mutedGray)

                            Text(LocalizedStringKey("today_goal_saving"))
                                .font(Typography.caption)
                                .foregroundColor(Color.Design.mutedGray)
                        }
                        .transition(.scale.combined(with: .opacity))
                    }

                    Spacer()

                    Text("\(text.count)/200")
                        .font(Typography.caption)
                        .foregroundColor(text.count > 200 ? .red : Color.Design.mutedGray)
                }
                .frame(height: 20)
                .animation(.easeInOut(duration: 0.3), value: saveStatus)

                Spacer()

                // 历史上的今天小卡片 - 在提示信息上方
                if showOnThisDay, let entry = onThisDayEntry {
                    OnThisDayMiniCard(entry: entry)
                        .onTapGesture {
                            selectedEntryForCard = entry
                        }
                        .transition(.asymmetric(
                            insertion: .move(edge: .bottom).combined(with: .opacity),
                            removal: .move(edge: .bottom).combined(with: .opacity)
                        ))
                        .padding(.horizontal, Spacing.xl)
                        .padding(.bottom, Spacing.sm)
                }

                // 提示文本
                VStack(spacing: Spacing.xs) {
                    Text(LocalizedStringKey("today_value_proposition"))
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray)

                    Text(LocalizedStringKey("today_input_tip"))
                        .font(.system(size: 11))
                        .foregroundColor(Color.Design.mutedGray.opacity(0.7))
                }
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)
                .padding(.bottom, Spacing.md)
            }
            .padding(.horizontal, Spacing.md)
        }
        .fullScreenCover(item: $selectedEntryForCard) { entry in
            OnThisDayCardView(entry: entry)
        }
        .onAppear {
            load()
            randomizeTitleText()

            // 延迟键盘聚焦，等待启动屏幕动画完成
            // 启动屏幕显示时间(1.5s) + 淡出动画(0.5s) + 缓冲(0.1s) = 2.1s
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.1) {
                focused = true
            }
        }
    }
}

// MARK: - 历史上的今天迷你卡片

struct OnThisDayMiniCard: View {
    let entry: GoalEntry
    @State private var appeared = false

    var timeLabel: String {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let entryDate = calendar.startOfDay(for: entry.lastUpdated)

        let components = calendar.dateComponents([.year, .month, .day], from: entryDate, to: today)

        if let year = components.year, year >= 1 {
            return String(format: NSLocalizedString("on_this_day_title_years", comment: ""), year)
        } else if let month = components.month, month >= 1 {
            return NSLocalizedString("on_this_day_title_month", comment: "")
        } else if let day = components.day, day >= 7 {
            return NSLocalizedString("on_this_day_title_week", comment: "")
        }

        return String(format: NSLocalizedString("on_this_day_title_years", comment: ""), 1) // Fallback
    }

    var body: some View {
        HStack(spacing: Spacing.md) {
            // 左侧图标
            Image(systemName: "sparkles")
                .font(.system(size: 16))
                .foregroundColor(Color.Design.sunriseGold)
                .rotationEffect(.degrees(appeared ? 0 : -180))

            // 内容
            VStack(alignment: .leading, spacing: 4) {
                Text(timeLabel)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(Color.Design.sunriseGold.opacity(0.9))

                Text(entry.goalText)
                    .font(.system(size: 13))
                    .foregroundColor(Color.Design.softWhite.opacity(0.8))
                    .lineLimit(2)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: CornerRadius.md)
                .fill(Color.Design.darkIndigo.opacity(0.4))
                .overlay(
                    RoundedRectangle(cornerRadius: CornerRadius.md)
                        .stroke(Color.Design.sunriseGold.opacity(0.2), lineWidth: 1)
                )
        )
        .scaleEffect(appeared ? 1 : 0.95)
        .opacity(appeared ? 1 : 0)
        .onAppear {
            withAnimation(.spring(response: 0.5, dampingFraction: 0.7)) {
                appeared = true
            }
        }
    }
}
