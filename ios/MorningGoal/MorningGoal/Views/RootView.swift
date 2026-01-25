import CoreData
import SwiftUI

struct RootView: View {
    @Environment(\.managedObjectContext) private var context
    @FetchRequest(
        entity: GoalEntry.entity(),
        sortDescriptors: [],
        predicate: NSPredicate(format: "dateString == %@", GoalEntry.todayString()),
        animation: .default
    )
    private var todayEntries: FetchedResults<GoalEntry>
    @FetchRequest(entity: UserSettings.entity(), sortDescriptors: [], animation: .default)
    private var settings: FetchedResults<UserSettings>
    @State private var showDebugMenu = false
    @State private var showStreakSettings = false
    @State private var streakDays: Int = 7
    @State private var showOnThisDaySettings = false
    @State private var onThisDayYears: Int = 1
    @State private var showTestOnThisDay = false
    @State private var testOnThisDayEntry: GoalEntry?

    // Sheet presentation state
    @State private var showTodayInput = false
    @State private var shouldCelebrate = false
    @State private var inputPresentationAllowed = false
    @State private var showModelTest = false
    @State private var showQualityDashboard = false
    @State private var showEnhancedQualityDashboard = false
    @State private var showModelDebug = false
    @State private var showCloudKitStatus = false

    var body: some View {
        Group {
            if settings.first?.committed != true {
                OnboardingView {
                    // Force refresh view state when onboarding completes
                }
            } else {
                TabView {
                    // Tab 1: Main (History)
                    HistoryListView(shouldCelebrate: $shouldCelebrate)
                        .fullScreenCover(isPresented: $showTodayInput) {
                            TodayInputView(onComplete: {
                                shouldCelebrate = true
                                showTodayInput = false
                            })
                            .environment(\.managedObjectContext, context)
                        }
                        .onAppear {
                            checkAndShowInput()
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2.1) {
                                inputPresentationAllowed = true
                                checkAndShowInput()
                            }
                        }
                        .onChange(of: todayEntries.count) { _, _ in
                            checkAndShowInput()
                        }
                        .tabItem {
                            Label("目标", systemImage: "list.bullet")
                        }

                    // Tab 2: Insight Stats (数据聚合统计)
                    InsightStatsTab(context: context)
                        .tabItem {
                            Label("统计", systemImage: "chart.bar.fill")
                        }

                    // Tab 3: Insight Model Debug
                    InsightModelDebugTab()
                        .tabItem {
                            Label("洞察", systemImage: "brain.head.profile")
                        }
                }
                // Apply global accent color
                .tint(Color.Design.sunriseGold)
            }
        }
        .overlay(alignment: .bottomTrailing) {
            #if DEBUG
            // 仅在非 onboarding 状态下显示调试按钮
            if settings.first?.committed == true {
                debugMenuButton
            }
            #endif
        }
        .confirmationDialog(LocalizedStringKey("debug_menu_title"), isPresented: $showDebugMenu, titleVisibility: .visible) {
            #if DEBUG

            Button(LocalizedStringKey("debug_reset_onboarding")) {
                resetOnboarding()
            }
            Button(LocalizedStringKey("debug_add_test_data")) {
                MockDataService.addMockData(to: context)
            }
            Button(LocalizedStringKey("debug_reset_today")) {
                MockDataService.resetToday(in: context)
            }
            Button(LocalizedStringKey("debug_set_streak")) {
                showStreakSettings = true
            }
            Button(LocalizedStringKey("debug_add_on_this_day")) {
                showOnThisDaySettings = true
            }
            Button("模型测试页面") {
                showModelTest = true
            }
            Button("模型质量仪表板") {
                showQualityDashboard = true
            }
            Button("增强质量分析") {
                showEnhancedQualityDashboard = true
            }
            Button("🔮 模型调试界面") {
                showModelDebug = true
            }
            Button("☁️ CloudKit 同步状态") {
                showCloudKitStatus = true
            }
            Button(LocalizedStringKey("debug_view_on_this_day")) {
                let entries = MockDataService.fetchOnThisDayEntries(in: context)
                print("📅 历史上的今天数据：\(entries.count) 条")
                for entry in entries {
                    print("  - \(entry.dateString): \(entry.goalText)")
                }
            }
            Button(LocalizedStringKey("debug_test_on_this_day_card")) {
                let entries = MockDataService.fetchOnThisDayEntries(in: context)
                if let entry = entries.randomElement() {
                    testOnThisDayEntry = entry
                } else {
                    print("❌ 没有历史数据可以测试")
                }
            }
            Button(LocalizedStringKey("debug_clear_all")) {
                MockDataService.clearAllData(from: context)
            }
            Button(LocalizedStringKey("cancel"), role: .cancel) {}
            #endif
        }
        .sheet(isPresented: $showModelTest) {
            NavigationStack { ModelTestView() }
                .environment(\.managedObjectContext, context)
        }
        .sheet(isPresented: $showQualityDashboard) {
            NavigationStack { ModelQualityDashboard() }
                .environment(\.managedObjectContext, context)
        }
        .sheet(isPresented: $showEnhancedQualityDashboard) {
            NavigationStack { EnhancedModelQualityDashboard() }
                .environment(\.managedObjectContext, context)
        }
        .sheet(isPresented: $showModelDebug) {
            ModelDebugView()
                .environment(\.managedObjectContext, context)
        }
        .sheet(isPresented: $showCloudKitStatus) {
            CloudKitSyncStatusView()
                .environment(\.managedObjectContext, context)
        }
        .fullScreenCover(item: $testOnThisDayEntry) { entry in
            OnThisDayCardView(entry: entry)
        }
    }

    private func checkAndShowInput() {
        // Automatically show input sheet if today's entry doesn't exist
        showTodayInput = (inputPresentationAllowed && todayEntries.first == nil && settings.first?.committed == true)
    }

    #if DEBUG
    private func resetOnboarding() {
        // 删除 UserSettings
        let fetchRequest = NSFetchRequest<UserSettings>(entityName: "UserSettings")
        if let settings = try? context.fetch(fetchRequest) {
            settings.forEach { context.delete($0) }
        }

        // 保存更改
        do {
            try context.save()
            print("✅ Onboarding已重置")
        } catch {
            print("❌ 重置Onboarding失败: \(error)")
        }
    }

    private var debugMenuButton: some View {
        Button(action: { showDebugMenu = true }, label: {
            Image(systemName: "gearshape.fill")
                .font(.system(size: 20))
                .foregroundColor(Color.Design.sunriseGold)
                .padding(12)
                .background(
                    Circle()
                        .fill(Color.Design.deepIndigo.opacity(0.8))
                        .overlay(
                            Circle()
                                .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                        )
                )
        })
        .padding(.trailing, Spacing.md)
        .padding(.bottom, Spacing.md)
        .transition(.scale.combined(with: .opacity))
    }
    #endif
}

#if DEBUG
struct StreakSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var streakDays: Int
    let onConfirm: (Int) -> Void

    var body: some View {
        NavigationStack {
            ZStack {
                Color.Design.deepIndigo.ignoresSafeArea()

                VStack(spacing: Spacing.lg) {
                    Text(LocalizedStringKey("streak_settings_header"))
                        .font(Typography.headline)
                        .foregroundColor(Color.Design.softWhite)
                        .padding(.top, Spacing.xl)

                    VStack(spacing: Spacing.md) {
                        Text(String(format: NSLocalizedString("streak_settings_current", comment: ""), streakDays))
                            .font(Typography.title)
                            .foregroundColor(Color.Design.sunriseGold)

                        // 滑块
                        VStack(spacing: Spacing.sm) {
                            Slider(value: Binding(
                                get: { Double(streakDays) },
                                set: { streakDays = Int($0) }
                            ), in: 0 ... 90, step: 1)
                                .tint(Color.Design.sunriseGold)

                            HStack {
                                Text("0天")
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.mutedGray)
                                Spacer()
                                Text("90天")
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.mutedGray)
                            }
                        }
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: CornerRadius.md)
                                .fill(Color.Design.darkIndigo.opacity(0.6))
                        )
                    }
                    .padding(.horizontal, Spacing.lg)

                    // 快速选择按钮
                    VStack(spacing: Spacing.sm) {
                        Text("快速选择")
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.mutedGray)

                        HStack(spacing: Spacing.sm) {
                            ForEach([7, 14, 30, 60, 90], id: \.self) { days in
                                Button("\(days)天") {
                                    streakDays = days
                                }
                                .buttonStyle(QuickSelectButtonStyle(isSelected: streakDays == days))
                            }
                        }
                    }
                    .padding(.horizontal, Spacing.lg)

                    Spacer()

                    // 说明文字
                    VStack(spacing: Spacing.xs) {
                        Text(LocalizedStringKey("streak_settings_tip_icon"))
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.sunriseGold)

                        Text(String(format: NSLocalizedString("streak_settings_tip_message", comment: ""), streakDays))
                            .font(.system(size: 11))
                            .foregroundColor(Color.Design.mutedGray)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.horizontal, Spacing.lg)

                    Button(LocalizedStringKey("streak_settings_confirm")) {
                        onConfirm(streakDays)
                        dismiss()
                    }
                    .buttonStyle(SunriseGoldButtonStyle())
                    .padding(.horizontal, Spacing.lg)
                    .padding(.bottom, Spacing.lg)
                }
            }
            .navigationTitle(LocalizedStringKey("streak_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(LocalizedStringKey("cancel")) {
                        dismiss()
                    }
                    .foregroundColor(Color.Design.sunriseGold)
                }
            }
        }
    }
}

struct QuickSelectButtonStyle: ButtonStyle {
    let isSelected: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typography.caption)
            .foregroundColor(isSelected ? .white : Color.Design.sunriseGold)
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.sm)
            .background(
                Capsule()
                    .fill(isSelected ? Color.Design.sunriseGold : Color.Design.darkIndigo.opacity(0.6))
                    .overlay(
                        Capsule()
                            .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                    )
            )
            .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
    }
}

struct OnThisDaySettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var yearsAgo: Int
    let onConfirm: (Int) -> Void

    var body: some View {
        NavigationStack {
            ZStack {
                Color.Design.deepIndigo.ignoresSafeArea()

                VStack(spacing: Spacing.lg) {
                    Text(LocalizedStringKey("on_this_day_settings_header"))
                        .font(Typography.headline)
                        .foregroundColor(Color.Design.softWhite)
                        .padding(.top, Spacing.xl)

                    VStack(spacing: Spacing.md) {
                        Text(String(format: NSLocalizedString("on_this_day_settings_years", comment: ""), yearsAgo))
                            .font(Typography.title)
                            .foregroundColor(Color.Design.sunriseGold)

                        // 滑块
                        VStack(spacing: Spacing.sm) {
                            Slider(value: Binding(
                                get: { Double(yearsAgo) },
                                set: { yearsAgo = Int($0) }
                            ), in: 1 ... 5, step: 1)
                                .tint(Color.Design.sunriseGold)

                            HStack {
                                Text("1 年")
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.mutedGray)
                                Spacer()
                                Text("5 年")
                                    .font(Typography.caption)
                                    .foregroundColor(Color.Design.mutedGray)
                            }
                        }
                        .padding(.horizontal, Spacing.lg)

                        // 快捷选择
                        HStack(spacing: Spacing.sm) {
                            ForEach([1, 2, 3, 5], id: \.self) { years in
                                Button("\(years) 年") {
                                    yearsAgo = years
                                }
                                .buttonStyle(QuickSelectButtonStyle(isSelected: yearsAgo == years))
                            }
                        }
                    }
                    .padding(.horizontal, Spacing.lg)

                    Spacer()

                    // 说明文字
                    VStack(spacing: Spacing.xs) {
                        Text(LocalizedStringKey("streak_settings_tip_icon"))
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.sunriseGold)

                        Text(String(format: NSLocalizedString("on_this_day_settings_tip_message", comment: ""), yearsAgo))
                            .font(.system(size: 11))
                            .foregroundColor(Color.Design.mutedGray)
                            .multilineTextAlignment(.center)
                    }
                    .padding(.horizontal, Spacing.lg)

                    Button(LocalizedStringKey("on_this_day_settings_confirm")) {
                        onConfirm(yearsAgo)
                        dismiss()
                    }
                    .buttonStyle(SunriseGoldButtonStyle())
                    .padding(.horizontal, Spacing.lg)
                    .padding(.bottom, Spacing.lg)
                }
            }
            .navigationTitle(LocalizedStringKey("on_this_day_settings_title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(LocalizedStringKey("cancel")) {
                        dismiss()
                    }
                    .foregroundColor(Color.Design.sunriseGold)
                }
            }
        }
    }
}
#endif
