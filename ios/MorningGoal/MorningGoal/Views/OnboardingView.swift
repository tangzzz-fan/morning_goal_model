import CoreData
import SwiftUI

struct OnboardingView: View {
    @Environment(\.managedObjectContext) private var context
    var onComplete: (() -> Void)?

    @State private var currentStep: OnboardingStep = .theory
    @State private var startHour: Int = 7
    @State private var startMinute: Int = 0
    @State private var endHour: Int = 9
    @State private var endMinute: Int = 0
    @State private var committed: Bool = false
    @State private var commitmentProgress: Double = 0
    @State private var isCommitting: Bool = false
    @State private var skipForNow: Bool = false
    @State private var commitSucceeded: Bool = false

    // Feature slides content
    private let slides = OnboardingContentService.shared.getFeatureSlides()

    enum OnboardingStep: Int, CaseIterable {
        case theory
        case focus
        case streak
        case morningWindow
        case commitment
        case permissions
    }

    private func load() {
        let settings = UserSettings.fetchOrCreate(in: context)
        startHour = Int(settings.morningStartHour)
        startMinute = Int(settings.morningStartMinute)
        endHour = Int(settings.morningEndHour)
        endMinute = Int(settings.morningEndMinute)
        committed = settings.committed

        // 如果已经承诺过，跳过引导
        if committed {
            currentStep = .permissions
        }
    }

    private func persist() {
        let settings = UserSettings.fetchOrCreate(in: context)
        settings.morningStartHour = Int16(startHour)
        settings.morningStartMinute = Int16(startMinute)
        settings.morningEndHour = Int16(endHour)
        settings.morningEndMinute = Int16(endMinute)
        try? context.save()
    }

    private func commit() {
        // Haptics handled by LongPressButton
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            currentStep = .permissions
        }
    }

    private func completeOnboarding() {
        let settings = UserSettings.fetchOrCreate(in: context)
        settings.committed = true
        try? context.save()
        onComplete?()
    }

    var body: some View {
        ZStack {
            Color.Design.deepIndigo.ignoresSafeArea()

            // 使用TabView实现线性滑动切换
            TabView(selection: $currentStep) {
                // Theory
                if slides.indices.contains(0) {
                    FeatureSlideView(content: slides[0])
                        .tag(OnboardingStep.theory)
                }

                // Focus
                if slides.indices.contains(1) {
                    FeatureSlideView(content: slides[1])
                        .tag(OnboardingStep.focus)
                }

                // Streak/Growth
                if slides.indices.contains(2) {
                    FeatureSlideView(content: slides[2])
                        .tag(OnboardingStep.streak)
                }

                MorningWindowStep(
                    startHour: $startHour,
                    startMinute: $startMinute,
                    endHour: $endHour,
                    endMinute: $endMinute
                )
                .tag(OnboardingStep.morningWindow)

                CommitmentStep(
                    commitmentProgress: $commitmentProgress,
                    isCommitting: $isCommitting,
                    commitSucceeded: $commitSucceeded,
                    onCommit: commit
                )
                .tag(OnboardingStep.commitment)

                PermissionsStep()
                    .tag(OnboardingStep.permissions)
            }
            .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
            .animation(.easeInOut, value: currentStep)
            .onChange(of: currentStep) { oldStep, newStep in
                // 离开早晨窗口页时保存设置
                if oldStep == .morningWindow {
                    persist()
                }

                if oldStep == .commitment && !committed && !skipForNow && !commitSucceeded {
                    if newStep == .permissions { currentStep = .commitment }
                }
            }

            // 底部操作栏 (Overlay)
            VStack(spacing: Spacing.md) {
                Spacer()

                HStack(spacing: Spacing.md) {
                    ForEach(OnboardingStep.allCases, id: \.self) { step in
                        Circle()
                            .fill(currentStep == step ? Color.Design.sunriseGold : Color.Design.mutedGray.opacity(0.3))
                            .frame(width: 8, height: 8)
                    }
                }
                .padding(.vertical, Spacing.md)

                bottomActions()
            }
            .frame(maxWidth: .infinity)
            .padding(.bottom, Spacing.lg)
            // Use background with correct padding to obscure content behind
            .background(
                VStack {
                    Spacer()
                    Color.Design.deepIndigo.opacity(0.95)
                        .frame(height: 140) // Approximate height of bottom area
                        .mask(LinearGradient(gradient: Gradient(colors: [.clear, .black, .black]), startPoint: .top, endPoint: .bottom))
                }
                .ignoresSafeArea()
            )
        }
        .onAppear { load() }
        .preferredColorScheme(.dark)
    }

    @ViewBuilder
    private func bottomActions() -> some View {
        switch currentStep {
        case .theory, .focus, .streak, .morningWindow:
            // 移除导航按钮，使用 PageControl 和滑动进行导航
            EmptyView()

        case .commitment:
            Button(LocalizedStringKey("onboarding_skip_for_now")) {
                skipForNow = true
                withAnimation { currentStep = .permissions }
            }
            .font(Typography.caption)
            .foregroundColor(Color.Design.mutedGray)

        case .permissions:
            VStack(spacing: Spacing.sm) {
                Button(LocalizedStringKey("onboarding_enable_notifications")) {
                    NotificationService.shared.requestAuthorization()
                    NotificationService.shared.scheduleDaily(
                        startHour: startHour,
                        startMinute: startMinute,
                        endHour: endHour,
                        endMinute: endMinute
                    )
                    NotificationService.shared.scheduleBadgeAtStart(
                        startHour: startHour,
                        startMinute: startMinute
                    )
                    completeOnboarding()
                }
                .buttonStyle(SunriseGoldButtonStyle())

                Button(LocalizedStringKey("onboarding_maybe_later")) {
                    completeOnboarding()
                }
                .font(Typography.caption)
                .foregroundColor(Color.Design.mutedGray)
            }
        }
    }
}

// MARK: - 步骤组件

struct FeatureSlideView: View {
    let content: FeatureSlideContent

    var body: some View {
        VStack(spacing: Spacing.xl) {
            Spacer()

            // 图标或图片
            Image(systemName: content.imageSystemName)
                .font(.system(size: 80))
                .foregroundColor(color(for: content.colorName))
                .padding(Spacing.xl)
                .background(
                    Circle()
                        .fill(color(for: content.colorName).opacity(0.1))
                        .frame(width: 160, height: 160)
                        .overlay(
                            Circle()
                                .stroke(color(for: content.colorName).opacity(0.3), lineWidth: 1)
                        )
                )
                .shadow(color: color(for: content.colorName).opacity(0.3), radius: 20, x: 0, y: 10)

            VStack(spacing: Spacing.md) {
                Text(content.title)
                    .font(Typography.title)
                    .foregroundColor(Color.Design.softWhite)
                    .multilineTextAlignment(.center)

                Text(content.subtitle)
                    .font(Typography.body)
                    .foregroundColor(Color.Design.mutedGray)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, Spacing.lg)
                    .fixedSize(horizontal: false, vertical: true) // 防止截断
            }

            Spacer()
        }
        .padding(.bottom, 60) // 为底部按钮留出空间
    }

    private func color(for name: String) -> Color {
        switch name {
        case "sunriseGold": return Color.Design.sunriseGold
        case "deepIndigo": return Color.Design.softWhite // deepIndigo 是背景色，这里用白色
        case "orange": return .orange
        default: return Color.Design.sunriseGold
        }
    }
}

struct MorningWindowStep: View {
    @Binding var startHour: Int
    @Binding var startMinute: Int
    @Binding var endHour: Int
    @Binding var endMinute: Int

    // 将小时和分钟转换为 Date 用于 DatePicker
    private var startTime: Binding<Date> {
        Binding(
            get: {
                var components = DateComponents()
                components.hour = startHour
                components.minute = startMinute
                return Calendar.current.date(from: components) ?? Date()
            },
            set: { newDate in
                let components = Calendar.current.dateComponents([.hour, .minute], from: newDate)
                startHour = components.hour ?? 7
                startMinute = components.minute ?? 0
            }
        )
    }

    private var endTime: Binding<Date> {
        Binding(
            get: {
                var components = DateComponents()
                components.hour = endHour
                components.minute = endMinute
                return Calendar.current.date(from: components) ?? Date()
            },
            set: { newDate in
                let components = Calendar.current.dateComponents([.hour, .minute], from: newDate)
                endHour = components.hour ?? 9
                endMinute = components.minute ?? 0
            }
        )
    }

    var body: some View {
        VStack(spacing: Spacing.lg) {
            Spacer()
                .frame(height: Spacing.xl)

            Text(LocalizedStringKey("onboarding_morning_window_subtitle"))
                .font(Typography.headline)
                .foregroundColor(Color.Design.softWhite)
                .multilineTextAlignment(.center)

            Text(LocalizedStringKey("onboarding_morning_window_title"))
                .font(Typography.body)
                .foregroundColor(Color.Design.mutedGray)
                .multilineTextAlignment(.center)

            HStack(spacing: Spacing.md) {
                VStack(alignment: .leading, spacing: Spacing.sm) {
                    Text(LocalizedStringKey("onboarding_time_from"))
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray)
                    DatePicker("", selection: startTime, displayedComponents: .hourAndMinute)
                        .datePickerStyle(.compact)
                        .labelsHidden()
                        .tint(Color.Design.sunriseGold)
                        .controlSize(.small)
                        .frame(height: 32)
                }

                Rectangle()
                    .fill(Color.Design.sunriseGold.opacity(0.2))
                    .frame(width: 1, height: 80)

                VStack(alignment: .leading, spacing: Spacing.sm) {
                    Text(LocalizedStringKey("onboarding_time_to"))
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray)
                    DatePicker("", selection: endTime, displayedComponents: .hourAndMinute)
                        .datePickerStyle(.compact)
                        .labelsHidden()
                        .tint(Color.Design.sunriseGold)
                        .controlSize(.small)
                        .frame(height: 32)
                }
            }
            .padding(.horizontal)
            .background(
                RoundedRectangle(cornerRadius: CornerRadius.md)
                    .fill(Color.Design.darkIndigo.opacity(0.5))
                    .overlay(
                        RoundedRectangle(cornerRadius: CornerRadius.md)
                            .stroke(Color.Design.sunriseGold.opacity(0.2), lineWidth: 1)
                    )
            )
            .frame(height: 240)

            Spacer()
                .frame(height: Spacing.xl)
        }
        .padding(.horizontal, Spacing.xl)
        .frame(maxWidth: .infinity)
        .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)))
    }
}

struct CommitmentStep: View {
    @Binding var commitmentProgress: Double
    @Binding var isCommitting: Bool
    @Binding var commitSucceeded: Bool
    let onCommit: () -> Void

    var body: some View {
        VStack(spacing: Spacing.lg) {
            Spacer()
                .frame(height: Spacing.xl)

            Text(LocalizedStringKey("onboarding_commitment_title_1"))
                .font(Typography.headline)
                .foregroundColor(Color.Design.softWhite)
                .multilineTextAlignment(.center)

            Text(LocalizedStringKey("onboarding_commitment_title_2"))
                .font(Typography.headline)
                .foregroundColor(Color.Design.softWhite)
                .multilineTextAlignment(.center)

            Spacer()

            CommitDeviceView(
                progress: $commitmentProgress,
                isCommitting: $isCommitting,
                size: 140, // Original large size for onboarding
                duration: 3.0 // Original duration
            ) {
                commitSucceeded = true
                onCommit()
            }

            Group {
                if commitSucceeded {
                    Text(LocalizedStringKey("onboarding_commitment_success"))
                } else if isCommitting {
                    let remaining = max(0, 3 - commitmentProgress * 3)
                    if remaining < 0.1 {
                        Text(LocalizedStringKey("onboarding_commitment_success"))
                    } else {
                        Text(String(format: NSLocalizedString("onboarding_commitment_holding_seconds", comment: ""), remaining))
                    }
                } else {
                    Text(LocalizedStringKey("onboarding_commitment_action"))
                }
            }
            .font(Typography.caption)
            .foregroundColor(Color.Design.mutedGray)

            Spacer()
        }
        .padding(.horizontal, Spacing.xl)
        .frame(maxWidth: .infinity)
        .transition(.asymmetric(insertion: .move(edge: .trailing), removal: .move(edge: .leading)))
    }
}

struct PermissionsStep: View {
    @State private var showSettings = false

    var body: some View {
        VStack(spacing: Spacing.lg) {
            Spacer()
                .frame(height: Spacing.xl)

            Text(LocalizedStringKey("onboarding_permissions_title"))
                .font(Typography.headline)
                .foregroundColor(Color.Design.softWhite)

            VStack(spacing: Spacing.md) {
                Image(systemName: "bell.badge.fill")
                    .font(.system(size: 48))
                    .foregroundColor(Color.Design.sunriseGold)

                Text(LocalizedStringKey("onboarding_enable_notifications"))
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.softWhite)

                Text(LocalizedStringKey("onboarding_permissions_message"))
                    .font(Typography.body)
                    .foregroundColor(Color.Design.mutedGray)
                    .multilineTextAlignment(.center)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: CornerRadius.md)
                    .fill(Color.Design.darkIndigo.opacity(0.5))
                    .overlay(
                        RoundedRectangle(cornerRadius: CornerRadius.md)
                            .stroke(Color.Design.sunriseGold.opacity(0.2), lineWidth: 1)
                    )
            )

            // 操作移至底部统一区域

            Spacer()
                .frame(height: Spacing.xl)
        }
        .padding(.horizontal, Spacing.xl)
        .frame(maxWidth: .infinity)
        .transition(.move(edge: .trailing))
    }
}

// MARK: - 自定义按钮样式

struct SunriseGoldButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(Typography.body)
            .foregroundColor(.white)
            .padding(.horizontal, Spacing.lg)
            .padding(.vertical, Spacing.md)
            .background(
                Capsule()
                    .fill(Color.Design.sunriseGold)
                    .scaleEffect(configuration.isPressed ? 0.95 : 1.0)
            )
            .animation(.easeInOut(duration: 0.1), value: configuration.isPressed)
    }
}
