//
//  MorningGoalApp.swift
//  MorningGoal
//
//  Created by 小苹果 on 2025/11/12.
//

import CoreData
import SwiftUI

@main
struct MorningGoalApp: App {
    let dataController = DataController.shared
    @Environment(\.scenePhase) private var scenePhase
    @State private var showLaunchScreen = true

    init() { Analytics.shared = NoopAnalytics() }
    #if DEBUG
    private func _debugValidateTimeWindow(context: NSManagedObjectContext) {
        let settings = UserSettings.fetchOrCreate(in: context)
        let now = Date()
        let cal = Calendar.current
        let comps = cal.dateComponents([.year, .month, .day], from: now)
        var start = comps
        start.hour = Int(settings.morningStartHour)
        start.minute = Int(settings.morningStartMinute)
        var end = comps
        end.hour = Int(settings.morningEndHour)
        end.minute = Int(settings.morningEndMinute)
        guard let sd = cal.date(from: start), let ed = cal.date(from: end) else { return }
        let wrap = sd > ed
        let manual = (wrap && (now >= sd || now <= ed)) || (!wrap && (now >= sd && now <= ed))
        assert(manual == settings.isWithinMorningWindow(now: now))
    }
    #endif

    var body: some Scene {
        WindowGroup {
            ZStack {
                ContentView()
                    .environment(\.managedObjectContext, dataController.container.viewContext)

                // 启动屏幕覆盖层
                if showLaunchScreen {
                    LaunchScreenView()
                        .transition(.opacity)
                        .zIndex(1)
                }
            }
            .onAppear {
                // Initialize CloudKit schema on first launch
                dataController.initializeCloudKitSchema()

                // Check CloudKit sync status after a delay to allow initial sync
                DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                    dataController.checkCloudKitStatus()
                }

                // 1.5秒后隐藏启动屏幕
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                    withAnimation(.easeOut(duration: 0.5)) {
                        showLaunchScreen = false
                    }
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                let context = dataController.container.viewContext
                let settings = UserSettings.fetchOrCreate(in: context)
                if settings.isWithinMorningWindow() && !PromptStateService.shared.hasPromptedToday() {
                    NotificationService.shared.clearBadgeAndCancelToday()
                    PromptStateService.shared.markPromptedToday()
                }
                #if DEBUG
                _debugValidateTimeWindow(context: context)
                #endif
            }
            if phase == .background { dataController.save() }
        }
    }
}
