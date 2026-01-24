import Foundation
import UIKit
import UserNotifications

protocol NotificationCenterType {
    func requestAuthorization(options: UNAuthorizationOptions, completionHandler: @escaping (Bool, Error?) -> Void)
    func add(_ request: UNNotificationRequest)
    func removePendingNotificationRequests(withIdentifiers: [String])
}

final class RealNotificationCenter: NotificationCenterType {
    func requestAuthorization(options: UNAuthorizationOptions, completionHandler: @escaping (Bool, Error?) -> Void) {
        UNUserNotificationCenter.current().requestAuthorization(options: options, completionHandler: completionHandler)
    }

    func add(_ request: UNNotificationRequest) {
        UNUserNotificationCenter.current().add(request)
    }

    func removePendingNotificationRequests(withIdentifiers: [String]) {
        UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: withIdentifiers)
    }
}

protocol BadgeSetter { func setBadge(_ number: Int) }

struct AppBadgeSetter: BadgeSetter {
    func setBadge(_ number: Int) {
        UIApplication.shared.applicationIconBadgeNumber = number
    }
}

final class NotificationService {
    static let shared = NotificationService()

    private let center: NotificationCenterType
    private let badge: BadgeSetter

    typealias ContentBuilder = () -> UNMutableNotificationContent
    typealias TriggerBuilder = (_ hour: Int, _ minute: Int) -> UNNotificationTrigger?

    private let buildContent: ContentBuilder
    private let buildTrigger: TriggerBuilder

    init(
        center: NotificationCenterType = RealNotificationCenter(),
        badge: BadgeSetter = AppBadgeSetter(),
        contentBuilder: @escaping ContentBuilder = {
            let content = UNMutableNotificationContent()
            content.title = "准备好设定今天的焦点了吗？"
            content.sound = .default

            return content
        },
        triggerBuilder: @escaping TriggerBuilder = { hour, minute in
            var date = DateComponents()
            date.hour = hour
            date.minute = minute
            return UNCalendarNotificationTrigger(dateMatching: date, repeats: true)
        }
    ) {
        self.center = center
        self.badge = badge
        self.buildContent = contentBuilder
        self.buildTrigger = triggerBuilder
    }

    func requestAuthorization() {
        center.requestAuthorization(options: [.alert, .badge, .sound]) { _, _ in }
    }

    func scheduleDaily(startHour: Int, startMinute: Int, endHour: Int, endMinute: Int) {
        let randomHour = Int.random(in: startHour ... endHour)
        let randomMinute = Int.random(in: 0 ... 59)
        center.removePendingNotificationRequests(withIdentifiers: ["morning_goal_nudge"])
        guard let trigger = buildTrigger(randomHour, randomMinute) else { return }
        let content = buildContent()
        let req = UNNotificationRequest(identifier: "morning_goal_nudge", content: content, trigger: trigger)
        center.add(req)
    }

    func scheduleBadgeAtStart(startHour: Int, startMinute: Int) {
        center.removePendingNotificationRequests(withIdentifiers: ["morning_goal_badge"])
        var dc = DateComponents()
        dc.hour = startHour
        dc.minute = startMinute
        let trigger = UNCalendarNotificationTrigger(dateMatching: dc, repeats: true)
        let content = UNMutableNotificationContent()
        content.badge = NSNumber(value: 1)
        let req = UNNotificationRequest(identifier: "morning_goal_badge", content: content, trigger: trigger)
        center.add(req)
    }

    func clearBadgeAndCancelToday() {
        badge.setBadge(0)
        center.removePendingNotificationRequests(withIdentifiers: ["morning_goal_nudge", "morning_goal_badge"])
    }
}
