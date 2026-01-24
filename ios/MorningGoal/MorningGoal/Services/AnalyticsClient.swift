import CoreData
import Foundation

protocol AnalyticsClient {
    func trackAppLaunch(installID: String)
}

final class NoopAnalytics: AnalyticsClient {
    func trackAppLaunch(installID: String) {}
}

final class FileAnalyticsClient: AnalyticsClient {
    private let formatter: ISO8601DateFormatter = {
        let dateFormatter = ISO8601DateFormatter()
        dateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return dateFormatter
    }()

    private func fileURL() -> URL? {
        let fm = FileManager.default
        guard let dir = fm.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        return dir.appendingPathComponent("analytics_events.jsonl")
    }

    private func ensureFileExists(at url: URL) {
        let fm = FileManager.default
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil) }
    }

    func trackAppLaunch(installID: String) {
        guard let url = fileURL() else { return }
        ensureFileExists(at: url)
        let payload: [String: String] = [
            "event": "app_launch",
            "install_id": installID,
            "timestamp": formatter.string(from: Date())
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }
        let line = data + Data("\n".utf8)
        if let handle = try? FileHandle(forWritingTo: url) {
            defer { try? handle.close() }
            try? handle.seekToEnd()
            try? handle.write(contentsOf: line)
        }
    }
}

enum Analytics {
    static var shared: AnalyticsClient = NoopAnalytics()

    static func trackLaunchIfOptedIn(context: NSManagedObjectContext) {
        let req = NSFetchRequest<UserSettings>(entityName: "UserSettings")
        req.fetchLimit = 1
        guard let settings = try? context.fetch(req).first, settings.analyticsOptIn == true else { return }
        let id = InstallIDStore.shared.installID
        shared.trackAppLaunch(installID: id)
    }
}

final class InstallIDStore {
    static let shared = InstallIDStore()
    private let defaults = UserDefaults.standard
    private let key = "morning_goal_install_id"

    private init() {}

    var installID: String {
        if let existingID = defaults.string(forKey: key) { return existingID }
        let newID = UUID().uuidString

        defaults.set(newID, forKey: key)
        return newID
    }
}
