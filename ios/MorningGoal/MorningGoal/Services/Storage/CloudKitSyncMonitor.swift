import Combine
import CoreData
import Foundation

class CloudKitSyncMonitor: SyncMonitorService {
    @Published var isSyncing: Bool = false
    @Published var lastSyncDate: Date?
    @Published var syncError: Error?

    private var container: NSPersistentCloudKitContainer
    private var cancellables = Set<AnyCancellable>()

    init(container: NSPersistentCloudKitContainer) {
        self.container = container
        setupCloudKitNotifications()
    }

    private func setupCloudKitNotifications() {
        NotificationCenter.default.publisher(for: NSPersistentCloudKitContainer.eventChangedNotification, object: container)
            .sink { [weak self] notification in
                self?.handleImportEvent(notification)
            }
            .store(in: &cancellables)

        debugPrint("🔔 CloudKit sync monitoring enabled")
    }

    private func handleImportEvent(_ notification: Notification) {
        guard let event = notification.userInfo?[NSPersistentCloudKitContainer.eventNotificationUserInfoKey] as? NSPersistentCloudKitContainer.Event
        else {
            return
        }

        // Update observable state on main thread if needed
        DispatchQueue.main.async { [weak self] in
            self?.processEvent(event)
        }
    }

    private func processEvent(_ event: NSPersistentCloudKitContainer.Event) {
        debugPrint("☁️ CloudKit Event:")
        debugPrint("  Type: \(event.type == .setup ? "Setup" : event.type == .import ? "Import" : event.type == .export ? "Export" : "Unknown")")
        debugPrint("  Start: \(event.startDate)")
        debugPrint("  End: \(event.endDate ?? Date())")

        if let error = event.error {
            handleError(error)
        } else {
            debugPrint("  ✅ Success")
            if event.type == .import {
                lastSyncDate = event.endDate
                debugPrint("  📥 Data imported from iCloud")
            } else if event.type == .export {
                lastSyncDate = event.endDate
                debugPrint("  📤 Data exported to iCloud")
            }
            isSyncing = event.endDate == nil
        }
    }

    private func handleError(_ error: Error) {
        debugPrint("  ❌ Error: \(error.localizedDescription)")

        let nsError = error as NSError
        debugPrint("  Error Domain: \(nsError.domain)")
        debugPrint("  Error Code: \(nsError.code)")

        if nsError.domain == "CKErrorDomain" {
            logCloudKitErrorDiagnosis(code: nsError.code)
        }

        syncError = error
        isSyncing = false
    }

    private func logCloudKitErrorDiagnosis(code: Int) {
        switch code {
        case 2: // CKErrorPartialFailure
            debugPrint("\n⚠️ CloudKit Partial Failure:")
            debugPrint("  This usually means the CloudKit container needs initialization")
            debugPrint("  🔧 Solutions:")
            debugPrint("     1. Wait 5-10 minutes and relaunch app (container auto-creates)")
            debugPrint("     2. Check iCloud settings: Settings → [Your Name] → iCloud")

        case 15: // CKErrorServerRejectedRequest
            debugPrint("\n❌ CloudKit Server Rejected Request:")
            debugPrint("  The CloudKit container may not exist yet")
            debugPrint("  🔧 Action Required:")
            debugPrint("     1. Verify iCloud capability in Xcode")

        case 3: // CKErrorNetworkUnavailable
            debugPrint("\n📡 Network unavailable - will retry when online")

        case 9: // CKErrorNotAuthenticated
            debugPrint("\n🔐 Not signed into iCloud - please sign in via Settings")

        default:
            debugPrint("\n💡 Unknown CloudKit error - check network and iCloud account")
        }
    }

    /// Check if CloudKit sync is likely working
    func checkCloudKitStatus(viewContext: NSManagedObjectContext) {
        debugPrint("\n📊 CloudKit Status Check:")
        debugPrint("  Last sync: \(lastSyncDate?.description ?? "Never")")
        debugPrint("  Is syncing: \(isSyncing)")

        if let error = syncError {
            debugPrint("  Last error: \(error.localizedDescription)")
        } else {
            debugPrint("  No errors")
        }

        let request = NSFetchRequest<NSFetchRequestResult>(entityName: "GoalEntry")
        request.includesSubentities = false

        do {
            let count = try viewContext.count(for: request)
            debugPrint("  Local entries: \(count)")
        } catch {
            debugPrint("  ❌ Failed to count entries: \(error)")
        }
    }
}
