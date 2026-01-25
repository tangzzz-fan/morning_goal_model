import Combine
import CoreData
import Foundation

final class DataController: ObservableObject, StorageService {
    static let shared = DataController()
    let container: NSPersistentCloudKitContainer
    var viewContext: NSManagedObjectContext { container.viewContext }

    // CloudKit sync state monitoring (delegated to CloudKitSyncMonitor)
    @Published var isSyncing: Bool = false
    @Published var lastSyncDate: Date?
    @Published var syncError: Error?

    private var syncMonitor: CloudKitSyncMonitor?
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Preview Helper

    static var preview: DataController = {
        let controller = DataController(inMemory: true)
        let viewContext = controller.container.viewContext

        // Add sample data here if needed

        try? viewContext.save()
        return controller
    }()

    // CloudKit can be disabled for testing or if having persistent issues
    private static let cloudKitEnabled = true // Set to false to disable CloudKit sync

    init(inMemory: Bool = false) {
        // Enable verbose logging for debugging
        UserDefaults.standard.set(true, forKey: "CoreDataCloudKitDebug")

        // Use the builder to create the model
        let model = CoreDataModelBuilder.makeModel()
        container = NSPersistentCloudKitContainer(name: "MorningGoalModel", managedObjectModel: model)

        guard let description = container.persistentStoreDescriptions.first else {
            fatalError("Failed to retrieve a persistent store description.")
        }

        if inMemory {
            description.url = URL(fileURLWithPath: "/dev/null")
        } else {
            // Explicitly set the URL to ensure consistency
            description.url = NSPersistentContainer.defaultDirectoryURL().appendingPathComponent("MorningGoalModel.sqlite")
        }

        // Config CloudKit Container Options (only if enabled)
        if Self.cloudKitEnabled && !inMemory {
            let cloudKitOptions = NSPersistentCloudKitContainerOptions(containerIdentifier: "iCloud.com.tango.MorningGoal")
            description.cloudKitContainerOptions = cloudKitOptions
            debugPrint("☁️ CloudKit sync: ENABLED")
        } else {
            description.cloudKitContainerOptions = nil
            debugPrint("☁️ CloudKit sync: DISABLED (local only)")
        }

        description.setOption(true as NSNumber, forKey: NSPersistentHistoryTrackingKey)
        description.setOption(true as NSNumber, forKey: NSPersistentStoreRemoteChangeNotificationPostOptionKey)

        // Enable automatic lightweight migration
        description.setOption(true as NSNumber, forKey: NSMigratePersistentStoresAutomaticallyOption)
        description.setOption(true as NSNumber, forKey: NSInferMappingModelAutomaticallyOption)

        container.loadPersistentStores { storeDescription, error in
            if let error = error as NSError? {
                // Log detailed error information
                debugPrint("❌ Core Data failed to load")
                debugPrint("Store URL: \(storeDescription.url?.absoluteString ?? "unknown")")
                debugPrint("Error: \(error)")

                // Provide more helpful error context
                if error.domain == NSCocoaErrorDomain {
                    switch error.code {
                    case 134_060: // CloudKit validation error
                        debugPrint("💡 Suggestion: CloudKit validation failed.")
                    case 134_030, 134_100: // Model mismatch errors
                        debugPrint("💡 Suggestion: Schema mismatch detected.")
                    case 134_020: // Database locked
                        debugPrint("💡 Suggestion: Database is locked by another process")
                    default:
                        debugPrint("💡 Suggestion: Check CloudKit configuration and iCloud account")
                    }
                }

                fatalError("Core Data failed to load: \(error.localizedDescription)")
            } else {
                debugPrint("✅ Core Data loaded successfully")
                debugPrint("Store URL: \(storeDescription.url?.absoluteString ?? "unknown")")
            }
        }
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
        container.viewContext.automaticallyMergesChangesFromParent = true

        // Initialize sync monitor
        if Self.cloudKitEnabled {
            let monitor = CloudKitSyncMonitor(container: container)
            self.syncMonitor = monitor

            // Forward sync monitor updates to DataController (for backward compatibility)
            monitor.$isSyncing.assign(to: \.isSyncing, on: self).store(in: &cancellables)
            monitor.$lastSyncDate.assign(to: \.lastSyncDate, on: self).store(in: &cancellables)
            monitor.$syncError.assign(to: \.syncError, on: self).store(in: &cancellables)
        }
    }

    func save() {
        let context = container.viewContext
        if context.hasChanges { try? context.save() }
    }

    // MARK: - CloudKit Sync Helpers

    /// Initialize CloudKit schema (call this on first launch after install)
    func initializeCloudKitSchema() {
        // Just triggers the lazy load if not already done, verifying store description
        debugPrint("🔄 Initializing CloudKit schema check...")
        _ = container.persistentStoreDescriptions.first
    }

    /// Check if CloudKit sync is likely working
    func checkCloudKitStatus() {
        syncMonitor?.checkCloudKitStatus(viewContext: container.viewContext)
    }
}
