import Combine
import CoreData

/// Defines the contract for data persistence
protocol StorageService {
    var viewContext: NSManagedObjectContext { get }
    func save()
}

/// Defines the contract for sync status monitoring (e.g. CloudKit)
protocol SyncMonitorService: ObservableObject {
    var isSyncing: Bool { get }
    var lastSyncDate: Date? { get }
    var syncError: Error? { get }
}
