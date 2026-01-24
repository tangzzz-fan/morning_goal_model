import XCTest
import CoreData
@testable import MorningGoal

final class InMemoryCoreData {
    static func container() -> NSPersistentContainer {
        let model = CoreDataModelBuilder.makeModel()
        let container = NSPersistentContainer(name: "TestModel", managedObjectModel: model)
        let description = NSPersistentStoreDescription()
        description.type = NSInMemoryStoreType
        container.persistentStoreDescriptions = [description]
        container.loadPersistentStores { _, _ in }
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
        return container
    }
}
