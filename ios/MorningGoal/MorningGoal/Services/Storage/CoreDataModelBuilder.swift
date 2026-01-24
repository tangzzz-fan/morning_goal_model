import CoreData
import Foundation

enum CoreDataModelBuilder {
    static func makeModel() -> NSManagedObjectModel {
        let model = NSManagedObjectModel()
        let goalEntry = makeGoalEntryEntity()
        let userSettings = makeUserSettingsEntity()
        model.entities = [goalEntry, userSettings]
        return model
    }

    private static func makeGoalEntryEntity() -> NSEntityDescription {
        let goalEntry = NSEntityDescription()
        goalEntry.name = "GoalEntry"
        goalEntry.managedObjectClassName = NSStringFromClass(GoalEntry.self)

        // CloudKit requires: all non-optional attributes must have default values
        let geDateString = attrString("dateString", optional: false, defaultValue: "")
        let geGoalText = attrString("goalText", optional: false, defaultValue: "")
        let geLastUpdated = attrDate("lastUpdated", optional: false, defaultValue: Date())
        let geCategory = attrString("category", optional: true)
        let geCategoryConfidence = attrDouble("categoryConfidence", defaultValue: 0.0)
        let geSentiment = attrString("sentiment", optional: true)
        let geSentimentScore = attrDouble("sentimentScore", defaultValue: 0.0)
        let geAnalyzedAt = attrDate("analyzedAt", optional: true, defaultValue: nil)
        let geCategoryUserCorrected = attrString("categoryUserCorrected", optional: true)
        let geSentimentUserCorrected = attrString("sentimentUserCorrected", optional: true)
        let geCorrectedAt = attrDate("correctedAt", optional: true, defaultValue: nil)
        let geIsTrainingSample = attrBool("isTrainingSample", defaultValue: false)

        goalEntry.properties = [
            geDateString,
            geGoalText,
            geLastUpdated,
            geCategory,
            geCategoryConfidence,
            geSentiment,
            geSentimentScore,
            geAnalyzedAt,
            geCategoryUserCorrected,
            geSentimentUserCorrected,
            geCorrectedAt,
            geIsTrainingSample
        ]
        // CloudKit does not support unique constraints - removed
        // goalEntry.uniquenessConstraints = [["dateString"]]
        return goalEntry
    }

    private static func makeUserSettingsEntity() -> NSEntityDescription {
        let userSettings = NSEntityDescription()
        userSettings.name = "UserSettings"
        userSettings.managedObjectClassName = NSStringFromClass(UserSettings.self)

        let usStartHour = attrInt16("morningStartHour", defaultValue: 7)
        let usStartMinute = attrInt16("morningStartMinute", defaultValue: 0)
        let usEndHour = attrInt16("morningEndHour", defaultValue: 9)
        let usEndMinute = attrInt16("morningEndMinute", defaultValue: 0)
        let usOptIn = attrBool("analyticsOptIn", defaultValue: false)
        let usCommitted = attrBool("committed", defaultValue: false)

        userSettings.properties = [usStartHour, usStartMinute, usEndHour, usEndMinute, usOptIn, usCommitted]
        return userSettings
    }

    // MARK: - Attribute Helpers

    private static func attrString(_ name: String, optional: Bool, defaultValue: String? = nil) -> NSAttributeDescription {
        let attr = NSAttributeDescription()
        attr.name = name
        attr.attributeType = .stringAttributeType
        attr.isOptional = optional
        if let defaultValue = defaultValue {
            attr.defaultValue = defaultValue
        }
        return attr
    }

    private static func attrDouble(_ name: String, defaultValue: Double) -> NSAttributeDescription {
        let attr = NSAttributeDescription()
        attr.name = name
        attr.attributeType = .doubleAttributeType
        attr.isOptional = false
        attr.defaultValue = defaultValue
        return attr
    }

    private static func attrDate(_ name: String, optional: Bool, defaultValue: Date?) -> NSAttributeDescription {
        let attr = NSAttributeDescription()
        attr.name = name
        attr.attributeType = .dateAttributeType
        attr.isOptional = optional
        attr.defaultValue = defaultValue
        return attr
    }

    private static func attrBool(_ name: String, defaultValue: Bool) -> NSAttributeDescription {
        let attr = NSAttributeDescription()
        attr.name = name
        attr.attributeType = .booleanAttributeType
        attr.isOptional = false
        attr.defaultValue = defaultValue
        return attr
    }

    private static func attrInt16(_ name: String, defaultValue: Int16 = 0) -> NSAttributeDescription {
        let attr = NSAttributeDescription()
        attr.name = name
        attr.attributeType = .integer16AttributeType
        attr.isOptional = false
        attr.defaultValue = defaultValue
        return attr
    }
}
