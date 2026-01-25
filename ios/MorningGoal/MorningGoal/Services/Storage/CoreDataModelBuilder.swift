import CoreData
import Foundation

enum CoreDataModelBuilder {
    private static var _cachedModel: NSManagedObjectModel?

    static func makeModel() -> NSManagedObjectModel {
        if let model = _cachedModel {
            return model
        }

        let model = NSManagedObjectModel()
        let goalEntry = makeGoalEntryEntity()
        let userSettings = makeUserSettingsEntity()
        model.entities = [goalEntry, userSettings]

        _cachedModel = model
        return model
    }

    private static func makeGoalEntryEntity() -> NSEntityDescription {
        let goalEntry = NSEntityDescription()
        goalEntry.name = "GoalEntry"
        goalEntry.managedObjectClassName = NSStringFromClass(GoalEntry.self)

        // CloudKit requires: all non-optional attributes must have default values

        // MARK: - 基础字段

        let geDateString = attrString("dateString", optional: false, defaultValue: "")
        let geGoalText = attrString("goalText", optional: false, defaultValue: "")
        let geLastUpdated = attrDate("lastUpdated", optional: false, defaultValue: Date())

        // MARK: - 分析结果字段 (7个维度)

        // Topic (主题) - 对应 category
        let geCategory = attrString("category", optional: true)
        let geCategoryConfidence = attrDouble("categoryConfidence", defaultValue: 0.0)

        // Sentiment (情感)
        let geSentiment = attrString("sentiment", optional: true)
        let geSentimentScore = attrDouble("sentimentScore", defaultValue: 0.0)

        // Urgency (紧急度) - 新增
        let geUrgency = attrString("urgency", optional: true)
        let geUrgencyConfidence = attrDouble("urgencyConfidence", defaultValue: 0.0)

        // TimeFrame (时间范围) - 新增
        let geTimeFrame = attrString("timeFrame", optional: true)
        let geTimeFrameConfidence = attrDouble("timeFrameConfidence", defaultValue: 0.0)

        // ActionType (行动类型) - 新增
        let geActionType = attrString("actionType", optional: true)
        let geActionTypeConfidence = attrDouble("actionTypeConfidence", defaultValue: 0.0)

        // Difficulty (难度) - 新增
        let geDifficulty = attrString("difficulty", optional: true)
        let geDifficultyConfidence = attrDouble("difficultyConfidence", defaultValue: 0.0)

        // Specificity (具体程度) - 新增
        let geSpecificity = attrString("specificity", optional: true)
        let geSpecificityConfidence = attrDouble("specificityConfidence", defaultValue: 0.0)

        // 分析时间
        let geAnalyzedAt = attrDate("analyzedAt", optional: true, defaultValue: nil)

        // Embedding (可选，用于相似度搜索)
        let geEmbedding = attrBinary("embedding", optional: true)

        // MARK: - 用户纠正字段

        let geCategoryUserCorrected = attrString("categoryUserCorrected", optional: true)
        let geSentimentUserCorrected = attrString("sentimentUserCorrected", optional: true)
        let geUrgencyUserCorrected = attrString("urgencyUserCorrected", optional: true)
        let geTimeFrameUserCorrected = attrString("timeFrameUserCorrected", optional: true)
        let geActionTypeUserCorrected = attrString("actionTypeUserCorrected", optional: true)
        let geDifficultyUserCorrected = attrString("difficultyUserCorrected", optional: true)
        let geSpecificityUserCorrected = attrString("specificityUserCorrected", optional: true)
        let geCorrectedAt = attrDate("correctedAt", optional: true, defaultValue: nil)
        let geIsTrainingSample = attrBool("isTrainingSample", defaultValue: false)

        goalEntry.properties = [
            // 基础字段
            geDateString,
            geGoalText,
            geLastUpdated,
            // 分析结果 - Topic/Sentiment (原有)
            geCategory,
            geCategoryConfidence,
            geSentiment,
            geSentimentScore,
            // 分析结果 - 5个新维度
            geUrgency,
            geUrgencyConfidence,
            geTimeFrame,
            geTimeFrameConfidence,
            geActionType,
            geActionTypeConfidence,
            geDifficulty,
            geDifficultyConfidence,
            geSpecificity,
            geSpecificityConfidence,
            // 分析时间和Embedding
            geAnalyzedAt,
            geEmbedding,
            // 用户纠正字段
            geCategoryUserCorrected,
            geSentimentUserCorrected,
            geUrgencyUserCorrected,
            geTimeFrameUserCorrected,
            geActionTypeUserCorrected,
            geDifficultyUserCorrected,
            geSpecificityUserCorrected,
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

    private static func attrBinary(_ name: String, optional: Bool) -> NSAttributeDescription {
        let attr = NSAttributeDescription()
        attr.name = name
        attr.attributeType = .binaryDataAttributeType
        attr.isOptional = optional
        attr.allowsExternalBinaryDataStorage = true // 允许外部存储大型二进制数据
        return attr
    }
}
