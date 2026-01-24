import CoreData
import Foundation

@objc(GoalEntry)
final class GoalEntry: NSManagedObject, Identifiable {
    // Identifiable conformance
    var id: NSManagedObjectID { objectID }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    // MARK: - 原有字段

    @NSManaged var dateString: String
    @NSManaged var goalText: String
    @NSManaged var lastUpdated: Date

    // MARK: - 分析结果字段 (7个维度)

    // Topic (主题) - 对应 category
    @NSManaged var category: String?
    @NSManaged var categoryConfidence: Double

    // Sentiment (情感)
    @NSManaged var sentiment: String?
    @NSManaged var sentimentScore: Double

    // Urgency (紧急度) - 新增
    @NSManaged var urgency: String?
    @NSManaged var urgencyConfidence: Double

    // TimeFrame (时间范围) - 新增
    @NSManaged var timeFrame: String?
    @NSManaged var timeFrameConfidence: Double

    // ActionType (行动类型) - 新增
    @NSManaged var actionType: String?
    @NSManaged var actionTypeConfidence: Double

    // Difficulty (难度) - 新增
    @NSManaged var difficulty: String?
    @NSManaged var difficultyConfidence: Double

    // Specificity (具体程度) - 新增
    @NSManaged var specificity: String?
    @NSManaged var specificityConfidence: Double

    // 分析时间
    @NSManaged var analyzedAt: Date?

    // Embedding (可选，用于相似度搜索)
    @NSManaged var embedding: Data?

    // MARK: - 用户纠正字段（用于设备端训练）

    // 原有纠正字段
    @NSManaged var categoryUserCorrected: String?
    @NSManaged var sentimentUserCorrected: String?

    // 新增纠正字段
    @NSManaged var urgencyUserCorrected: String?
    @NSManaged var timeFrameUserCorrected: String?
    @NSManaged var actionTypeUserCorrected: String?
    @NSManaged var difficultyUserCorrected: String?
    @NSManaged var specificityUserCorrected: String?

    @NSManaged var correctedAt: Date?
    @NSManaged var isTrainingSample: Bool

    static func todayString() -> String {
        return dateStringFrom(Date())
    }

    // MARK: - 有效值访问器（优先用户纠正）

    /// 获取最终分类（优先用户纠正）
    var effectiveCategory: String? {
        return categoryUserCorrected ?? category
    }

    /// 获取最终情感（优先用户纠正）
    var effectiveSentiment: String? {
        return sentimentUserCorrected ?? sentiment
    }

    /// 获取最终紧急度（优先用户纠正）
    var effectiveUrgency: String? {
        return urgencyUserCorrected ?? urgency
    }

    /// 获取最终时间范围（优先用户纠正）
    var effectiveTimeFrame: String? {
        return timeFrameUserCorrected ?? timeFrame
    }

    /// 获取最终行动类型（优先用户纠正）
    var effectiveActionType: String? {
        return actionTypeUserCorrected ?? actionType
    }

    /// 获取最终难度（优先用户纠正）
    var effectiveDifficulty: String? {
        return difficultyUserCorrected ?? difficulty
    }

    /// 获取最终具体程度（优先用户纠正）
    var effectiveSpecificity: String? {
        return specificityUserCorrected ?? specificity
    }

    /// 检查是否有任何用户纠正
    var hasUserCorrections: Bool {
        return categoryUserCorrected != nil ||
            sentimentUserCorrected != nil ||
            urgencyUserCorrected != nil ||
            timeFrameUserCorrected != nil ||
            actionTypeUserCorrected != nil ||
            difficultyUserCorrected != nil ||
            specificityUserCorrected != nil
    }

    /// 日期字符串转Date
    static func dateFrom(_ dateString: String) -> Date? {
        return dateFormatter.date(from: dateString)
    }

    /// Date转日期字符串
    static func dateStringFrom(_ date: Date) -> String {
        let cal = Calendar.current
        let comps = cal.dateComponents([.year, .month, .day], from: date)
        guard let year = comps.year, let month = comps.month, let day = comps.day else {
            return dateFormatter.string(from: date)
        }
        return String(format: "%04d-%02d-%02d", year, month, day)
    }

    @nonobjc static func fetchRequest() -> NSFetchRequest<GoalEntry> {
        NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
    }
}
