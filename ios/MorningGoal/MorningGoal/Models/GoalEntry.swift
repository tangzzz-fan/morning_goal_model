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

    // 原有字段
    @NSManaged var dateString: String
    @NSManaged var goalText: String
    @NSManaged var lastUpdated: Date

    // 分析结果字段
    @NSManaged var category: String?
    @NSManaged var categoryConfidence: Double
    @NSManaged var sentiment: String?
    @NSManaged var sentimentScore: Double
    @NSManaged var analyzedAt: Date?

    // 用户纠正字段（用于设备端训练）
    @NSManaged var categoryUserCorrected: String?
    @NSManaged var sentimentUserCorrected: String?
    @NSManaged var correctedAt: Date?
    @NSManaged var isTrainingSample: Bool

    static func todayString() -> String {
        return dateStringFrom(Date())
    }

    /// 获取最终分类（优先用户纠正）
    var effectiveCategory: String? {
        return categoryUserCorrected ?? category
    }

    /// 获取最终情感（优先用户纠正）
    var effectiveSentiment: String? {
        return sentimentUserCorrected ?? sentiment
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
