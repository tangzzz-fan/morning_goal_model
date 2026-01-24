import CoreData
import Foundation

enum MockDataService {
    static func addMockData(to context: NSManagedObjectContext) {
        let calendar = Calendar.current
        let today = Date()

        // 1. 创建过去一周的数据（保持原有逻辑）
        for dayOffset in 0 ..< 7 {
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: today) else { continue }
            createMockEntry(date: date, text: mockText(for: dayOffset), in: context)
        }

        // 2. 增加一个月前的数据（用于测试“历史上的今天”）
        if let oneMonthAgo = calendar.date(byAdding: .month, value: -1, to: today) {
            createMockEntry(date: oneMonthAgo, text: "一个月前的今天：开始构思这个项目，充满激情！", in: context)
        }

        // 3. 增加一周前的数据（用于测试“历史上的今天”）
        if let oneWeekAgo = calendar.date(byAdding: .day, value: -7, to: today) {
            createMockEntry(date: oneWeekAgo, text: "一周前的今天：还在调整 UI 细节。", in: context)
        }

        // 4. 增加一年前的数据（用于测试“历史上的今天”）
        if let oneYearAgo = calendar.date(byAdding: .year, value: -1, to: today) {
            createMockEntry(date: oneYearAgo, text: "一年前的今天：第一次尝试 SwiftUI，感觉很神奇。", in: context)
        }

        // 保存到数据库
        do {
            try context.save()
            print("✅ Mock data added successfully (including 1 week, 1 month, 1 year ago)")
        } catch {
            print("❌ Error saving mock data: \(error)")
        }
    }

    private static func createMockEntry(date: Date, text: String, in context: NSManagedObjectContext) {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let dateString = dateFormatter.string(from: date)

        let fetchRequest = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        fetchRequest.predicate = NSPredicate(format: "dateString == %@", dateString)
        fetchRequest.fetchLimit = 1

        if (try? context.fetch(fetchRequest).first) == nil {
            let entry = GoalEntry(context: context)
            entry.dateString = dateString
            entry.lastUpdated = date
            entry.goalText = text
        }
    }

    private static func mockText(for dayOffset: Int) -> String {
        switch dayOffset {
        case 0: return "完成今天的核心功能开发，确保代码质量"
        case 1: return "优化用户界面交互，提升用户体验"
        case 2: return "修复已知的bug，完善测试用例"
        case 3: return "学习新的iOS开发技术，提升技能"
        case 4: return "整理项目文档，更新技术说明"
        case 5: return "与团队沟通项目进展，协调工作安排"
        case 6: return "回顾本周工作，制定下周计划"
        default: return "完成今日目标，保持专注"
        }
    }

    static func clearAllData(from context: NSManagedObjectContext) {
        let fetchRequest = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        let entries = try? context.fetch(fetchRequest)

        entries?.forEach { context.delete($0) }

        do {
            try context.save()
            print("✅ All data cleared successfully")
        } catch {
            print("❌ Error clearing data: \(error)")
        }
    }

    /// 重设今天状态 - 删除今天的目标记录，让用户可以重新输入
    static func resetToday(in context: NSManagedObjectContext) {
        let todayString = GoalEntry.todayString()
        let fetchRequest = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        fetchRequest.predicate = NSPredicate(format: "dateString == %@", todayString)
        fetchRequest.fetchLimit = 1

        if let todayEntry = try? context.fetch(fetchRequest).first {
            context.delete(todayEntry)
            do {
                try context.save()
                print("✅ Today's entry deleted successfully")
            } catch {
                print("❌ Error deleting today's entry: \(error)")
            }
        }
    }

    /// 设置连续记录天数 - 通过添加连续的历史数据
    static func setStreak(days: Int, in context: NSManagedObjectContext) {
        let calendar = Calendar.current
        let today = Date()

        // 先清除现有数据
        clearAllData(from: context)

        // 从昨天开始往回添加连续天数（不包括今天，让用户可以输入今天的目标）
        for dayOffset in 1 ... days {
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: today) else { continue }

            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd"
            let dateString = dateFormatter.string(from: date)

            let entry = GoalEntry(context: context)
            entry.dateString = dateString
            entry.lastUpdated = date

            // 根据天数设置不同的目标内容
            let sampleGoals = [
                "完成核心功能开发",
                "优化用户界面交互",
                "修复已知bug",
                "学习新技术",
                "整理项目文档",
                "与团队沟通协调",
                "回顾并制定计划"
            ]
            entry.goalText = sampleGoals[dayOffset % sampleGoals.count]
        }

        // 保存到数据库
        do {
            try context.save()
            print("✅ Streak set to \(days) days successfully")
        } catch {
            print("❌ Error setting streak: \(error)")
        }
    }

    /// 添加历史上的今天数据 - 用于测试"历史上的今天"功能
    /// - Parameter yearsAgo: 添加几年前的今天的数据（1-5年）
    static func addOnThisDayData(yearsAgo: Int, in context: NSManagedObjectContext) {
        let calendar = Calendar.current
        let today = Date()

        for year in 1 ... yearsAgo {
            // 计算N年前的今天
            guard let pastDate = calendar.date(byAdding: .year, value: -year, to: today) else { continue }

            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd"
            let dateString = dateFormatter.string(from: pastDate)

            // 检查是否已存在
            let fetchRequest = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
            fetchRequest.predicate = NSPredicate(format: "dateString == %@", dateString)
            fetchRequest.fetchLimit = 1

            if (try? context.fetch(fetchRequest).first) != nil {
                // 如果已存在，跳过
                print("⚠️ Entry for \(dateString) already exists, skipping")
                continue
            }

            let entry = GoalEntry(context: context)
            entry.dateString = dateString
            entry.lastUpdated = pastDate

            // 根据年份设置不同的目标内容
            let onThisDayGoals = [
                "\(year)年前的今天：专注于个人成长和学习",
                "\(year)年前的今天：完成重要的项目里程碑",
                "\(year)年前的今天：建立健康的生活习惯",
                "\(year)年前的今天：花时间与家人朋友相处",
                "\(year)年前的今天：突破自己的舒适区"
            ]
            entry.goalText = onThisDayGoals[year % onThisDayGoals.count]
        }

        // 保存到数据库
        do {
            try context.save()
            print("✅ On This Day data added for \(yearsAgo) year(s) ago")
        } catch {
            print("❌ Error adding On This Day data: \(error)")
        }
    }

    /// 获取历史上的今天的记录（包括1周前、1月前、几年前）
    /// - Returns: 历史记录列表
    static func fetchOnThisDayEntries(in context: NSManagedObjectContext) -> [GoalEntry] {
        let calendar = Calendar.current
        let today = Date()

        print("🔍 开始查询历史回顾数据...")

        var entries: [GoalEntry] = []
        var datesToCheck: [(date: Date, label: String)] = []

        // 1. 检查1周前
        if let weekAgo = calendar.date(byAdding: .day, value: -7, to: today) {
            datesToCheck.append((weekAgo, "1周前"))
        }

        // 2. 检查1月前
        if let monthAgo = calendar.date(byAdding: .month, value: -1, to: today) {
            datesToCheck.append((monthAgo, "1月前"))
        }

        // 3. 检查过去5年
        for yearsAgo in 1 ... 5 {
            if let pastDate = calendar.date(byAdding: .year, value: -yearsAgo, to: today) {
                datesToCheck.append((pastDate, "\(yearsAgo)年前"))
            }
        }

        for (date, label) in datesToCheck {
            let dateString = GoalEntry.dateStringFrom(date)
            // print("  查询 \(label): \(dateString)")

            let req = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
            req.predicate = NSPredicate(format: "dateString == %@", dateString)
            req.fetchLimit = 1

            if let entry = try? context.fetch(req).first {
                entries.append(entry)
                print("  ✅ 找到 \(label) 的记录: \(entry.goalText)")
            }
        }

        return entries
    }
}
