import CoreData
import Foundation

enum StreakService {
    static func count(in context: NSManagedObjectContext) -> Int {
        var count = 0
        let cal = Calendar.current
        var date = Date()

        // 辅助函数：检查指定日期是否有目标记录
        func checkExists(_ dateToCheck: Date) -> Bool {
            let key = GoalEntry.dateStringFrom(dateToCheck)
            let req = NSFetchRequest<NSFetchRequestResult>(entityName: "GoalEntry")
            req.predicate = NSPredicate(format: "dateString == %@", key)
            req.fetchLimit = 1
            return ((try? context.count(for: req)) ?? 0) > 0
        }

        // 标记是否是第一次检查（即今天）
        var isFirstCheck = true

        while true {
            let exists = checkExists(date)

            if exists {
                count += 1
            } else {
                // 如果是今天且没有记录，不中断 Streak，继续检查昨天
                // 这样可以显示"当前的连续记录"，即使今天还没打卡
                if !isFirstCheck {
                    break
                }
            }

            isFirstCheck = false
            guard let prev = cal.date(byAdding: .day, value: -1, to: date) else { break }
            date = prev
        }
        return count
    }
}
