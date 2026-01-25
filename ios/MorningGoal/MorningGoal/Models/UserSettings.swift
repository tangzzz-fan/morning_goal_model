import CoreData
import Foundation

final class UserSettings: NSManagedObject {
    @NSManaged var morningStartHour: Int16
    @NSManaged var morningStartMinute: Int16
    @NSManaged var morningEndHour: Int16
    @NSManaged var morningEndMinute: Int16
    @NSManaged var analyticsOptIn: Bool
    @NSManaged var committed: Bool

    static func fetchOrCreate(in context: NSManagedObjectContext) -> UserSettings {
        let req = NSFetchRequest<UserSettings>(entityName: "UserSettings")
        req.fetchLimit = 1
        if let existing = try? context.fetch(req).first { return existing }
        let obj = UserSettings(context: context)
        obj.morningStartHour = 7
        obj.morningStartMinute = 0
        obj.morningEndHour = 9
        obj.morningEndMinute = 0
        obj.analyticsOptIn = false
        obj.committed = false
        return obj
    }

    func isWithinMorningWindow(now: Date = Date()) -> Bool {
        let cal = Calendar.current
        let comps = cal.dateComponents([.year, .month, .day], from: now)
        var start = comps
        start.hour = Int(morningStartHour)
        start.minute = Int(morningStartMinute)
        var end = comps
        end.hour = Int(morningEndHour)
        end.minute = Int(morningEndMinute)
        guard let startDate = cal.date(from: start), let endDate = cal.date(from: end) else { return false }
        if startDate <= endDate { return now >= startDate && now <= endDate }
        return now >= startDate || now <= endDate
    }
}
