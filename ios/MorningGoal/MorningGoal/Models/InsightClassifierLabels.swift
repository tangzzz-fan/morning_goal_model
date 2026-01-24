//
//  InsightClassifierLabels.swift
//  MorningGoal
//
//  洞察分类器标签定义
//  与训练数据索引保持一致
//

import Foundation

/// 洞察分类器标签定义
enum InsightLabels {
    // MARK: - Topic (主题分类) - 16 categories

    enum Topic: String, CaseIterable, Identifiable {
        case work = "工作"
        case health = "健康"
        case family = "家庭"
        case personalDevelopment = "个人发展"
        case finance = "理财"
        case social = "社交"
        case housework = "家务"
        case learning = "学习"
        case sleep = "睡眠"
        case diet = "饮食"
        case mindset = "心态"
        case entertainment = "娱乐"
        case travel = "出行"
        case career = "职业发展"
        case communication = "沟通"
        case parenting = "育儿"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .work: return "briefcase.fill"
            case .health: return "heart.fill"
            case .family: return "house.fill"
            case .personalDevelopment: return "person.fill.checkmark"
            case .finance: return "dollarsign.circle.fill"
            case .social: return "person.2.fill"
            case .housework: return "washer.fill"
            case .learning: return "book.fill"
            case .sleep: return "bed.double.fill"
            case .diet: return "fork.knife"
            case .mindset: return "brain.head.profile"
            case .entertainment: return "gamecontroller.fill"
            case .travel: return "car.fill"
            case .career: return "chart.line.uptrend.xyaxis"
            case .communication: return "bubble.left.and.bubble.right.fill"
            case .parenting: return "figure.and.child.holdinghands"
            }
        }

        var index: Int {
            switch self {
            case .work: return 0
            case .health: return 1
            case .family: return 2
            case .personalDevelopment: return 3
            case .finance: return 4
            case .social: return 5
            case .housework: return 6
            case .learning: return 7
            case .sleep: return 8
            case .diet: return 9
            case .mindset: return 10
            case .entertainment: return 11
            case .travel: return 12
            case .career: return 13
            case .communication: return 14
            case .parenting: return 15
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - Sentiment (情感) - 3 categories

    enum Sentiment: String, CaseIterable, Identifiable {
        case negative = "消极"
        case neutral = "中性"
        case positive = "积极"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .positive: return "face.smiling.fill"
            case .neutral: return "face.smiling"
            case .negative: return "cloud.rain.fill"
            }
        }

        var index: Int {
            switch self {
            case .negative: return 0
            case .neutral: return 1
            case .positive: return 2
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - Urgency (紧急度) - 3 categories

    enum Urgency: String, CaseIterable, Identifiable {
        case low = "低"
        case medium = "中"
        case high = "高"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .low: return "tortoise.fill"
            case .medium: return "hare.fill"
            case .high: return "flame.fill"
            }
        }

        var index: Int {
            switch self {
            case .low: return 0
            case .medium: return 1
            case .high: return 2
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - TimeFrame (时间范围) - 4 categories

    enum TimeFrame: String, CaseIterable, Identifiable {
        case today = "今天"
        case thisWeek = "本周"
        case thisMonth = "本月"
        case longTerm = "长期"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .today: return "sun.max.fill"
            case .thisWeek: return "calendar"
            case .thisMonth: return "calendar.badge.clock"
            case .longTerm: return "infinity"
            }
        }

        var index: Int {
            switch self {
            case .today: return 0
            case .thisWeek: return 1
            case .thisMonth: return 2
            case .longTerm: return 3
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - ActionType (行动类型) - 5 categories

    enum ActionType: String, CaseIterable, Identifiable {
        case learning = "学习"
        case exercise = "运动"
        case work = "工作"
        case lifestyle = "生活"
        case social = "社交"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .learning: return "book.fill"
            case .exercise: return "figure.run"
            case .work: return "briefcase.fill"
            case .lifestyle: return "house.fill"
            case .social: return "person.2.fill"
            }
        }

        var index: Int {
            switch self {
            case .learning: return 0
            case .exercise: return 1
            case .work: return 2
            case .lifestyle: return 3
            case .social: return 4
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - Difficulty (难度) - 3 categories

    enum Difficulty: String, CaseIterable, Identifiable {
        case easy = "简单"
        case moderate = "中等"
        case hard = "困难"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .easy: return "star"
            case .moderate: return "star.leadinghalf.filled"
            case .hard: return "star.fill"
            }
        }

        var index: Int {
            switch self {
            case .easy: return 0
            case .moderate: return 1
            case .hard: return 2
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - Specificity (具体程度) - 3 categories

    enum Specificity: String, CaseIterable, Identifiable {
        case vague = "模糊"
        case moderate = "一般"
        case specific = "具体"

        var id: String { rawValue }

        var displayName: String { rawValue }

        var icon: String {
            switch self {
            case .vague: return "cloud.fill"
            case .moderate: return "cloud.sun.fill"
            case .specific: return "scope"
            }
        }

        var index: Int {
            switch self {
            case .vague: return 0
            case .moderate: return 1
            case .specific: return 2
            }
        }

        static var labels: [String] { allCases.map { $0.rawValue } }
    }

    // MARK: - 辅助方法

    /// 获取所有分类器配置
    static var allDimensions: [(name: String, numClasses: Int, labels: [String])] {
        [
            ("topic", 16, Topic.labels),
            ("sentiment", 3, Sentiment.labels),
            ("urgency", 3, Urgency.labels),
            ("timeFrame", 4, TimeFrame.labels),
            ("actionType", 5, ActionType.labels),
            ("difficulty", 3, Difficulty.labels),
            ("specificity", 3, Specificity.labels)
        ]
    }

    /// 根据维度名称获取显示名称
    static func displayName(for dimension: String, label: String) -> String {
        // 直接返回标签，因为标签本身就是中文
        return label
    }

    /// 获取维度的中文名称
    static func dimensionDisplayName(_ dimension: String) -> String {
        switch dimension {
        case "topic": return "主题分类"
        case "sentiment": return "情感倾向"
        case "urgency": return "紧急度"
        case "timeFrame": return "时间范围"
        case "actionType": return "行动类型"
        case "difficulty": return "难度"
        case "specificity": return "具体程度"
        case "featureExtractor": return "特征提取器"
        default: return dimension
        }
    }

    /// 获取维度的图标
    static func dimensionIcon(_ dimension: String) -> String {
        switch dimension {
        case "topic": return "tag.fill"
        case "sentiment": return "heart.fill"
        case "urgency": return "exclamationmark.triangle.fill"
        case "timeFrame": return "clock.fill"
        case "actionType": return "figure.walk"
        case "difficulty": return "speedometer"
        case "specificity": return "target"
        case "featureExtractor": return "cpu.fill"
        default: return "questionmark.circle"
        }
    }
}
