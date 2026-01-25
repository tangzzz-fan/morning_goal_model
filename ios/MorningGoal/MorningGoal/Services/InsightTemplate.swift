//
//  InsightTemplate.swift
//  MorningGoal
//
//  洞察模板系统 - Phase 5.4
//  支持预定义模板和变量填充
//

import Foundation

// MARK: - 模板变量

/// 模板变量类型
enum TemplateVariable: String, CaseIterable {
    case category = "{{category}}"
    case percentage = "{{percentage}}"
    case days = "{{days}}"
    case count = "{{count}}"
    case direction = "{{direction}}"
    case weekday = "{{weekday}}"
    case thisWeek = "{{thisWeek}}"
    case lastWeek = "{{lastWeek}}"
    case suggestedType = "{{suggestedType}}"
    case message = "{{message}}"
}

// MARK: - 洞察模板

/// 洞察模板
struct InsightTemplate {
    let id: String
    let titleTemplate: String
    let descriptionTemplate: String
    let icon: String
    let priority: InsightPriority

    /// 使用变量填充模板
    func render(variables: [TemplateVariable: String]) -> (title: String, description: String) {
        var title = titleTemplate
        var description = descriptionTemplate

        for (variable, value) in variables {
            title = title.replacingOccurrences(of: variable.rawValue, with: value)
            description = description.replacingOccurrences(of: variable.rawValue, with: value)
        }

        return (title, description)
    }
}

// MARK: - 预定义模板库

enum InsightTemplateLibrary {
    // MARK: - 成就类模板

    static let streakCelebration = InsightTemplate(
        id: "streak_celebration",
        titleTemplate: "🔥 连续记录",
        descriptionTemplate: "太棒了！你已经连续记录 {{days}} 天了！{{message}}",
        icon: "flame.fill",
        priority: .high
    )

    static let weeklyGoal = InsightTemplate(
        id: "weekly_goal",
        titleTemplate: "📈 本周成就",
        descriptionTemplate: "本周你记录了 {{thisWeek}} 个目标，比上周多 {{percentage}}%！",
        icon: "chart.bar.fill",
        priority: .medium
    )

    // MARK: - 分析类模板

    static let categoryFocus = InsightTemplate(
        id: "category_focus",
        titleTemplate: "🎯 主题聚焦",
        descriptionTemplate: "在过去30天里，你 {{percentage}}% 的目标与「{{category}}」相关。",
        icon: "chart.pie.fill",
        priority: .low
    )

    static let balanceSuggestion = InsightTemplate(
        id: "balance_suggestion",
        titleTemplate: "⚖️ 目标平衡",
        descriptionTemplate: "你 {{percentage}}% 的目标都是「{{category}}」类型。尝试添加一些「{{suggestedType}}」类型的目标来保持平衡！",
        icon: "scale.3d",
        priority: .high
    )

    static let weekdayInsight = InsightTemplate(
        id: "weekday_insight",
        titleTemplate: "📅 周期规律",
        descriptionTemplate: "你在{{weekday}}最活跃（{{count}}个目标）。",
        icon: "calendar.badge.clock",
        priority: .low
    )

    // MARK: - 趋势类模板

    static let sentimentUp = InsightTemplate(
        id: "sentiment_up",
        titleTemplate: "😊 情感上升",
        descriptionTemplate: "你的整体情感倾向呈现上升趋势，继续保持积极的心态！",
        icon: "face.smiling.fill",
        priority: .medium
    )

    static let sentimentDown = InsightTemplate(
        id: "sentiment_down",
        titleTemplate: "💪 关注自我",
        descriptionTemplate: "你的情感倾向最近有所下降，记得关爱自己！",
        icon: "heart.fill",
        priority: .medium
    )

    static let volumeIncrease = InsightTemplate(
        id: "volume_increase",
        titleTemplate: "🚀 活跃度提升",
        descriptionTemplate: "太棒了！你的目标记录活跃度较上周提升了 {{percentage}}%。",
        icon: "arrow.up.right.circle.fill",
        priority: .medium
    )

    static let volumeDecrease = InsightTemplate(
        id: "volume_decrease",
        titleTemplate: "📝 继续加油",
        descriptionTemplate: "你的目标记录活跃度较上周下降了 {{percentage}}%，继续加油！",
        icon: "hand.thumbsup.fill",
        priority: .medium
    )

    // MARK: - 鼓励类模板

    static let encouragementMessages: [String] = [
        "继续保持！",
        "你做得很棒！",
        "每一天都在进步！",
        "一周的坚持，了不起！",
        "习惯正在形成！",
        "你的毅力令人印象深刻！",
        "两周的坚持，太棒了！",
        "你已经是习惯大师了！",
        "持续的力量！",
        "传奇！超过一个月的坚持！",
        "你是真正的目标达人！",
        "无与伦比的毅力！"
    ]

    static let newUserWelcome = InsightTemplate(
        id: "new_user_welcome",
        titleTemplate: "👋 欢迎！",
        descriptionTemplate: "开始记录你的第一个目标，我们会帮助你发现有趣的模式！",
        icon: "hand.wave.fill",
        priority: .high
    )

    static let comeBack = InsightTemplate(
        id: "come_back",
        titleTemplate: "💫 欢迎回来",
        descriptionTemplate: "好久不见！今天是开始新习惯的好日子。",
        icon: "sparkles",
        priority: .high
    )

    // MARK: - 获取所有模板

    static var allTemplates: [InsightTemplate] {
        [
            streakCelebration,
            weeklyGoal,
            categoryFocus,
            balanceSuggestion,
            weekdayInsight,
            sentimentUp,
            sentimentDown,
            volumeIncrease,
            volumeDecrease,
            newUserWelcome,
            comeBack
        ]
    }

    /// 根据 ID 获取模板
    static func template(for id: String) -> InsightTemplate? {
        allTemplates.first { $0.id == id }
    }
}

// MARK: - 模板渲染扩展

extension Insight {
    /// 使用模板系统获取描述（可选，提供更灵活的本地化支持）
    func renderWithTemplate() -> (title: String, description: String) {
        switch self {
        case let .categoryDistribution(category, percentage):
            return InsightTemplateLibrary.categoryFocus.render(variables: [
                .category: category,
                .percentage: String(percentage)
            ])

        case let .balance(mainType, percentage, suggestedType):
            return InsightTemplateLibrary.balanceSuggestion.render(variables: [
                .category: mainType,
                .percentage: String(percentage),
                .suggestedType: suggestedType
            ])

        case let .encouragement(streakDays, message):
            return InsightTemplateLibrary.streakCelebration.render(variables: [
                .days: String(streakDays),
                .message: message
            ])

        case let .volumeTrend(direction, changePercent):
            if direction == "上升" {
                return InsightTemplateLibrary.volumeIncrease.render(variables: [
                    .percentage: String(changePercent)
                ])
            } else {
                return InsightTemplateLibrary.volumeDecrease.render(variables: [
                    .percentage: String(changePercent)
                ])
            }

        default:
            // 对于其他类型，使用内置的 title 和 description
            return (title, description)
        }
    }
}
