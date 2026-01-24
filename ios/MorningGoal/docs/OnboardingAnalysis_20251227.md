# Onboarding 流程分析与改进建议

## 1. 现状分析

通过分析 `OnboardingView.swift` 及其子组件，当前应用的引导流程主要包含以下四个步骤：

1.  **Value Proposition (价值主张)**: 简单的欢迎标题和统计数据（如果有）。
2.  **Morning Window (晨间窗口)**: 设置开始和结束时间。
3.  **Commitment (承诺)**: 交互式的承诺确认。
4.  **Permissions (权限)**: 请求通知权限。

### 用户反馈问题
用户反馈 "没有能很好的知道这个 app 是干啥的, 有什么功能"。

### 核心问题定位
*   **信息密度不足**: `ValuePropositionStep` 过于单薄，可能只展示了一个标题，缺乏对应用核心功能（目标设定、习惯追踪、AI 洞察）的直观介绍。
*   **直接进入设置**: 用户在未完全理解产品价值前，就被要求设置 "Morning Window"，造成认知门槛。
*   **缺乏场景化引导**: 没有通过截图或动效展示实际的使用场景（如：每天早上花 1 分钟输入目标）。

## 2. 改进建议

建议将 Onboarding 流程重构为 "价值探索" + "个性化设置" 的混合模式。

### 2.1 新增 "功能介绍轮播" (Feature Carousel)
在进入设置之前，增加 3 页滑动介绍，清晰阐述 APP 的核心价值：

*   **页面 1: 专注当下 (Focus)**
    *   **文案**: "每天清晨，一件事。"
    *   **副标**: "排除干扰，只设定今日最重要的一个目标。"
    *   **视觉**: 展示 `TodayInputView` 的简洁界面，强调极简输入。

*   **页面 2: 建立连胜 (Streak)**
    *   **文案**: "保持连胜，见证坚持。"
    *   **副标**: "看着火焰点燃，让自律成为一种习惯。"
    *   **视觉**: 展示 `StreakCounterView` 的火焰动效和数字增长。

*   **页面 3: 历史回顾 (Review)**
    *   **文案**: "时光倒流，回顾足迹。"
    *   **副标**: "查看 '历史上的今天'，见证你的成长轨迹。"
    *   **视觉**: 展示 `HistoryListView` 或 `OnThisDayCardView`。

### 2.2 优化 Value Proposition Step
将现有的 `ValuePropositionStep` 改造为上述轮播图的容器。

*   **交互**: 用户左右滑动查看，最后一张显示 "开始体验" 按钮。
*   **数据驱动**: 如果是老用户重装（有 iCloud 数据），可以在第一页动态显示 "欢迎回来，你已经记录了 X 个目标"。

### 2.3 调整引导顺序
建议顺序调整为：
1.  **功能介绍轮播** (解决 "不知道是干啥的")
2.  **Morning Window 设置** (基于 "专注当下" 的功能介绍，用户更理解为什么要设置时间)
3.  **Permissions** (解释：为了在 Morning Window 期间提醒你)
4.  **Commitment** (最后的仪式感)

## 3. 技术实现方案

### 修改 `OnboardingView.swift`
*   引入 `TabView` (PageStyle) 来实现轮播。
*   创建新的 View 组件 `FeatureSlideView`。

```swift
struct FeatureSlideView: View {
    let image: String // 或 Image View
    let title: LocalizedStringKey
    let subtitle: LocalizedStringKey
    // ...
}
```

### 资源需求
*   需要设计 3 张核心功能的示意图或录屏 GIF。
*   新增对应的多语言文案。

## 4. 预期效果
*   用户在首屏即可明确 APP 的三大核心功能：**设定目标、保持连胜、回顾历史**。
*   通过视觉化的演示，降低用户的理解成本。
*   提升用户完成 Onboarding 的转化率和后续留存。
