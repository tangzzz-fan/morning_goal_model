# 项目改动记录 (2025-12-27)

## 1. Onboarding 流程重构

### 核心改动
- **结构扁平化**：将原本嵌套的 `FeatureCarouselStep` 拆解为 3 个独立的线性页面，形成了全新的 6 页线性流程。
- **流程定义**：
  1. **Theory (page1)**: 强调 "42% 达成率" 的心理学研究背景。
  2. **Focus (page2)**: 介绍极简主义，每天专注一件事。
  3. **Streak (page3)**: 引入自然生长和连胜概念。
  4. **MorningWindow**: 设定早晨窗口时间。
  5. **Commitment**: 长按承诺交互。
  6. **Permissions**: 开启通知权限。
- **代码重构**：
  - 更新 `OnboardingStep` 枚举，使用语义化命名 (`theory`, `focus`, `streak`, `morningWindow`, `commitment`, `permissions`)。
  - 在 `OnboardingView` 中移除了 `FeatureCarouselStep` 嵌套逻辑，直接在 `TabView` 中渲染 `FeatureSlideView`。
  - 统一了所有步骤（包括非核心步骤）在 `TabView` 中的索引安全检查逻辑。

### 相关文件
- `MorningGoal/Views/OnboardingView.swift`: 核心逻辑重构。
- `MorningGoal/Services/OnboardingContentService.swift`: 适配新的线性流程文案。
- `zh-Hans.lproj/Localizable.strings`: 新增 6 页流程的本地化文案，移除“自律”相关措辞。

## 2. "历史上的今天" (On This Day) 功能增强

### 核心改动
- **多维度回顾**：支持回顾 **1周前**、**1月前** 和 **N年前** 的数据（原逻辑仅支持年份）。
- **展示逻辑优化**：
  - **触发机制**：用户完成今日目标保存后，有 30% 概率触发。
  - **交互升级**：在输入框上方弹出 `OnThisDayMiniCard`，点击可全屏展开 `OnThisDayCardView`。
  - **调试支持**：在 `TodayInputView` 中添加了**双击标题**强制触发回顾卡片的隐藏手势。
- **UI 组件更新**：
  - `OnThisDayCardView`: 标题逻辑重构，支持动态显示 "1周前的今天"、"1月前的今天" 或 "N年前的今天"。
  - `OnThisDayMiniCard`: 同步支持多维度时间文案。

### 相关文件
- `MorningGoal/Services/MockDataService.swift`: 新增生成一周前、一月前 Mock 数据的逻辑；更新查询逻辑。
- `MorningGoal/Views/TodayInputView.swift`: 集成 Mini 卡片，添加点击全屏展示逻辑及双击调试手势。
- `MorningGoal/Views/OnThisDayCardView.swift`: 修复标题文案逻辑，适配非年份的时间跨度。

## 3. 视觉与体验优化

### 核心改动
- **Settings 页面修复**：
  - 修复了深色模式下 `HistoryListView` 中 Settings 弹窗文字颜色不可见的问题（显式指定 `Color.Design.softWhite`）。
- **文案调整**：
  - 全局移除了“自律”相关的压力性词汇，转而强调“自然生长”、“极简”和“无需刻意坚持”。
  - 在 `TodayInputView` 底部增加了 "每天30秒，提高42%的目标实现率" 的心理学提示。

### 相关文件
- `MorningGoal/Views/HistoryListView.swift`: 修复 Settings 列表样式。
- `zh-Hans.lproj/Localizable.strings`: 文案去“自律”化，增强心理学暗示。

## 4. 测试与调试

### 新增能力
- **Mock 数据增强**：Debug Menu 现在可以生成一周前、一月前的测试数据，方便验证回顾功能。
- **交互调试**：无需反复输入数据，通过双击首页标题即可快速验证“历史上的今天”卡片弹出效果。

---

*此文档旨在记录本次迭代的关键架构调整与功能增强，便于后续维护与回溯。*
