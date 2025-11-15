## 文档理解
- 目标与约束：v0.1 仅验证核心习惯循环与留存，数据100%本地，无账户/云；v0.1 可临时加入匿名启动事件分析并可一键移除；冷启动到可输入<2秒。
- 功能范围（E1–E7）：
  - E1 引导：设置“早晨窗口”并获取通知权限，本地存储。
  - E2 承诺设备：引导末步长按3秒按钮以形成心理投入。
  - E3 核心输入：今日唯一目标，自动聚焦，退出自动保存。
  - E4 提醒：仅当今日未记录时，在“早晨窗口”期间推送一次本地通知；角标显示提醒；已记录清除角标。
  - E5 反馈（Streak）：从今天向后连续天数的计数与微交互动效。
  - E6 历史：极简倒序列表，无日历视图。
  - E7 Beta分析（v0.1专属）：仅“应用启动”事件，匿名安装ID，Opt-In；可移除封装。
- 非目标：无社交、无云、无复杂编辑/附件/日历；一次性付费及长期洞察在v1.0/v2.0阶段。

## v0.01 范围裁剪
- 交付内容：
  - 今日输入页（自动聚焦、退出即存）。
  - 历史列表（倒序）。
  - Streak计数与保存后动画。
  - 引导页：设置“早晨窗口”与通知授权；承诺设备长按。
  - 本地通知：每天“早晨窗口”起点触发一次；保存后清除角标与取消当天待触发通知。
- 暂缓项：
  - 轻量分析SDK实际接入（仅保留可插拔接口与Opt-In UI）。
  - 任何云/付费相关功能。

## 架构设计
- App层：`MorningGoalApp` 注入Core Data上下文，启动时评估是否显示引导或主界面（现有入口见 `MorningGoal/MorningGoal/MorningGoalApp.swift:10-16`）。
- 数据层：
  - `CoreData` 持久化容器（`DataController`），主/后台上下文，自动保存。
  - 实体：`GoalEntry`、`UserSettings`；唯一性约束保证每日仅一条记录。
- 领域服务：
  - `StreakService`：计算连续记录天数（时区安全、DST安全）。
  - `NotificationService`：UNUserNotificationCenter授权与调度；保存后撤销当日未触发通知；角标与徽章更新。
- 表现层（SwiftUI）：
  - `OnboardingView`（E1/E2/E7）：晨窗设置、通知授权、长按承诺、分析Opt-In。
  - `TodayInputView`（E3/E5）：今日目标输入、保存触发动画与Streak更新。
  - `HistoryListView`（E6）：按日期倒序展示。
  - `RootView`：根据是否有今日记录在输入/历史间切换（E3逻辑）。

## 数据模型（Core Data）
- `GoalEntry`：
  - `date`（Date，标准化到本地午夜；唯一性约束或`dateString`作为唯一键）。
  - `goalText`（String，纯文本）。
  - `lastUpdated`（Date）。
- `UserSettings`：
  - `morningStart`（DateComponents：小时/分钟）。
  - `morningEnd`（DateComponents：小时/分钟）。
  - `analyticsOptIn`（Bool，默认false）。
- 合并策略：`NSMergeByPropertyObjectTrump`，插入前先查找当日记录，确保单条。

## 关键逻辑
- 今日视图选择（E3）：
  - IF 今日无`GoalEntry` → 显示输入页并自动弹出键盘；ELSE → 显示历史列表。
- Streak计算（E5）：
  - 从今日开始向后迭代，按`Calendar.current`的天粒度检测是否连续存在记录；缺失则终止。保存后触发`withAnimation(.spring)`的计数跃迁。
- 提醒与角标（E4）：
  - 每天在`morningStart`创建重复日历触发器；在保存当日目标后移除当天未触发的pending通知，并将`badge`清零。
  - iOS仅支持数字徽章，不支持“点”样式 → 以`badge=1`近似满足“点”的意图。
- 引导与承诺（E1/E2）：
  - 三步引导：愿景说明→晨窗设置→长按承诺；长按≥3秒才生效。
- 分析封装（E7）：
  - 定义`AnalyticsClient`协议与`AnalyticsProvider`占位实现；仅在`analyticsOptIn==true`时记录一次`app_launch`事件；移除时只需替换占位实现。

## 性能与隐私
- 冷启动路径最短化：延迟初始化非关键服务；SwiftUI状态恢复直达输入框。
- 100% 本地：仅使用Core Data与本地通知；不引入任何网络请求。

## UI与交互原则
- 极简：单输入框、单列表、显著Streak数字；无富文本/附件/日历。
- 动效：保存后计数数字弹跳；其余动效节制，避免干扰。

## 验收映射（v0.01）
- E1：引导≤4屏，成功授权通知，晨窗设置持久化。
- E2：承诺按钮需长按≥3秒，状态记录。
- E3：冷启动到可输入<2秒，键盘自动弹出，退出自动保存。
- E4：每日一条本地通知（仅未记录时），徽章≈“点”，记录后清除。
- E5：Streak计数准确，保存后动效反馈。
- E6：倒序历史列表，UI极简。
- E7：Opt-In界面与可插拔分析封装（默认不发送）。

## 文件级改动计划
- 入口：调整 `MorningGoal/MorningGoal/MorningGoalApp.swift:10-16` 注入Core Data上下文与RootView路由。
- 视图：重构 `MorningGoal/MorningGoal/ContentView.swift:10-20` 为`RootView`，新增`OnboardingView`/`TodayInputView`/`HistoryListView`。
- 数据：新增Core Data模型文件与`DataController`、实体`NSManagedObject`子类。
- 服务：新增`StreakService`、`NotificationService`、`AnalyticsClient`封装。

## 里程碑
- M1（当天）：搭建Core Data与基础视图路由；实现今日输入与自动保存；历史列表初版；Streak计算。
- M2（次日）：引导与承诺设备；本地通知与徽章；保存后动效；Opt-In占位与可插拔分析接口。
- M3（收尾）：QA与性能核查；边界条件（时区、DST、重复记录）验证；准备TestFlight构建。