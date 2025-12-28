# MorningGoal 测试使用文档

## 运行环境

- Xcode 15+/iOS 26 SDK
- 目标设备：iPhone 17 Pro Max 模拟器

## 运行测试

- IDE：在 Xcode 选择 Scheme `MorningGoal`，菜单 `Product > Test` 或按 `⌘U`
- 命令行：

```
xcodebuild -project "MorningGoal/MorningGoal.xcodeproj" -scheme MorningGoal -destination 'platform=iOS Simulator,name=iPhone 17 Pro Max' test
```

### 代码覆盖率

IDE：在 Scheme 的 Test 动作勾选 `Gather coverage for all targets`。

命令行：

```
xcodebuild -project "MorningGoal/MorningGoal.xcodeproj" -scheme MorningGoal -destination 'platform=iOS Simulator,name=iPhone 17 Pro Max' -enableCodeCoverage YES test
```

## 测试内容

- 单元测试：
  - StreakServiceTests：连续与断点计数
  - DataControllerTests：每日唯一、默认设置
  - NotificationServiceTests：调度与清除徽章
  - AnalyticsTests：Opt-In 行为
- 轻量集成测试：GoalFlowTests

## 注意事项

- 测试不依赖网络与真实通知权限；使用内存数据库与 Fake 依赖。
- 若模拟器名称或系统版本不同，替换命令行中的 `destination` 即可。
 - 若需泛化运行：`-destination 'generic/platform=iOS Simulator'`。