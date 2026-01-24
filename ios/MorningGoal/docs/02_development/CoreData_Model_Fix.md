# Core Data 模型更新说明

## ✅ 已修复的问题

### 错误信息
```
CoreData: error: keypath isTrainingSample not found in entity GoalEntry
*** Terminating app due to uncaught exception 'NSInvalidArgumentException'
```

### 根本原因
`DataController.makeModel()` 中的 Core Data 模型定义缺少了以下字段:
- ❌ `category` - 分类结果
- ❌ `categoryConfidence` - 分类置信度
- ❌ `sentiment` - 情感结果
- ❌ `sentimentScore` - 情感分数
- ❌ `analyzedAt` - 分析时间
- ❌ `categoryUserCorrected` - 用户纠正的分类
- ❌ `sentimentUserCorrected` - 用户纠正的情感
- ❌ `correctedAt` - 纠正时间
- ❌ **`isTrainingSample`** - 是否为训练样本 (导致崩溃)

## ✅ 已完成的修复

### 更新的文件
**[DataController.swift](file:///Users/apple/Developments/MorningGoalPython/MorningGoal/MorningGoal/DataController.swift)**

### 添加的字段

```swift
// ML 分析结果字段
- category: String? (可选)
- categoryConfidence: Double (默认 0.0)
- sentiment: String? (可选)
- sentimentScore: Double (默认 0.0)
- analyzedAt: Date? (可选)

// 用户纠正字段（用于设备端训练）
- categoryUserCorrected: String? (可选)
- sentimentUserCorrected: String? (可选)
- correctedAt: Date? (可选)
- isTrainingSample: Bool (默认 false)
```

## ⚠️ 重要: 需要重新安装应用

### 为什么需要重新安装?

Core Data 模型结构已更改,现有的数据库与新模型不兼容。

### 解决方案

#### 方案 1: 删除应用重新安装 (推荐)

**模拟器**:
1. 在 Xcode 中停止应用
2. 长按应用图标 → 删除应用
3. 重新运行 (`Cmd + R`)

**真机**:
1. 长按应用图标 → 删除应用
2. 在 Xcode 中重新安装

#### 方案 2: 清除应用数据

**模拟器**:
```bash
# 重置模拟器
xcrun simctl erase all
```

**真机**:
- 设置 → 通用 → iPhone 储存空间 → Morning Goal → 删除 App

#### 方案 3: 添加数据迁移 (生产环境)

如果应用已发布,需要实现 Core Data 轻量级迁移:

```swift
// 在 DataController.init() 中
container.loadPersistentStores { description, error in
    if let error = error {
        // 处理迁移错误
        fatalError("Core Data migration failed: \(error)")
    }
}

// 启用自动迁移
let description = container.persistentStoreDescriptions.first
description?.shouldMigrateStoreAutomatically = true
description?.shouldInferMappingModelAutomatically = true
```

## 🎯 验证修复

### 步骤 1: 删除应用
删除模拟器或真机上的应用

### 步骤 2: 重新运行
在 Xcode 中按 `Cmd + R` 重新运行

### 步骤 3: 打开调试界面
1. 点击右下角齿轮图标
2. 选择 "🔮 模型调试界面"

### 步骤 4: 验证功能
应该能够正常:
- ✅ 查看模型信息
- ✅ 运行预测
- ✅ 保存纠正
- ✅ 查看样本数量
- ✅ 触发训练

### 预期日志

**应用启动**:
```
🚀 [INIT] Starting AdaptiveModelService initialization...
✅ [INIT] AdaptiveModelService initialization COMPLETE
```

**保存纠正后**:
```
✅ 已保存纠正: 工作 / 积极
```

**查询样本数量**:
```
待训练样本: 1 个
```

## 📊 Core Data 模型完整结构

### GoalEntry 实体

| 字段名 | 类型 | 可选 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `dateString` | String | ❌ | - | 日期字符串 (唯一) |
| `goalText` | String | ❌ | - | 目标文本 |
| `lastUpdated` | Date | ❌ | Date() | 最后更新时间 |
| `category` | String | ✅ | nil | ML 预测分类 |
| `categoryConfidence` | Double | ❌ | 0.0 | 分类置信度 |
| `sentiment` | String | ✅ | nil | ML 预测情感 |
| `sentimentScore` | Double | ❌ | 0.0 | 情感分数 |
| `analyzedAt` | Date | ✅ | nil | 分析时间 |
| `categoryUserCorrected` | String | ✅ | nil | 用户纠正的分类 |
| `sentimentUserCorrected` | String | ✅ | nil | 用户纠正的情感 |
| `correctedAt` | Date | ✅ | nil | 纠正时间 |
| `isTrainingSample` | Bool | ❌ | false | 是否为训练样本 |

## 🔍 故障排查

### 问题: 仍然崩溃

**检查**:
1. 确认已完全删除应用
2. 清理构建 (`Cmd + Shift + K`)
3. 重新构建 (`Cmd + B`)
4. 重新运行 (`Cmd + R`)

### 问题: 数据丢失

**说明**: 
- 删除应用会清除所有本地数据
- 这是正常的,因为模型结构已更改
- 生产环境需要实现数据迁移

### 问题: 模拟器问题

**解决**:
```bash
# 完全重置所有模拟器
xcrun simctl erase all

# 或重置特定模拟器
xcrun simctl erase <UDID>
```

## 📝 相关文档

- Core Data 控制器: [DataController.swift](file:///Users/apple/Developments/MorningGoalPython/MorningGoal/MorningGoal/DataController.swift)
- 实体定义: [GoalEntry.swift](file:///Users/apple/Developments/MorningGoalPython/MorningGoal/MorningGoal/Models/GoalEntry.swift)
- 调试界面: [ModelDebugView.swift](file:///Users/apple/Developments/MorningGoalPython/MorningGoal/MorningGoal/Views/ModelDebugView.swift)

## 🚀 下一步

1. ✅ 删除应用
2. ✅ 重新运行 (`Cmd + R`)
3. ✅ 打开调试界面
4. ✅ 测试预测功能
5. ✅ 保存纠正
6. ✅ 验证样本数量
7. ✅ 触发训练

现在应该可以正常使用了! 🎉
