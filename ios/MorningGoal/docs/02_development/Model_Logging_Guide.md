# 模型日志查看指南

## 概述

已在 `AdaptiveModelService` 中添加了详细的日志记录,帮助你追踪模型的加载、预测和训练过程。

## 日志类别

所有日志使用 emoji 前缀便于识别:

### 🚀 [INIT] - 模型初始化
- 模型文件搜索路径
- 加载的模型类型(Bundle 或 Documents)
- 元数据检测
- Tokenizer 加载

### 🔮 [PREDICT] - 预测操作
- 输入文本
- Tokenization 过程
- 推理时间
- 预测结果(分类和情感)

### 🎓 [TRAIN] - 端侧训练
- 训练样本数量和内容
- MLUpdateTask 状态
- 模型保存位置
- 重新加载确认

## 如何查看日志

### 方法 1: Xcode Console (推荐)

1. **运行应用**: 在 Xcode 中按 `Cmd + R` 运行应用
2. **打开 Console**: 按 `Cmd + Shift + Y` 打开底部控制台
3. **过滤日志**: 在搜索框中输入:
   - `[INIT]` - 查看初始化日志
   - `[PREDICT]` - 查看预测日志
   - `[TRAIN]` - 查看训练日志
   - `adaptive` - 查看所有模型相关日志

### 方法 2: Console.app

1. **打开 Console.app**: 在 macOS 中打开 "控制台" 应用
2. **选择设备**: 选择你的模拟器或真机
3. **过滤**: 在搜索框中输入 `com.morninggoal.app`
4. **查看**: 实时查看所有日志输出

### 方法 3: 命令行

```bash
# 实时查看模拟器日志
xcrun simctl spawn booted log stream --predicate 'subsystem == "com.morninggoal.app"'

# 过滤特定类别
xcrun simctl spawn booted log stream --predicate 'subsystem == "com.morninggoal.app" AND category == "adaptive"'
```

## 日志示例

### 应用启动时

```
🚀 [INIT] Starting AdaptiveModelService initialization...
📂 [INIT] Documents directory: /Users/.../Documents
🔍 [INIT] Checking for updated model at: .../student_sequence_classification.mlmodelc
📦 [INIT] No updated model found, loading bundled model...
📦 [INIT] Found bundled model at: .../student_sequence_classification.mlpackage
✅ [INIT] Successfully loaded BUNDLED model from: .../
🔧 [INIT] Detecting model metadata...
✅ [INIT] Model metadata detected: shape=[1, 128], maxLength=128
📖 [INIT] Loading tokenizer vocabulary...
📖 [INIT] Found vocab at: .../vocab.txt
✅ [INIT] Tokenizer initialized successfully
🎉 [INIT] AdaptiveModelService initialization COMPLETE
📊 [INIT] Model config: maxLength=128, shape=[1, 128], computeUnits=cpuAndNeuralEngine
```

### 用户输入目标时

```
🔮 [PREDICT] Starting goal analysis...
📝 [PREDICT] Input text: '今天要完成项目开发工作'
🔤 [PREDICT] Tokenizing with maxLength=128...
✅ [PREDICT] Tokenization complete: 12 tokens
🔢 [PREDICT] Token IDs (first 10): [101, 791, 1921, 6206, 2130, 2768, 7555, 4500, ...]
🔧 [PREDICT] Creating MultiArrays with shape=[1, 128]...
⚡ [PREDICT] Running model inference...
✅ [PREDICT] Inference complete in 45.23ms
🎯 [PREDICT] Prediction result: category='工作' (conf=0.892), sentiment='中性' (score=0.756)
```

### 端侧训练时

```
🎓 [TRAIN] Starting on-device training...
📊 [TRAIN] Training samples count: 15
📝 [TRAIN] Sample 1: text='今天要完成项目开发工作...', category='工作', sentiment='积极'
📝 [TRAIN] Sample 2: text='晚上去健身房锻炼...', category='健康', sentiment='积极'
📝 [TRAIN] Sample 3: text='和家人一起吃饭...', category='家庭', sentiment='中性'
🔧 [TRAIN] Preparing batch provider...
📂 [TRAIN] Source model: .../student_sequence_classification.mlpackage
💾 [TRAIN] Target model path: .../Documents/student_sequence_classification.mlmodelc
⚡ [TRAIN] Starting MLUpdateTask...
▶️ [TRAIN] MLUpdateTask created and resumed
✅ [TRAIN] Training completed successfully
💾 [TRAIN] Updated model saved to: .../Documents/student_sequence_classification.mlmodelc
🔄 [TRAIN] Reloading updated model...
✅ [TRAIN] Model reloaded successfully, now using updated model
🎉 [TRAIN] On-device training complete!
```

## 验证模型使用

### 检查点 1: 应用启动
✅ 查看是否有 `🎉 [INIT] AdaptiveModelService initialization COMPLETE`
✅ 确认加载的是哪个模型 (Bundle 或 Documents)

### 检查点 2: 用户输入目标
✅ 每次用户保存目标时应该看到 `🔮 [PREDICT] Starting goal analysis...`
✅ 查看预测结果是否合理

### 检查点 3: 端侧训练后
✅ 训练完成后应该看到 `✅ [TRAIN] Model reloaded successfully, now using updated model`
✅ 下次预测应该使用更新后的模型

## 故障排查

### 问题: 没有看到 [INIT] 日志
**原因**: 模型服务未初始化
**解决**: 检查应用是否正常启动

### 问题: 没有看到 [PREDICT] 日志
**原因**: 模型未被调用
**解决**: 检查应用逻辑,确认是否调用了 `analyzeGoal`

### 问题: 看到 ❌ [INIT] FATAL: Model file not found
**原因**: 模型文件未添加到 Bundle
**解决**: 在 Xcode 中确认 `student_sequence_classification.mlpackage` 已添加到 target

### 问题: [PREDICT] 推理时间过长 (>500ms)
**原因**: 可能在模拟器上运行或模型过大
**解决**: 在真机上测试,或优化模型

## 性能监控

通过日志可以监控:
- **推理时间**: 每次预测的 `Inference complete in XXms`
- **模型来源**: 是否使用了训练后的模型
- **训练频率**: 查看 `[TRAIN]` 日志的出现频率
- **准确率**: 观察预测结果是否符合预期

## 下一步

1. ✅ 运行应用并查看初始化日志
2. ✅ 输入几个目标,观察预测日志
3. ✅ 触发端侧训练,观察训练日志
4. ✅ 验证训练后模型是否被正确加载和使用
