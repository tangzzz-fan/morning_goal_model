# 快速测试指南

## 如何运行模型质量测试

### 方法 1: 使用测试视图 (推荐)

1. 在 Xcode 中打开项目
2. 在任意视图中添加测试按钮:

```swift
#if DEBUG
Button("测试模型") {
    showModelTest = true
}
.sheet(isPresented: $showModelTest) {
    ModelQualityTestView()
}
#endif
```

3. 运行应用,点击"测试模型"按钮
4. 查看测试结果

### 方法 2: 使用代码直接测试

在任意位置添加:

```swift
Task {
    let context = DataController.shared.container.viewContext
    let result = await ModelBench.run(in: context)
    print("分类准确率: \(result.catAcc)")
    print("情感准确率: \(result.senAcc)")
    print("平均推理时间: \(result.avg)ms")
}
```

### 方法 3: 查看控制台日志

运行应用后,在 Xcode Console 中搜索:
- `bench_complete` - 查看性能指标
- `bench_accuracy` - 查看准确率
- `bench_resources` - 查看资源使用

## 评估标准

✅ **优秀** (>= 90分):
- 分类准确率 >= 80%
- 情感准确率 >= 75%
- 推理时间 < 100ms

⚠️ **需改进** (< 70分):
- 考虑重新训练模型
- 检查训练数据质量
- 优化模型配置

## 测试数据

测试使用 24 个标准样本,覆盖:
- 8 个分类: 工作、健康、家庭、学习、财务、社交、休闲、个人发展
- 3 种情感: 积极、中性、消极
- 不同难度级别

详细信息请查看 `Model_Quality_Testing_Guide.md`
