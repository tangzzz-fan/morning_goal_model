# 代码质量工具配置指南

## 概述

本项目已配置 SwiftLint 和 SwiftFormat 来确保代码质量和格式一致性。这些工具会在每次构建时自动运行。

## 已完成的配置

✅ **SPM 依赖管理**
- 已通过 Swift Package Manager 添加 SwiftLint 和 SwiftFormat 包
- SwiftLint: v0.55.0+
- SwiftFormat: v0.54.0+

✅ **Build Phase 脚本**
- SwiftLint 脚本（在 Sources 阶段之前运行）
- SwiftFormat 脚本（在 Sources 阶段之前运行）

✅ **配置文件**
- `.swiftlint.yml` - SwiftLint 规则配置
- `.swiftformat` - SwiftFormat 格式化配置

## 安装命令行工具（推荐）

虽然项目已通过 SPM 集成，但建议安装命令行工具以便手动执行格式化：

### 方式一：使用 Homebrew（推荐）

```bash
# 安装 SwiftLint
brew install swiftlint

# 安装 SwiftFormat
brew install swiftformat
```

### 方式二：使用 Mint

```bash
# 安装 Mint（如果尚未安装）
brew install mint

# 安装 SwiftLint
mint install realm/SwiftLint

# 安装 SwiftFormat
mint install nicklockwood/SwiftFormat
```

## 使用方法

### 自动运行（构建时）

每次在 Xcode 中构建项目时，SwiftLint 和 SwiftFormat 会自动运行：

1. **SwiftLint** 会检查代码质量问题并在 Xcode 中显示警告和错误
2. **SwiftFormat** 会自动格式化代码（仅在有变更时）

### 手动运行

#### SwiftLint

```bash
# 进入项目目录
cd /Users/apple/Developments/MorningGoalPython/MorningGoal

# 检查所有文件
swiftlint

# 自动修复可修复的问题
swiftlint --fix

# 检查特定文件
swiftlint lint --path MorningGoal/ContentView.swift
```

#### SwiftFormat

```bash
# 进入项目目录
cd /Users/apple/Developments/MorningGoalPython/MorningGoal

# 格式化所有文件
swiftformat MorningGoal

# 仅检查不修改（dry run）
swiftformat MorningGoal --lint

# 格式化特定文件
swiftformat MorningGoal/ContentView.swift
```

## 配置说明

### SwiftLint 配置 (.swiftlint.yml)

主要规则：
- **排除目录**：Pods, Carthage, DerivedData, CoreML 模型等
- **行长度**：警告 150 字符，错误 200 字符
- **文件长度**：警告 500 行，错误 1000 行
- **函数体长度**：警告 60 行，错误 100 行
- **类型长度**：警告 300 行，错误 500 行
- **圈复杂度**：警告 15，错误 25

启用的可选规则：
- `empty_count` - 使用 `isEmpty` 而不是 `count == 0`
- `force_unwrapping` - 避免强制解包
- `implicitly_unwrapped_optional` - 避免隐式解包
- 等等...

### SwiftFormat 配置 (.swiftformat)

主要设置：
- **缩进**：4 个空格
- **最大行宽**：150 字符
- **分号**：不使用分号
- **导入排序**：testable 导入在顶部
- **self 使用**：仅在 init 中使用
- **括号位置**：else 在同一行

## 集成到 CI/CD

### GitHub Actions 示例

```yaml
name: Code Quality Check

on: [push, pull_request]

jobs:
  swiftlint:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: SwiftLint
        run: |
          brew install swiftlint
          cd MorningGoal
          swiftlint lint --strict

  swiftformat:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: SwiftFormat
        run: |
          brew install swiftformat
          cd MorningGoal
          swiftformat --lint MorningGoal
```

## 自定义规则

### 修改 SwiftLint 规则

编辑 `.swiftlint.yml` 文件：

```yaml
# 禁用某个规则
disabled_rules:
  - line_length

# 调整规则参数
line_length:
  warning: 200
  error: 300
```

### 修改 SwiftFormat 规则

编辑 `.swiftformat` 文件：

```
# 改为 2 个空格缩进
--indent 2

# 改为使用 Allman 风格的括号
--allman true
```

## 常见问题

### Q: 构建时看不到 SwiftLint/SwiftFormat 的输出

**A:** 检查以下几点：
1. 确保已安装命令行工具
2. 确认 `ENABLE_USER_SCRIPT_SANDBOXING` 设置为 `NO`
3. 清理构建文件夹（Command + Shift + K）

### Q: SwiftFormat 改变了我不想改变的代码

**A:** 在 `.swiftformat` 文件中禁用特定规则，或者使用注释忽略：

```swift
// swiftformat:disable all
// 你的代码
// swiftformat:enable all
```

### Q: SwiftLint 报告太多警告

**A:** 在 `.swiftlint.yml` 中调整规则严格程度，或者使用注释忽略：

```swift
// swiftlint:disable line_length
let veryLongLine = "..."
// swiftlint:enable line_length
```

### Q: 如何在特定文件中禁用规则

**A:** 在文件顶部添加注释：

```swift
// swiftlint:disable file_length
```

## Xcode 设置建议

### 启用格式化建议

1. 打开 Xcode Preferences
2. Text Editing → Editing
3. 启用 "Including whitespace-only lines"
4. 设置 Tab Width 为 4

### 设置保存时自动格式化

虽然 Xcode 没有内置的保存时自动格式化，但可以：
1. 使用 Xcode 快捷键 `Control + I` 重新缩进
2. 使用第三方工具如 XCFormat

## 验证配置

运行以下命令验证配置是否正确：

```bash
cd /Users/apple/Developments/MorningGoalPython/MorningGoal

# 验证 SwiftLint 配置
swiftlint --config .swiftlint.yml

# 验证 SwiftFormat 配置
swiftformat --config .swiftformat --inferoptions MorningGoal
```

## 更多资源

- [SwiftLint 官方文档](https://github.com/realm/SwiftLint)
- [SwiftFormat 官方文档](https://github.com/nicklockwood/SwiftFormat)
- [SwiftLint 规则列表](https://realm.github.io/SwiftLint/rule-directory.html)
- [SwiftFormat 规则列表](https://github.com/nicklockwood/SwiftFormat/blob/main/Rules.md)

## 维护

- 定期更新工具版本：`brew upgrade swiftlint swiftformat`
- 根据团队反馈调整规则配置
- 保持配置文件与团队编码规范同步

