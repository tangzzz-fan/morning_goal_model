# 🚀 快速开始指南

欢迎！你的项目已经成功配置了 SPM 依赖管理和代码质量工具。

## ⚡️ 30 秒快速设置

```bash
cd /Users/apple/Developments/MorningGoalPython/MorningGoal

# 1. 安装工具（首次）
make install

# 2. 验证配置
./verify_setup.sh

# 3. 格式化代码
make format

# 4. 完成！在 Xcode 中打开项目
open MorningGoal.xcodeproj
```

## 📋 已完成的配置

### ✅ SPM 依赖
- **SwiftLint** (v0.55.0+) - 代码质量检查
- **SwiftFormat** (v0.54.0+) - 自动代码格式化

### ✅ Build Phases
- **SwiftLint 脚本** - 构建前检查代码质量
- **SwiftFormat 脚本** - 构建前自动格式化

### ✅ 配置文件
- `.swiftlint.yml` - 代码质量规则
- `.swiftformat` - 格式化规则
- `Makefile` - 便捷命令

### ✅ 文档
- `CODE_QUALITY_SETUP.md` - 详细设置指南
- `QUICK_REFERENCE.md` - 快速参考
- `SPM_SETUP_SUMMARY.md` - 技术总结
- `GETTING_STARTED.md` - 本文档

## 🎯 日常使用

### 在 Xcode 中
1. 打开项目：`open MorningGoal.xcodeproj`
2. 编写代码
3. 按 `⌘B` 构建
4. SwiftLint 和 SwiftFormat 自动运行
5. 查看 Xcode 中的警告和错误

### 在命令行中
```bash
# 查看所有命令
make help

# 格式化代码
make format

# 检查代码质量
make lint

# 自动修复问题
make lint-fix

# 运行所有检查
make check

# 生成 HTML 报告
make report
```

## 📚 常用场景

### 场景 1: 提交代码前
```bash
make check          # 运行所有检查
git add .
git commit -m "..."
```

### 场景 2: 修复代码质量问题
```bash
make lint-fix       # 自动修复
make format         # 格式化代码
make lint           # 再次检查
```

### 场景 3: 查看详细报告
```bash
make report         # 生成 HTML 报告并打开
```

### 场景 4: 团队成员加入
```bash
make install        # 安装工具
./verify_setup.sh   # 验证配置
make check          # 测试
```

## 🔧 可选设置

### 安装 Pre-commit Hook（推荐）
自动在提交前运行检查：

```bash
cd /Users/apple/Developments/MorningGoalPython
cp MorningGoal/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

好处：
- ✅ 自动检查，不会忘记
- ✅ 提交前发现问题
- ✅ 保持代码质量

跳过检查（偶尔需要）：
```bash
git commit --no-verify
```

## 📖 文档导航

- **新手？** 👉 从这里开始 (本文档)
- **日常使用？** 👉 `QUICK_REFERENCE.md`
- **遇到问题？** 👉 `CODE_QUALITY_SETUP.md`
- **技术细节？** 👉 `SPM_SETUP_SUMMARY.md`

## 🎓 学习更多

### SwiftLint 基础
```bash
# 检查单个文件
swiftlint lint --path MorningGoal/ContentView.swift

# 查看所有规则
swiftlint rules

# 生成默认配置
swiftlint generate-docs
```

### SwiftFormat 基础
```bash
# 格式化单个文件
swiftformat MorningGoal/ContentView.swift

# 查看会改变什么（不实际修改）
swiftformat MorningGoal --dryrun

# 查看详细输出
swiftformat MorningGoal --verbose
```

### 代码中控制规则
```swift
// SwiftLint: 禁用特定规则
// swiftlint:disable force_cast
let result = something as! String
// swiftlint:enable force_cast

// SwiftFormat: 禁用格式化
// swiftformat:disable all
// 保持原有格式的代码
// swiftformat:enable all
```

## ⚙️ 自定义配置

### 调整规则严格度
编辑 `.swiftlint.yml`:
```yaml
line_length:
  warning: 200  # 更宽松
  error: 300
```

### 调整格式化风格
编辑 `.swiftformat`:
```
--indent 2      # 改为 2 空格缩进
--maxwidth 200  # 更长的行
```

## 🤝 团队协作

### 团队规范
1. ✅ 所有成员安装工具：`make install`
2. ✅ 提交前检查：`make check`
3. ✅ 不要修改配置文件（除非团队讨论）
4. ✅ 定期更新工具：`brew upgrade swiftlint swiftformat`

### 代码审查
- 工具已经检查了格式和基本质量
- 审查重点：逻辑、架构、性能、安全
- 减少格式相关的讨论时间

## 🐛 故障排查

### 问题：构建时没有运行 SwiftLint/SwiftFormat
**解决方案**：
1. 检查 Build Settings → `ENABLE_USER_SCRIPT_SANDBOXING` = `NO`
2. 清理构建：`⌘⇧K`
3. 重新构建：`⌘B`

### 问题：脚本找不到命令
**解决方案**：
```bash
# 安装工具
make install

# 验证
which swiftlint
which swiftformat
```

### 问题：警告太多，不知道从哪里开始
**解决方案**：
```bash
# 先自动修复
make lint-fix

# 再格式化
make format

# 最后手动修复剩余问题
make lint
```

### 问题：某个文件总是报错
**解决方案**：
在文件顶部添加：
```swift
// swiftlint:disable file_length
// swiftformat:disable all
```

## 📊 项目状态

运行验证脚本查看当前状态：
```bash
./verify_setup.sh
```

查看工具版本：
```bash
make version
```

## 🎉 下一步

1. ✅ 工具已安装？运行 `make install`
2. ✅ 配置已验证？运行 `./verify_setup.sh`
3. ✅ 代码已格式化？运行 `make format`
4. ✅ 质量已检查？运行 `make lint`
5. ✅ Hook 已安装？复制 `pre-commit.sample`
6. 🚀 开始编码！

## 💡 提示

- **定期运行** `make check` 保持代码质量
- **提交前运行** `make format` 保持格式一致
- **遇到问题**查看 `CODE_QUALITY_SETUP.md`
- **团队协作**保持工具和配置版本一致

---

**需要帮助？** 查看其他文档或运行 `make help`

**准备好了？** 在 Xcode 中打开项目开始编码！🚀

