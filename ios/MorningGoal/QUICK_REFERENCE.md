# 代码质量工具快速参考

## 快速命令

```bash
# 进入项目目录
cd /Users/apple/Developments/MorningGoalPython/MorningGoal

# 安装工具
make install

# 格式化代码
make format

# 检查代码质量
make lint

# 自动修复问题
make lint-fix

# 运行所有检查
make check

# 显示工具版本
make version

# 生成 HTML 报告
make report
```

## SwiftLint 常用命令

```bash
# 基本检查
swiftlint

# 自动修复
swiftlint --fix

# 检查特定文件
swiftlint lint --path MorningGoal/ContentView.swift

# 严格模式（警告也算错误）
swiftlint --strict

# 生成配置文件示例
swiftlint rules > rules.txt
```

## SwiftFormat 常用命令

```bash
# 格式化所有文件
swiftformat MorningGoal

# 仅检查不修改
swiftformat MorningGoal --lint

# 格式化特定文件
swiftformat MorningGoal/ContentView.swift

# 显示差异但不修改
swiftformat MorningGoal --dryrun

# 显示详细输出
swiftformat MorningGoal --verbose
```

## 代码中的注释控制

### SwiftLint

```swift
// 禁用整个文件的规则
// swiftlint:disable file_length

// 禁用特定规则
// swiftlint:disable line_length
let longLine = "very long string..."
// swiftlint:enable line_length

// 禁用下一行
// swiftlint:disable:next force_cast
let result = something as! String

// 禁用这一行
let result = something as! String // swiftlint:disable:this force_cast

// 禁用多个规则
// swiftlint:disable force_cast line_length
```

### SwiftFormat

```swift
// 禁用所有规则
// swiftformat:disable all
// 你的代码
// swiftformat:enable all

// 禁用特定规则
// swiftformat:disable indent
func example() {
  // 保持原有缩进
  }
// swiftformat:enable indent
```

## 配置文件位置

- **SwiftLint**: `.swiftlint.yml`
- **SwiftFormat**: `.swiftformat`
- **Makefile**: `Makefile`
- **Pre-commit hook**: `pre-commit.sample`

## 安装 Pre-commit Hook

```bash
cd /Users/apple/Developments/MorningGoalPython

# 复制并设置权限
cp MorningGoal/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# 测试
.git/hooks/pre-commit
```

## Xcode 集成

### 查看 Build Phase

1. 打开 Xcode
2. 选择 MorningGoal Target
3. Build Phases 标签
4. 应该看到：
   - ✅ SwiftLint
   - ✅ SwiftFormat
   - Sources
   - Frameworks
   - Resources

### 查看 SPM 依赖

1. 打开 Xcode
2. 左侧导航栏 → Package Dependencies
3. 应该看到：
   - ✅ SwiftLint (0.55.0+)
   - ✅ SwiftFormat (0.54.0+)

## 常见问题速查

### 构建时不运行脚本？

```bash
# 检查设置
# Build Settings → ENABLE_USER_SCRIPT_SANDBOXING = NO ✅
```

### 脚本找不到命令？

```bash
# 安装工具
brew install swiftlint swiftformat

# 验证安装
which swiftlint
which swiftformat
```

### 警告太多？

编辑 `.swiftlint.yml`，调整规则或禁用某些规则

### 格式化后代码变丑？

编辑 `.swiftformat`，调整格式化选项

### 想跳过某次检查？

```bash
# 跳过 pre-commit hook
git commit --no-verify
```

## 规则调整示例

### SwiftLint - 调整行长度

编辑 `.swiftlint.yml`:

```yaml
line_length:
  warning: 200  # 从 150 改为 200
  error: 300    # 从 200 改为 300
```

### SwiftFormat - 改为 2 空格缩进

编辑 `.swiftformat`:

```
--indent 2  # 从 4 改为 2
```

## 团队协作建议

1. **所有成员安装工具**
   ```bash
   make install
   ```

2. **提交前检查**
   ```bash
   make check
   ```

3. **安装 pre-commit hook**
   ```bash
   cp MorningGoal/pre-commit.sample .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

4. **定期更新工具**
   ```bash
   brew upgrade swiftlint swiftformat
   ```

5. **统一配置**
   - 不要修改 `.swiftlint.yml` 和 `.swiftformat` 
   - 如需修改，先与团队讨论

## 性能提示

- **只格式化修改的文件**：SwiftFormat 会自动检测
- **使用 Xcode 缓存**：不要频繁清理 DerivedData
- **并行构建**：已在项目中启用
- **选择性检查**：使用 `--path` 参数检查特定文件

## 更多帮助

```bash
# 查看所有 make 命令
make help

# SwiftLint 帮助
swiftlint help

# SwiftFormat 帮助
swiftformat --help

# 查看详细文档
cat CODE_QUALITY_SETUP.md
```

