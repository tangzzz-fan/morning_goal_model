.PHONY: help install lint lint-fix format format-check check clean

# 默认目标
help:
	@echo "MorningGoal 代码质量工具"
	@echo ""
	@echo "可用命令:"
	@echo "  make install        - 安装 SwiftLint 和 SwiftFormat"
	@echo "  make lint           - 运行 SwiftLint 检查"
	@echo "  make lint-fix       - 运行 SwiftLint 并自动修复"
	@echo "  make format         - 运行 SwiftFormat 格式化代码"
	@echo "  make format-check   - 检查代码格式但不修改"
	@echo "  make check          - 运行所有检查（lint + format-check）"
	@echo "  make clean          - 清理构建缓存"
	@echo ""

# 安装工具
install:
	@echo "🔧 安装代码质量工具..."
	@if command -v brew >/dev/null 2>&1; then \
		echo "📦 通过 Homebrew 安装 SwiftLint..."; \
		brew install swiftlint || brew upgrade swiftlint; \
		echo "📦 通过 Homebrew 安装 SwiftFormat..."; \
		brew install swiftformat || brew upgrade swiftformat; \
		echo "✅ 安装完成！"; \
	else \
		echo "❌ 未找到 Homebrew，请先安装 Homebrew: https://brew.sh"; \
		exit 1; \
	fi

# 运行 SwiftLint
lint:
	@echo "🔍 运行 SwiftLint 检查..."
	@if command -v swiftlint >/dev/null 2>&1; then \
		swiftlint --config .swiftlint.yml; \
	else \
		echo "❌ SwiftLint 未安装，请运行: make install"; \
		exit 1; \
	fi

# 运行 SwiftLint 并自动修复
lint-fix:
	@echo "🔧 运行 SwiftLint 自动修复..."
	@if command -v swiftlint >/dev/null 2>&1; then \
		swiftlint --fix --config .swiftlint.yml; \
		echo "✅ 自动修复完成！"; \
	else \
		echo "❌ SwiftLint 未安装，请运行: make install"; \
		exit 1; \
	fi

# 运行 SwiftFormat
format:
	@echo "🎨 运行 SwiftFormat 格式化..."
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat MorningGoal --config .swiftformat; \
		echo "✅ 格式化完成！"; \
	else \
		echo "❌ SwiftFormat 未安装，请运行: make install"; \
		exit 1; \
	fi

# 检查格式但不修改
format-check:
	@echo "🔍 检查代码格式..."
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat MorningGoal --config .swiftformat --lint; \
	else \
		echo "❌ SwiftFormat 未安装，请运行: make install"; \
		exit 1; \
	fi

# 运行所有检查
check: lint format-check
	@echo "✅ 所有检查完成！"

# 清理构建缓存
clean:
	@echo "🧹 清理构建缓存..."
	@rm -rf ~/Library/Developer/Xcode/DerivedData/MorningGoal-*
	@echo "✅ 清理完成！"

# 显示工具版本
version:
	@echo "📋 已安装工具版本:"
	@if command -v swiftlint >/dev/null 2>&1; then \
		echo "SwiftLint: $$(swiftlint version)"; \
	else \
		echo "SwiftLint: 未安装"; \
	fi
	@if command -v swiftformat >/dev/null 2>&1; then \
		echo "SwiftFormat: $$(swiftformat --version)"; \
	else \
		echo "SwiftFormat: 未安装"; \
	fi

# 运行 SwiftLint 分析并生成报告
report:
	@echo "📊 生成 SwiftLint 报告..."
	@if command -v swiftlint >/dev/null 2>&1; then \
		mkdir -p reports; \
		swiftlint --config .swiftlint.yml --reporter html > reports/swiftlint-report.html; \
		echo "✅ 报告已生成: reports/swiftlint-report.html"; \
		open reports/swiftlint-report.html; \
	else \
		echo "❌ SwiftLint 未安装，请运行: make install"; \
		exit 1; \
	fi

