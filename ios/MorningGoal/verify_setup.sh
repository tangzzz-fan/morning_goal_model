#!/bin/bash

# 验证 SPM 和代码质量工具配置脚本
# 用于检查所有配置是否正确

echo "🔍 验证 MorningGoal SPM 和代码质量工具配置"
echo "=============================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计
PASSED=0
FAILED=0
WARNINGS=0

# 检查函数
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} 文件存在: $1"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} 文件缺失: $1"
        ((FAILED++))
        return 1
    fi
}

check_command() {
    if command -v "$1" >/dev/null 2>&1; then
        VERSION=$($1 --version 2>/dev/null || $1 version 2>/dev/null || echo "未知")
        echo -e "${GREEN}✓${NC} 命令可用: $1 ($VERSION)"
        ((PASSED++))
        return 0
    else
        echo -e "${YELLOW}⚠${NC} 命令未安装: $1"
        ((WARNINGS++))
        return 1
    fi
}

check_xcode_project() {
    if grep -q "SwiftLint" "$1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Xcode 项目包含 SwiftLint 配置"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} Xcode 项目缺少 SwiftLint 配置"
        ((FAILED++))
        return 1
    fi
}

# 进入项目目录
cd "$(dirname "$0")" || exit 1
PROJECT_DIR=$(pwd)

echo "📁 项目目录: $PROJECT_DIR"
echo ""

# 1. 检查配置文件
echo "1️⃣ 检查配置文件"
echo "-------------------"
check_file ".swiftlint.yml"
check_file ".swiftformat"
check_file "Makefile"
check_file "pre-commit.sample"
check_file "CODE_QUALITY_SETUP.md"
check_file "QUICK_REFERENCE.md"
check_file "SPM_SETUP_SUMMARY.md"
echo ""

# 2. 检查命令行工具
echo "2️⃣ 检查命令行工具"
echo "-------------------"
check_command "swiftlint"
check_command "swiftformat"
check_command "make"
check_command "git"
echo ""

# 3. 检查 Xcode 项目配置
echo "3️⃣ 检查 Xcode 项目配置"
echo "-------------------"
PBXPROJ="MorningGoal.xcodeproj/project.pbxproj"
if [ -f "$PBXPROJ" ]; then
    check_file "$PBXPROJ"
    
    # 检查 SPM 包引用
    if grep -q "XCRemoteSwiftPackageReference \"SwiftLint\"" "$PBXPROJ"; then
        echo -e "${GREEN}✓${NC} SPM 包引用: SwiftLint"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} SPM 包引用缺失: SwiftLint"
        ((FAILED++))
    fi
    
    if grep -q "XCRemoteSwiftPackageReference \"SwiftFormat\"" "$PBXPROJ"; then
        echo -e "${GREEN}✓${NC} SPM 包引用: SwiftFormat"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} SPM 包引用缺失: SwiftFormat"
        ((FAILED++))
    fi
    
    # 检查 Build Phase
    if grep -q "PBXShellScriptBuildPhase" "$PBXPROJ"; then
        echo -e "${GREEN}✓${NC} Build Phase 脚本已配置"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} Build Phase 脚本缺失"
        ((FAILED++))
    fi
    
    # 检查沙盒设置
    if grep -q "ENABLE_USER_SCRIPT_SANDBOXING = NO" "$PBXPROJ"; then
        echo -e "${GREEN}✓${NC} 用户脚本沙盒已禁用"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} 用户脚本沙盒未禁用（脚本可能无法运行）"
        ((FAILED++))
    fi
else
    echo -e "${RED}✗${NC} 找不到 Xcode 项目文件"
    ((FAILED++))
fi
echo ""

# 4. 验证配置文件语法
echo "4️⃣ 验证配置文件语法"
echo "-------------------"

# 验证 YAML 语法（简单检查）
if [ -f ".swiftlint.yml" ]; then
    if grep -q "^excluded:" ".swiftlint.yml" && grep -q "^included:" ".swiftlint.yml"; then
        echo -e "${GREEN}✓${NC} .swiftlint.yml 语法检查通过"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} .swiftlint.yml 可能存在语法问题"
        ((WARNINGS++))
    fi
fi

# 验证 SwiftFormat 配置
if [ -f ".swiftformat" ]; then
    if grep -q "^--indent" ".swiftformat" && grep -q "^--maxwidth" ".swiftformat"; then
        echo -e "${GREEN}✓${NC} .swiftformat 语法检查通过"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} .swiftformat 可能存在语法问题"
        ((WARNINGS++))
    fi
fi

# 验证 Makefile
if [ -f "Makefile" ]; then
    if grep -q "^help:" "Makefile" && grep -q "^install:" "Makefile"; then
        echo -e "${GREEN}✓${NC} Makefile 语法检查通过"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} Makefile 可能存在语法问题"
        ((WARNINGS++))
    fi
fi
echo ""

# 5. 测试 Makefile 命令
echo "5️⃣ 测试 Makefile 命令"
echo "-------------------"
if make help >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} make help 可以运行"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} make help 运行失败"
    ((FAILED++))
fi

if make version >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} make version 可以运行"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠${NC} make version 运行失败（可能工具未安装）"
    ((WARNINGS++))
fi
echo ""

# 6. 检查 Git 状态
echo "6️⃣ 检查 Git 配置"
echo "-------------------"
if [ -d "../.git" ]; then
    echo -e "${GREEN}✓${NC} Git 仓库已初始化"
    ((PASSED++))
    
    if [ -f "../.git/hooks/pre-commit" ]; then
        echo -e "${GREEN}✓${NC} Pre-commit hook 已安装"
        ((PASSED++))
    else
        echo -e "${YELLOW}⚠${NC} Pre-commit hook 未安装"
        echo "   提示: 运行 'cp pre-commit.sample ../.git/hooks/pre-commit && chmod +x ../.git/hooks/pre-commit'"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} Git 仓库未找到"
    ((WARNINGS++))
fi
echo ""

# 总结
echo "=============================================="
echo "📊 验证总结"
echo "=============================================="
echo -e "通过: ${GREEN}$PASSED${NC}"
echo -e "失败: ${RED}$FAILED${NC}"
echo -e "警告: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ 所有关键配置检查通过！${NC}"
    echo ""
    echo "📚 下一步操作:"
    echo "1. 安装命令行工具 (如果未安装): make install"
    echo "2. 格式化现有代码: make format"
    echo "3. 检查代码质量: make lint"
    echo "4. 安装 pre-commit hook: cp pre-commit.sample ../.git/hooks/pre-commit && chmod +x ../.git/hooks/pre-commit"
    echo "5. 在 Xcode 中构建项目测试"
    echo ""
    exit 0
else
    echo -e "${RED}❌ 发现 $FAILED 个问题，请检查上述错误${NC}"
    echo ""
    echo "📚 修复建议:"
    echo "1. 确保在正确的目录中运行脚本"
    echo "2. 检查是否所有文件都已正确创建"
    echo "3. 验证 Xcode 项目文件是否正确修改"
    echo "4. 查看 CODE_QUALITY_SETUP.md 获取详细说明"
    echo ""
    exit 1
fi

