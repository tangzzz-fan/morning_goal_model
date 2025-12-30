#!/bin/bash
# 运行模型质量测试的便捷脚本

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   模型质量测试工具${NC}"
echo -e "${GREEN}========================================${NC}"

# 默认值
TEST_LEVEL="all"
MODEL_PATH="models/trained/distill_student"
MAX_SAMPLES=500
VERBOSE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-level)
            TEST_LEVEL="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -s|--samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-vv"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -t, --test-level LEVEL    测试级别: unit, integration, quality, regression, all (默认: all)"
            echo "  -m, --model-path PATH     模型路径 (默认: models/trained/distill_student)"
            echo "  -s, --samples NUM         测试样本数量 (默认: 500)"
            echo "  -v, --verbose             详细输出"
            echo "  -h, --help                显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 运行所有测试"
            echo "  $0 -t unit                           # 只运行单元测试"
            echo "  $0 -t quality -s 1000                # 运行质量测试，使用1000个样本"
            echo "  $0 -m models/trained/new_model       # 测试指定模型"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python${NC}"
    exit 1
fi

# 检查pytest
if ! python -c "import pytest" &> /dev/null; then
    echo -e "${YELLOW}警告: 未安装pytest，正在安装...${NC}"
    pip install pytest pytest-html pytest-cov
fi

# 创建报告目录
mkdir -p test_reports
mkdir -p tests/reports

# 设置环境变量
export TEST_MAX_SAMPLES=$MAX_SAMPLES
export REGRESSION_TEST_SAMPLES=$((MAX_SAMPLES / 2))

echo ""
echo -e "${GREEN}测试配置:${NC}"
echo "  测试级别: $TEST_LEVEL"
echo "  模型路径: $MODEL_PATH"
echo "  样本数量: $MAX_SAMPLES"
echo ""

# 运行测试
case $TEST_LEVEL in
    unit)
        echo -e "${GREEN}运行单元测试...${NC}"
        pytest tests/test_model_unit.py $VERBOSE \
            --html=test_reports/unit_test_report.html --self-contained-html \
            --cov=src --cov-report=html --cov-report=term
        ;;
    
    integration)
        echo -e "${GREEN}运行集成测试...${NC}"
        pytest tests/test_model_integration.py $VERBOSE \
            --html=test_reports/integration_test_report.html --self-contained-html
        ;;
    
    quality)
        echo -e "${GREEN}运行质量测试...${NC}"
        pytest tests/test_model_quality.py $VERBOSE \
            --html=test_reports/quality_test_report.html --self-contained-html
        ;;
    
    regression)
        echo -e "${GREEN}运行回归测试...${NC}"
        pytest tests/test_model_regression.py $VERBOSE \
            --html=test_reports/regression_test_report.html --self-contained-html
        ;;
    
    all)
        echo -e "${GREEN}运行所有测试...${NC}"
        pytest tests/ $VERBOSE \
            --html=test_reports/all_tests_report.html --self-contained-html \
            --cov=src --cov-report=html --cov-report=term
        ;;
    
    *)
        echo -e "${RED}未知测试级别: $TEST_LEVEL${NC}"
        echo "可用的测试级别: unit, integration, quality, regression, all"
        exit 1
        ;;
esac

TEST_EXIT_CODE=$?

echo ""
echo -e "${GREEN}========================================${NC}"
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 测试通过！${NC}"
else
    echo -e "${RED}❌ 测试失败！${NC}"
fi
echo -e "${GREEN}========================================${NC}"

echo ""
echo -e "${YELLOW}测试报告已保存到:${NC}"
echo "  - HTML报告: test_reports/"
echo "  - JSON报告: tests/reports/"
if [ -d "htmlcov" ]; then
    echo "  - 覆盖率报告: htmlcov/index.html"
fi

# 如果有质量测试报告，显示摘要
if [ -f "tests/reports/current_metrics.json" ]; then
    echo ""
    echo -e "${GREEN}性能指标摘要:${NC}"
    python -c "
import json
try:
    with open('tests/reports/current_metrics.json') as f:
        metrics = json.load(f)
    for key, value in metrics.items():
        print(f'  - {key}: {value:.4f}')
except:
    pass
    "
fi

exit $TEST_EXIT_CODE

