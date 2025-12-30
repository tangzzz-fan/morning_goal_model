#!/usr/bin/env python
"""
更新基准指标的脚本
在确认新模型性能良好后，使用此脚本更新baseline_metrics.json
"""
import json
import argparse
from pathlib import Path
from datetime import datetime


def update_baseline(
    current_metrics_path: str,
    baseline_path: str,
    model_version: str = None,
    force: bool = False
):
    """
    更新基准指标
    
    Args:
        current_metrics_path: 当前模型指标文件路径
        baseline_path: 基准指标文件路径
        model_version: 模型版本标识
        force: 是否强制更新（即使新模型性能更差）
    """
    current_metrics_file = Path(current_metrics_path)
    baseline_file = Path(baseline_path)
    
    # 读取当前指标
    if not current_metrics_file.exists():
        print(f"错误: 未找到当前指标文件: {current_metrics_path}")
        print("请先运行测试生成指标文件")
        return False
    
    with open(current_metrics_file) as f:
        current_metrics = json.load(f)
    
    print("当前模型指标:")
    for key, value in current_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 读取基准指标
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_metrics = json.load(f)
        
        print("\n基准模型指标:")
        for key in ['topic_accuracy', 'topic_f1', 'sentiment_accuracy', 'sentiment_f1']:
            if key in baseline_metrics:
                print(f"  {key}: {baseline_metrics[key]:.4f}")
        
        # 比较性能
        print("\n性能变化:")
        all_better_or_equal = True
        for key in ['topic_accuracy', 'topic_f1', 'sentiment_accuracy', 'sentiment_f1']:
            if key in baseline_metrics and key in current_metrics:
                diff = current_metrics[key] - baseline_metrics[key]
                symbol = "✅" if diff >= 0 else "⚠️"
                print(f"  {symbol} {key}: {diff:+.4f}")
                if diff < 0:
                    all_better_or_equal = False
        
        # 检查是否应该更新
        if not force and not all_better_or_equal:
            print("\n⚠️  警告: 新模型在某些指标上表现更差")
            response = input("是否仍要更新基准? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("已取消更新")
                return False
    else:
        print("\n未找到现有基准，将创建新基准")
    
    # 更新基准
    new_baseline = {
        "model_version": model_version or "unknown",
        "test_date": datetime.now().strftime("%Y-%m-%d"),
        **current_metrics,
        "notes": "Updated from test results"
    }
    
    # 备份旧基准
    if baseline_file.exists():
        backup_file = baseline_file.with_suffix('.json.backup')
        import shutil
        shutil.copy2(baseline_file, backup_file)
        print(f"\n旧基准已备份到: {backup_file}")
    
    # 保存新基准
    with open(baseline_file, 'w') as f:
        json.dump(new_baseline, f, indent=2)
    
    print(f"\n✅ 基准已更新: {baseline_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="更新模型基准指标")
    parser.add_argument(
        "--current-metrics",
        default="tests/reports/current_metrics.json",
        help="当前模型指标文件路径"
    )
    parser.add_argument(
        "--baseline",
        default="tests/baseline_metrics.json",
        help="基准指标文件路径"
    )
    parser.add_argument(
        "--model-version",
        help="模型版本标识"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制更新，即使新模型性能更差"
    )
    
    args = parser.parse_args()
    
    success = update_baseline(
        args.current_metrics,
        args.baseline,
        args.model_version,
        args.force
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

