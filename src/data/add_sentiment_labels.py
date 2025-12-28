"""
为现有数据集添加情感标签
使用规则 + 关键词的方法自动标注情感
"""
import argparse
from pathlib import Path
import pandas as pd
import re


# 情感关键词词典
POSITIVE_KEYWORDS = [
    # 积极动词
    '完成', '成功', '达成', '实现', '突破', '进步', '提升', '改善', '优化',
    # 积极形容词
    '开心', '快乐', '满意', '棒', '好', '优秀', '出色',
    # 积极emoji
    '✅', '😊', '💪', '👍', '🎉', '❤️', '😄', '😃', '🌟', '⭐', '📈',
    # 其他积极表达
    '加油', '努力', '坚持', '继续',
]

NEGATIVE_KEYWORDS = [
    # 消极动词
    '失败', '放弃', '拖延', '错过', '忘记',
    # 消极形容词/名词
    '焦虑', '压力', '困难', '问题', '挑战', '累', '疲惫', '难',
    # 消极emoji
    '😢', '😭', '😞', '😔', '❌', '😰', '😓',
    # 其他消极表达
    '没', '不', '未',
]

# 中性关键词（计划类、记录类）
NEUTRAL_KEYWORDS = [
    '打卡', '记录', '计划', '安排', '准备', '整理', '复盘',
    '今天', '明天', '本周', '当日', '优先', '事项',
]


def auto_label_sentiment(text: str) -> int:
    """
    自动标注情感
    
    Args:
        text: 输入文本
    
    Returns:
        int: 0=消极, 1=中性, 2=积极
    """
    text = str(text).lower()
    
    # 计算各类关键词出现次数
    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    neutral_count = sum(1 for kw in NEUTRAL_KEYWORDS if kw in text)
    
    # 决策逻辑
    # 1. 如果有明显的积极信号
    if positive_count > negative_count and positive_count > 0:
        return 2  # 积极
    
    # 2. 如果有明显的消极信号
    if negative_count > positive_count and negative_count > 0:
        return 0  # 消极
    
    # 3. 如果有中性关键词或无明显倾向
    if neutral_count > 0 or (positive_count == 0 and negative_count == 0):
        return 1  # 中性
    
    # 4. 默认中性
    return 1


def add_sentiment_labels(input_csv: str, output_csv: str):
    """
    为CSV文件添加情感标签列
    
    Args:
        input_csv: 输入CSV路径 (格式: text,label)
        output_csv: 输出CSV路径 (格式: text,topic_label,sentiment_label)
    """
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 检查列名
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Input CSV must have 'text' and 'label' columns")
    
    # 重命名 label -> topic_label
    df = df.rename(columns={'label': 'topic_label'})
    
    # 添加情感标签
    print("Auto-labeling sentiment...")
    df['sentiment_label'] = df['text'].apply(auto_label_sentiment)
    
    # 统计情感分布
    sentiment_dist = df['sentiment_label'].value_counts().sort_index()
    print("\nSentiment distribution:")
    print(f"  0 (消极): {sentiment_dist.get(0, 0)} ({sentiment_dist.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  1 (中性): {sentiment_dist.get(1, 0)} ({sentiment_dist.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  2 (积极): {sentiment_dist.get(2, 0)} ({sentiment_dist.get(2, 0)/len(df)*100:.1f}%)")
    
    # 保存
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    
    # 显示示例
    print("\nSample rows:")
    print(df.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Add sentiment labels to existing dataset"
    )
    parser.add_argument(
        "--input_csv",
        default="data/processed/train.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output_csv",
        default="data/processed/train_multitask.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()
    
    add_sentiment_labels(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
