import argparse
from pathlib import Path
import random
import pandas as pd
from sklearn.model_selection import train_test_split

CATS = [
    "工作",
    "健康",
    "家庭",
    "个人发展",
    "理财",
    "社交",
    "家务",
    "学习",
    "睡眠",
    "饮食",
    "心态",
    "娱乐",
    "出行",
    "职业发展",
    "沟通",
    "育儿",
]

PHRASES = {
    "工作": [
        "完成项目进度汇报",
        "整理会议纪要",
        "推进需求评审",
        "优化模块代码",
        "复盘昨天任务",
        "准备PPT演示",
        "撰写周报",
        "更新OKR进度",
        "安排一对一沟通",
        "清理邮箱",
        "完善接口文档",
        "提交代码Review",
        "搭建测试用例",
        "规划本周任务",
    ],
    "健康": [
        "晨跑三公里",
        "力量训练二十分钟",
        "早睡不熬夜",
        "饮水两升",
        "无糖饮食一天",
        "午休十五分钟",
        "拉伸肩颈",
        "瑜伽三十分钟",
        "骑行五公里",
        "步行一万步",
        "减少咖啡因",
        "多吃蔬果",
    ],
    "家庭": [
        "与父母通话",
        "陪孩子阅读",
        "整理客厅",
        "准备家庭晚餐",
        "修理书架",
        "安排周末出游",
        "清理冰箱",
        "采购日用品",
        "洗衣与收纳",
        "家庭预算记录",
    ],
    "个人发展": [
        "阅读三十页",
        "英语口语练习",
        "完成一节课程",
        "写作五百字",
        "冥想十分钟",
        "复盘年度目标",
        "练习演讲",
        "刷题四十分钟",
        "学习新技能",
        "整理学习笔记",
    ],
    "理财": [
        "记账十五分钟",
        "审查本月预算",
        "还款计划确认",
        "储蓄目标更新",
        "投资复盘",
        "整理发票",
    ],
    "社交": [
        "联系一位老朋友",
        "回复消息清零",
        "安排咖啡聊天",
        "参加社区活动",
        "社交媒体断舍离",
        "写感谢信",
    ],
    "家务": [
        "整理房间",
        "清洁厨房",
        "洗衣与收纳",
        "倒垃圾",
        "整理书桌",
        "拖地",
    ],
    "学习": [
        "复习笔记",
        "完成作业",
        "练习编程",
        "背单词",
        "阅读论文",
        "课堂总结",
    ],
    "睡眠": [
        "早睡",
        "午休十五分钟",
        "睡前不看手机",
        "固定作息",
        "睡眠追踪",
        "呼吸放松",
    ],
    "饮食": [
        "健康早餐",
        "备餐",
        "多吃蔬果",
        "减少外卖",
        "无糖一天",
        "喝水两升",
    ],
    "心态": [
        "感恩记录",
        "正念冥想",
        "情绪复盘",
        "积极肯定",
        "呼吸练习",
        "写日记",
    ],
    "娱乐": [
        "看电影",
        "弹吉他",
        "绘画练习",
        "摄影练习",
        "听音乐",
        "游戏时间控制",
    ],
    "出行": [
        "骑行五公里",
        "散步三十分钟",
        "出行规划",
        "订票",
        "行李整理",
        "通勤步行",
    ],
    "职业发展": [
        "更新简历",
        "优化LinkedIn",
        "练习面试",
        "职业规划",
        "学习行业报告",
        "拓展人脉",
    ],
    "沟通": [
        "给同事反馈",
        "一对一会谈",
        "演讲练习",
        "邮件清零",
        "写会议纪要",
        "准备发言",
    ],
    "育儿": [
        "陪孩子玩耍",
        "家庭阅读",
        "课后辅导",
        "亲子运动",
        "早睡安排",
        "练习拼图",
    ],
}

TEMPLATES = [
    "今天的目标：{p}",
    "专注完成：{p}",
    "只做一件事：{p}",
    "当日焦点：{p}",
    "优先事项：{p}",
    "Focus: {p}",
    "Just one thing: {p}",
    "计划：{p}",
    "打卡：{p}",
]

EMOJIS = [
    "💪",
    "😊",
    "✅",
    "🔥",
    "🏃",
    "📚",
    "🧘",
    "🍎",
    "☕",
    "📈",
]

EN_TOKENS = [
    "focus",
    "workout",
    "study",
    "meeting",
    "review",
    "plan",
]

HASH_TAGS = [
    "#健身",
    "#work",
    "#study",
    "#family",
    "#reading",
]


def apply_noise(text: str, noise_rate: float, emoji_rate: float) -> str:
    t = text
    if random.random() < noise_rate:
        t = t + random.choice(["！", "～", "…", "。", "??", "?!"])
    if random.random() < noise_rate:
        t = t + " " + random.choice(EN_TOKENS)
    if random.random() < noise_rate:
        t = t + " " + random.choice(HASH_TAGS)
    if random.random() < noise_rate:
        t = t.replace(" ", "  ")
    if random.random() < noise_rate:
        if len(t) > 3:
            i = random.randint(0, len(t) - 2)
            t = t[:i] + t[i] + t[i:]
    if random.random() < emoji_rate:
        t = t + random.choice(EMOJIS)
    return t


def gen_samples(
    count: int,
    seed: int,
    noise_rate: float,
    emoji_rate: float,
    freeform_rate: float,
) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for _ in range(count):
        cat = random.choice(CATS)
        phrase = random.choice(PHRASES[cat])
        use_template = random.random() >= freeform_rate
        if use_template:
            base = random.choice(TEMPLATES).format(p=phrase)
        else:
            base = phrase
        text = apply_noise(base, noise_rate, emoji_rate)
        label = CATS.index(cat)
        rows.append({"text": text, "label": label})
    return pd.DataFrame(rows)


def split_and_save(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    y = df["label"]
    train, tmp, y_train, y_tmp = train_test_split(
        df, y, test_size=0.3, stratify=y, random_state=seed
    )
    val, test, _, _ = train_test_split(
        tmp, y_tmp, test_size=1 / 3, stratify=y_tmp, random_state=seed
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--noise_rate", type=float, default=0.3)
    parser.add_argument("--emoji_rate", type=float, default=0.25)
    parser.add_argument("--freeform_rate", type=float, default=0.4)
    args = parser.parse_args()
    df = gen_samples(
        args.count,
        args.seed,
        args.noise_rate,
        args.emoji_rate,
        args.freeform_rate,
    )
    split_and_save(df, Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
