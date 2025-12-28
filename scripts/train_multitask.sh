#!/bin/bash
# 多任务模型训练脚本
# Multitask Model Training Script

echo "=========================================="
echo "多任务模型训练 (Multitask Model Training)"
echo "=========================================="
echo ""

# 激活虚拟环境
source activate_venv.sh

# 训练多任务模型
echo "开始训练多任务模型..."
echo "Training multitask model (Topic Classification + Sentiment Analysis)..."
echo ""

python src/training/train_multitask.py \
    --data_dir data/processed \
    --base_model models/trained/distill_student \
    --output_dir models/trained/multitask_model \
    --batch_size 24 \
    --epochs 3 \
    --lr 2e-5 \
    --max_length 128 \
    --num_topic_labels 16 \
    --num_sentiment_labels 3 \
    --limit_train 20000 \
    --limit_eval 4000 \
    --seed 42

echo ""
echo "=========================================="
echo "训练完成！"
echo "Training completed!"
echo "=========================================="
echo ""
echo "模型保存位置: models/trained/multitask_model"
echo "Model saved to: models/trained/multitask_model"
echo ""
echo "下一步: 导出CoreML模型"
echo "Next step: Export to CoreML"
echo "  python src/export/export_multitask_coreml.py --model_dir models/trained/multitask_model"
