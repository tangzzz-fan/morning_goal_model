#!/bin/bash
# CoreML导出脚本 - 多任务模型
# CoreML Export Script - Multitask Model

echo "=========================================="
echo "导出多任务模型到CoreML"
echo "Export Multitask Model to CoreML"
echo "=========================================="
echo ""

# 激活虚拟环境
source activate_venv.sh

# 导出CoreML模型
echo "开始导出..."
echo "Exporting..."
echo ""

python src/export/export_multitask_coreml.py \
    --model_dir models/trained/multitask_model \
    --output_dir models/coreml \
    --seq_len 128 \
    --num_topic_labels 16 \
    --num_sentiment_labels 3

echo ""
echo "=========================================="
echo "导出完成！"
echo "Export completed!"
echo "=========================================="
echo ""
echo "CoreML模型位置: models/coreml/multitask_model.mlpackage"
echo "CoreML model saved to: models/coreml/multitask_model.mlpackage"
echo ""
echo "标签映射: models/coreml/label_mapping.json"
echo "Label mapping: models/coreml/label_mapping.json"
