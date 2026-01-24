# Task: Multitask Model Distillation

**Objective**: Distill the knowledge from a large, fine-tuned Multitask BERT (Teacher) into a smaller, faster model (Student) suitable for iOS deployment, maintaining high accuracy on both Topic Classification and Sentiment Analysis.

## 1. Prerequisites
*   [ ] **Teacher Model**: A fully trained `bert-base-chinese` multitask model with high accuracy.
    *   Location: `models/trained/bert_base_chinese_multitask/` (Need to verify or train this first).
*   [ ] **Student Config**: A lightweight configuration (e.g., 4 layers, 512 hidden size).

## 2. Implementation Steps

### 1. Verify/Train Teacher Model [Completed]
- **Goal**: Ensure a high-performance teacher model exists.
- **Action**: Run `src/training/train_multitask.py` with `bert-base-chinese`.
- **Status**: Done. Teacher trained at `models/trained/bert_base_chinese_multitask`.

### 2. Create Distillation Script [Completed]
- **Goal**: Implement `distill_multitask.py` with KL Divergence loss.
- **Key Components**:
  - `MultitaskDistillTrainer`: Custom trainer overriding `compute_loss`.
  - Loss Function: `Alpha * (KL_topic + KL_sentiment) + (1 - Alpha) * (CE_topic + CE_sentiment)`
- **Status**: Done. File created at `src/training/distill_multitask.py`.

### 3. Run Distillation [Completed]
- **Goal**: Transfer knowledge to student model.
- **Command**:
  ```bash
  python src/training/distill_multitask.py \
    --teacher_path models/trained/bert_base_chinese_multitask \
    --student_model_name uer/chinese_roberta_L-4_H-512 \
    --alpha 0.5 \
    --temperature 4.0
  ```
- **Status**: Done. Student model saved at `models/trained/distill_multitask_student`.

### 4. Evaluate & Convert [Pending]
- **Goal**: Verify metrics and convert to CoreML.
- **Next Step**: Check `metrics_test.json` and proceed to CoreML conversion (next task).

## 3. Success Criteria
*   Student Size: < 50MB (Int8 quantized).
*   Accuracy Drop: < 3% compared to Teacher.
*   Inference Speed: < 50ms on iPhone (Neural Engine).
