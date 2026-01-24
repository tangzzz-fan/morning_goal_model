"""
Multitask Knowledge Distillation Script
Distills knowledge from a large Multitask Teacher to a smaller Student model.
Supports both Topic Classification and Sentiment Analysis tasks simultaneously.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer
)
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification
from src.training.train_multitask import MultitaskDataset, read_multitask_csvs, compute_multitask_metrics


class MultitaskDistillTrainer(Trainer):
    """
    Trainer for Multitask Knowledge Distillation.
    Computes loss based on KL Divergence for both tasks + Cross Entropy.
    """
    def __init__(self, teacher_model, temperature, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.teacher.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward pass Student
        outputs_s = model(**inputs)
        # Student returns {loss, topic_logits, sentiment_logits, ...}
        # Note: outputs_s['loss'] is already (CE_topic + CE_sentiment)
        
        # 2. Forward pass Teacher (no grad)
        with torch.no_grad():
            outputs_t = self.teacher(**inputs)
        
        # 3. Compute KL Divergence Loss
        loss_kl_topic = self.compute_kl_loss(
            outputs_s['topic_logits'], 
            outputs_t['topic_logits']
        )
        loss_kl_sentiment = self.compute_kl_loss(
            outputs_s['sentiment_logits'], 
            outputs_t['sentiment_logits']
        )
        loss_kl_total = loss_kl_topic + loss_kl_sentiment
        
        # 4. Compute Hard Label Loss (Cross Entropy)
        # The model already computes this if labels are provided
        loss_ce_total = outputs_s['loss']
        
        # 5. Combined Loss
        # Loss = alpha * KL + (1 - alpha) * CE
        # Note: Standard distillation often uses: alpha * KL * T^2 + (1 - alpha) * CE
        loss = (self.alpha * loss_kl_total) + ((1 - self.alpha) * loss_ce_total)
        
        return (loss, outputs_s) if return_outputs else loss

    def compute_kl_loss(self, logits_s, logits_t):
        """
        Compute KL Divergence: KL(softmax(t/T), softmax(s/T)) * T^2
        """
        T = self.temperature
        
        # Log Softmax Student
        log_prob_s = F.log_softmax(logits_s / T, dim=-1)
        
        # Softmax Teacher
        prob_t = F.softmax(logits_t / T, dim=-1)
        
        # KL Div
        kl = F.kl_div(log_prob_s, prob_t, reduction="batchmean") * (T ** 2)
        return kl

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Standard multitask prediction step (copied from MultitaskTrainer)"""
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs['loss'] if 'loss' in outputs else None
            
            topic_logits = outputs['topic_logits']
            sentiment_logits = outputs['sentiment_logits']
            logits = (topic_logits.cpu(), sentiment_logits.cpu())
        
        if prediction_loss_only:
            return (loss, None, None)
        
        topic_labels = inputs['topic_labels'].cpu()
        sentiment_labels = inputs['sentiment_labels'].cpu()
        labels = torch.stack([topic_labels, sentiment_labels], dim=1)
        
        return (loss, logits, labels)


def run_distillation(args):
    torch.manual_seed(args.seed)
    
    # 1. Load Data
    print("Loading datasets...")
    train_df, val_df, test_df = read_multitask_csvs(args.data_dir)
    
    if args.limit_train > 0:
        train_df = train_df.sample(n=min(args.limit_train, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(n=min(args.limit_eval, len(val_df)), random_state=args.seed)
    
    # 2. Load Teacher
    print(f"Loading Teacher Model from {args.teacher_path}...")
    teacher_config = AutoConfig.from_pretrained(args.teacher_path)
    teacher = MultitaskBertForClassification.from_pretrained(args.teacher_path, config=teacher_config)
    
    # 3. Initialize Student
    print(f"Initializing Student Model: {args.student_model_name}...")
    # Load tokenizer from teacher (or base) to ensure compatibility
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path)
    
    # Load student config (e.g. uer/chinese_roberta_L-4_H-512)
    student_config = AutoConfig.from_pretrained(args.student_model_name)
    student_config.num_labels = teacher_config.num_labels # Not strictly used but good practice
    
    # Initialize Student Weights
    student = MultitaskBertForClassification.from_pretrained(
        args.student_model_name,
        config=student_config,
        num_topic_labels=teacher.num_topic_labels,
        num_sentiment_labels=teacher.num_sentiment_labels,
        ignore_mismatched_sizes=True
    )
    
    print(f"Student Architecture: {student_config.num_hidden_layers} layers, {student_config.hidden_size} hidden")

    # 4. Prepare Datasets
    train_ds = MultitaskDataset(
        train_df['text'].tolist(),
        train_df['topic_label'].tolist(),
        train_df['sentiment_label'].tolist(),
        tokenizer,
        args.max_length
    )
    val_ds = MultitaskDataset(
        val_df['text'].tolist(),
        val_df['topic_label'].tolist(),
        val_df['sentiment_label'].tolist(),
        tokenizer,
        args.max_length
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        report_to="none" # Disable wandb
    )

    # 6. Initialize DistillTrainer
    trainer = MultitaskDistillTrainer(
        teacher_model=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_multitask_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 7. Train
    print("Starting Distillation...")
    trainer.train()
    
    # 8. Save
    print(f"Saving distilled model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--teacher_path", required=True, help="Path to trained teacher model")
    parser.add_argument("--student_model_name", default="uer/chinese_roberta_L-4_H-512")
    parser.add_argument("--output_dir", default="models/trained/distill_multitask_student")
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=128)
    
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for KL Loss (0.5 means equal weight)")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_eval", type=int, default=0)
    
    args = parser.parse_args()
    run_distillation(args)
