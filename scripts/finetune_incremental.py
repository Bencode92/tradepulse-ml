#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental Fine-tuning with LoRA for TradePulse ML
Supports sentiment and importance classification with gating
"""
import os
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from packaging import version

import torch
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from huggingface_hub import login, HfApi

# Labels for each task
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
IMPORTANCE_LABELS = ["general", "important", "critical"]


def load_jsonl(path, task="sentiment"):
    """Load JSONL dataset for the specified task"""
    rows = []
    label_field = f"label_{task}" if task != "sentiment" else "label_sentiment"
    
    # Support both label and label_sentiment fields
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                text = r.get("text", "").strip()
                
                # Try different label field names
                label = r.get(label_field) or r.get("label")
                
                if text and label:
                    rows.append({"text": text, "label": label})
            except:
                continue
    
    return rows


def compute_metrics(eval_pred):
    """Compute comprehensive metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    
    # Detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision,
        "recall_weighted": recall
    }


def load_or_create_model(model_name, num_labels, id2label, label2id, incremental=False, 
                         lora_r=8, lora_alpha=16, lora_dropout=0.05):
    """Load base model and optionally apply LoRA configuration"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"Warning: Could not load {model_name}, falling back to distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    
    if incremental:
        # Apply LoRA configuration
        try:
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["query", "key", "value", "dense"],
                modules_to_save=["classifier"]  # Also save classifier for num_labels changes
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"Warning: Could not apply LoRA config: {e}")
            print("Continuing without LoRA...")
    
    return model


def check_metrics_gate(current_metrics, gate_drop=0.01, metrics_file="outputs/last_metrics.json"):
    """Check if current metrics pass the quality gate"""
    current_f1 = current_metrics.get("eval_f1_macro", 0.0)
    
    # Load previous metrics if available
    prev_f1 = 0.0
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                prev_metrics = json.load(f)
                prev_f1 = prev_metrics.get("f1_macro", 0.0)
        except:
            pass
    
    # Save current metrics
    os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump({"f1_macro": current_f1, "timestamp": datetime.now().isoformat()}, f)
    
    # Check gate
    ok_to_push = (prev_f1 - current_f1) <= gate_drop
    
    print(f"📊 Metrics Gate Check:")
    print(f"   Previous F1: {prev_f1:.4f}")
    print(f"   Current F1:  {current_f1:.4f}")
    print(f"   Difference:  {current_f1 - prev_f1:+.4f}")
    print(f"   Gate Drop:   {gate_drop:.4f}")
    print(f"   ✅ PASS" if ok_to_push else f"   ❌ FAIL")
    
    return ok_to_push


def main():
    parser = argparse.ArgumentParser(description="Incremental fine-tuning with LoRA")
    
    # Model arguments
    parser.add_argument("--model", default="distilbert-base-uncased",
                        help="Base model name from HuggingFace")
    parser.add_argument("--task", choices=["sentiment", "importance"], default="sentiment",
                        help="Classification task")
    parser.add_argument("--incremental", action="store_true",
                        help="Enable incremental learning with LoRA")
    
    # Data arguments
    parser.add_argument("--dataset", required=True,
                        help="Path to JSONL dataset")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Validation split ratio")
    
    # Training arguments
    parser.add_argument("--output_dir", default="outputs/incremental",
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Quality gate arguments
    parser.add_argument("--gate_drop", type=float, default=0.01,
                        help="Maximum allowed F1 score drop")
    parser.add_argument("--metrics_file", default="outputs/last_metrics.json",
                        help="File to store metrics history")
    
    # HuggingFace Hub arguments
    parser.add_argument("--hf_repo", default="",
                        help="HuggingFace repository to push model")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                        help="HuggingFace token")
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = f"{args.output_dir}/{args.task}/{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        try:
            login(token=args.hf_token)
        except:
            print("Warning: Could not login to HuggingFace")
    
    # Load data
    print(f"📥 Loading dataset from {args.dataset}...")
    data = load_jsonl(args.dataset, args.task)
    
    if not data:
        print("❌ No valid data found!")
        return
    
    # Get labels
    if args.task == "sentiment":
        labels = SENTIMENT_LABELS
    elif args.task == "importance":
        labels = IMPORTANCE_LABELS
    else:
        labels = sorted(list(set(d["label"] for d in data)))
    
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    # Convert labels to IDs
    for d in data:
        d["labels"] = label2id.get(d["label"], 0)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {len(data)}")
    print(f"   Labels: {labels}")
    
    # Create datasets
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=args.validation_split, seed=42)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except:
        print(f"Warning: Could not load tokenizer for {args.model}, using distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or "[PAD]"
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "label"])
    
    # Load model
    print(f"🤖 Loading model {args.model}...")
    model = load_or_create_model(
        args.model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        incremental=args.incremental,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Training arguments - compatible with both transformers 4.x and 5.x
    common_args = dict(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        push_to_hub=False,  # We'll handle this manually with gating
        save_total_limit=2,
        fp16=False,  # Disable for CPU
    )
    
    # Handle version differences for evaluation_strategy
    ta_kwargs = {}
    try:
        if version.parse(transformers.__version__).major >= 5:
            ta_kwargs.update(dict(eval_strategy="epoch", save_strategy="epoch"))
        else:
            ta_kwargs.update(dict(evaluation_strategy="epoch", save_strategy="epoch"))
    except:
        # Fallback to most common naming
        ta_kwargs.update(dict(evaluation_strategy="epoch", save_strategy="epoch"))
    
    try:
        training_args = TrainingArguments(**common_args, **ta_kwargs)
    except TypeError as e:
        # If evaluation_strategy fails, try without it
        print(f"Warning: {e}")
        print("Trying without evaluation strategy...")
        training_args = TrainingArguments(**common_args)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train
    print(f"🚀 Starting training...")
    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        print("Attempting to continue with minimal training...")
        # Try with minimal settings
        training_args.num_train_epochs = 1
        training_args.per_device_train_batch_size = 2
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        train_result = trainer.train()
    
    # Evaluate
    print(f"📊 Evaluating model...")
    eval_metrics = trainer.evaluate()
    
    # Save metrics
    metrics_output = {
        "task": args.task,
        "model": args.model,
        "incremental": args.incremental,
        "timestamp": datetime.now().isoformat(),
        "train_metrics": train_result.metrics if hasattr(train_result, 'metrics') else {},
        "eval_metrics": eval_metrics,
        "args": vars(args)
    }
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    
    print(f"\n📊 Evaluation Results:")
    for key, value in eval_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
    
    # Check quality gate
    if args.hf_repo:
        ok_to_push = check_metrics_gate(
            eval_metrics, 
            args.gate_drop, 
            f"{args.metrics_file}.{args.task}"
        )
        
        if ok_to_push:
            print(f"\n🚀 Pushing model to {args.hf_repo}...")
            
            # Save model locally first
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Push to hub
            try:
                if args.incremental:
                    # For LoRA models, push as a new revision
                    model.push_to_hub(
                        args.hf_repo,
                        commit_message=f"Update {args.task} model - F1: {eval_metrics.get('eval_f1_macro', 0):.4f}"
                    )
                else:
                    # For full models
                    trainer.push_to_hub(
                        repo_id=args.hf_repo,
                        commit_message=f"Update {args.task} model - F1: {eval_metrics.get('eval_f1_macro', 0):.4f}"
                    )
                
                print(f"✅ Model pushed to {args.hf_repo}")
            except Exception as e:
                print(f"⚠️ Could not push to HuggingFace: {e}")
        else:
            print(f"⛔ Model push skipped due to quality gate failure")
    
    print(f"\n✅ Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
