import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import os

# Configuration
MODEL_NAME = "microsoft/deberta-v3-large"
BASE_MODEL_SAVE_PATH = "NLI_DeBERTa_V3"
DATA_PATH = "ADD PATH"
NUM_RUNS = 3 
RANDOM_SEEDS = [42, 123, 456]  #Seeds for each run

print(f"Model loaded: {MODEL_NAME}")
print(f"Will train {NUM_RUNS} models with seeds: {RANDOM_SEEDS}")

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print(f"Using {n_gpu} GPU(s)")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")

# NLI to stance mapping (for evaluation metrics)
NLI_TO_STANCE = {
    0: 2,  # Entailment -> Favor
    1: 1,  # Neutral -> Neutral
    2: 0   # Contradiction -> Against
}

STANCE_LABELS = ["Against", "Neutral", "Favor"]
INPUT_TYPES = ["child_only", "parent_child"]

def load_nli_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]
    return Dataset.from_list(lines)

# Load and tokenize data once (same for all runs)
print("Loading and tokenizing data...")
dataset = load_nli_data(DATA_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(example['premise'], example['hypothesis'], 
                    truncation=True, padding='max_length', max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'original_stance', 'input_type'])

# Training loop
for run_idx in range(NUM_RUNS):
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING RUN {run_idx + 1}/{NUM_RUNS}")
    print(f"Random seed: {RANDOM_SEEDS[run_idx]}")
    print(f"{'='*60}")
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEEDS[run_idx])
    np.random.seed(RANDOM_SEEDS[run_idx])
    
    # Create train/test split with current seed
    dataset_split = dataset.train_test_split(test_size=0.1, seed=RANDOM_SEEDS[run_idx])
    
    # Model save path for this run
    model_save_path = f"{BASE_MODEL_SAVE_PATH}_run_{run_idx + 1}"
    
    # Load model (with 3 classes for NLI)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.hidden_dropout_prob = 0.25
    config.attention_probs_dropout_prob = 0.25
    config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1.2e-05, 
        per_device_train_batch_size=8, 
        gradient_accumulation_steps=16, 
        per_device_eval_batch_size=4, 
        num_train_epochs=5,
        weight_decay=0.05, 
        warmup_ratio=0.06, 
        fp16=True, 
        gradient_checkpointing=True, 
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        label_smoothing_factor=0.15, 
        report_to=[],
        seed=RANDOM_SEEDS[run_idx]
    )

    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Weight decay: {training_args.weight_decay}")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=1,  #moved to 1 to avoid overfitting
        early_stopping_threshold=0.005 #moved to 0.005 to avoid overfitting
    )

    # Metrics (convert back to stance for reporting)
    def compute_metrics(pred):
        nli_labels = pred.label_ids
        nli_preds = pred.predictions.argmax(-1)
        
        # Convert NLI predictions back to stance labels for metrics
        stance_preds = np.array([NLI_TO_STANCE[p] for p in nli_preds])
        stance_labels = np.array([example["original_stance"] for example in dataset_split["test"]])
        
        # Report both NLI and stance distributions
        print("NLI predictions distribution:", np.bincount(nli_preds, minlength=3))
        print("Stance predictions distribution:", np.bincount(stance_preds, minlength=3))
        print("\nStance Classification Report:")
        print(classification_report(stance_labels, stance_preds, 
                                   target_names=STANCE_LABELS, digits=3))
        
        return {
            "accuracy": accuracy_score(stance_labels, stance_preds),
            "f1_macro": f1_score(stance_labels, stance_preds, average="macro"),
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

    print(f"Starting training for run {run_idx + 1}...")
    trainer.train()
    trainer.save_model(model_save_path)
    print(f"Model {run_idx + 1} saved to {model_save_path}")

print(f"\n{'='*60}")
print("ALL TRAINING RUNS COMPLETED!")
print(f"Trained {NUM_RUNS} models:")
for i in range(NUM_RUNS):
    print(f"  - {BASE_MODEL_SAVE_PATH}_run_{i + 1}")
print(f"{'='*60}")
