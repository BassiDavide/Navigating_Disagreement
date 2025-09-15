import json
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# Configuration
MODEL_NAME = "ADD MODEL"
BASE_MODEL_SAVE_PATH = "ADD PATH"
DATA_PATH = "ADD PATH"
NUM_RUNS = 3

print(f"Model loaded: {MODEL_NAME}")

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print(f"Using {n_gpu} GPU(s)")
else:
    print("No GPU available, using CPU")
    device = torch.device("cpu")

STANCE_LABELS = ["Against", "Neutral", "Favor"]

def load_stance_data(path):
    """Load stance data - filter for parent_child and extract text from premise"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]
    
    # Filter for parent_child entries
    parent_child_lines = [line for line in lines if line['input_type'] == 'parent_child']
    print(f"Filtered {len(parent_child_lines)} parent_child entries from {len(lines)} total entries")
    
    # Extract text from premise (remove "Context: " prefix)
    processed_lines = []
    for line in parent_child_lines:
        premise = line['premise']
        # Remove "Context: " prefix if present
        if premise.startswith("Context: "):
            text = premise[9:]  # Remove "Context: " (9 characters)
        else:
            text = premise
        
        processed_lines.append({
            'text': text,
            'stance': line['original_stance']
        })
    
    return Dataset.from_list(processed_lines)

def tokenize_parent_child(example):
    """Tokenize the parent+child comment text"""
    return tokenizer(example['text'], 
                    truncation=True, padding='max_length', max_length=512)  # Increased max_length for parent+child

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    print("Predictions distribution:", np.bincount(preds, minlength=3))
    print("\nStance Classification Report:")
    print(classification_report(labels, preds, 
                               target_names=STANCE_LABELS, digits=3))
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def run_experiment(run_id):
    print(f"\n{'='*50}")
    print(f"STARTING RUN {run_id + 1}/{NUM_RUNS}")
    print(f"{'='*50}")
    
    # Set seed for reproducibility within each run
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    
    # Load and split data (keeping the same split with seed=42 for reproducibility)
    dataset = load_stance_data(DATA_PATH)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Load tokenizer and model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenization
    dataset = dataset.map(tokenize_parent_child, batched=True)
    dataset = dataset.rename_column("stance", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Load model (with 3 classes for stance)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # Training arguments (adjusted batch size for longer sequences)
    model_save_path = f"{BASE_MODEL_SAVE_PATH}_run_{run_id + 1}"
    training_args = TrainingArguments(
        output_dir=model_save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1.4663308655432806e-05,  
        per_device_train_batch_size=4,  # Reduced batch size for longer sequences
        gradient_accumulation_steps=32,  # Increased to maintain effective batch size
        per_device_eval_batch_size=2,   # Reduced eval batch size
        num_train_epochs=6,
        weight_decay=0.018633494087413147,
        warmup_ratio=0.033214041219251964, 
        fp16=True, 
        gradient_checkpointing=True, 
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        label_smoothing_factor=0.12124737647320344,
        report_to=[],
        seed=42 + run_id,
        data_seed=42 + run_id,
    )
    
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  
        early_stopping_threshold=0.0
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )
    
    print(f"Starting training for run {run_id + 1}...")
    trainer.train()
    
    # Save model
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Get final metrics
    final_metrics = trainer.evaluate()
    
    return {
        'run_id': run_id + 1,
        'accuracy': final_metrics['eval_accuracy'],
        'f1_macro': final_metrics['eval_f1_macro'],
        'model_path': model_save_path
    }

def main():
    print("Starting Parent+Child Stance Detection Experiments (No NLI)")
    print(f"Running {NUM_RUNS} experiments for averaging...")
    
    results = []
    
    for run_id in range(NUM_RUNS):
        result = run_experiment(run_id)
        results.append(result)
        print(f"\nRun {run_id + 1} Results:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1-Macro: {result['f1_macro']:.4f}")
    
    # Calculate averages and std
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print("\nIndividual Run Results:")
    for result in results:
        print(f"Run {result['run_id']}: Accuracy={result['accuracy']:.4f}, F1-Macro={result['f1_macro']:.4f}")
    
    print(f"\nAverage Performance:")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1-Macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    # Save results to file
    results_summary = {
        'individual_runs': results,
        'averages': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'f1_macro_mean': float(np.mean(f1_scores)),
            'f1_macro_std': float(np.std(f1_scores))
        }
    }
    
    with open('parent_child_training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to parent_child_training_results.json")

if __name__ == "__main__":
    main()
