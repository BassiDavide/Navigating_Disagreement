import json
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os

# Configuration
MODEL_BASE_PATH = "ADD PATH"
TEST_DATA_PATH = "ADD PATH"
NUM_RUNS = 3

STANCE_LABELS = ["Against", "Neutral", "Favor"]

def load_stance_data(path):
    """Load stance data - expecting format with 'text' and 'stance' fields"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]
    return Dataset.from_list(lines)

def tokenize_child_only(tokenizer, example):
    """Tokenize only the child comment text"""
    return tokenizer(example['CommentText'], 
                    truncation=True, padding='max_length', max_length=256)

def evaluate_model(model_path, test_dataset):
    """Evaluate a single model on the test dataset"""
    print(f"Loading model from {model_path}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Debug: Check vocabulary sizes
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Prepare predictions
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            # Tokenize
            inputs = tokenizer(example['CommentText'], 
                             truncation=True, padding='max_length', max_length=256,
                             return_tensors='pt')
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            predictions.append(pred)
            true_labels.append(example['Stance_Label'])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=STANCE_LABELS, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': predictions,
        'true_labels': true_labels,
        'classification_report': classification_report(true_labels, predictions, 
                                                     target_names=STANCE_LABELS, 
                                                     output_dict=True),
        'confusion_matrix': cm.tolist()
    }

def main():
    print("Evaluating Child-Only Stance Detection Models")
    print(f"Evaluating {NUM_RUNS} trained models...")
    
    # Load test data (your actual test set)
    test_dataset = load_stance_data(TEST_DATA_PATH)
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Load tokenizer
    
    
    results = []
    
    for run_id in range(NUM_RUNS):
        model_path = f"{MODEL_BASE_PATH}_run_{run_id + 1}"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist. Skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"EVALUATING RUN {run_id + 1}/{NUM_RUNS}")
        print(f"{'='*50}")
        
        result = evaluate_model(model_path, test_dataset)
        result['run_id'] = run_id + 1
        result['model_path'] = model_path
        results.append(result)
    
    if not results:
        print("No models found for evaluation!")
        return
    
    # Calculate averages and std
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print("\nIndividual Run Results:")
    for result in results:
        print(f"Run {result['run_id']}: Accuracy={result['accuracy']:.4f}, F1-Macro={result['f1_macro']:.4f}")
    
    print(f"\nAverage Performance:")
    print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"F1-Macro: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    # Calculate average confusion matrix
    avg_cm = np.mean([r['confusion_matrix'] for r in results], axis=0)
    print(f"\nAverage Confusion Matrix:")
    print(avg_cm)
    
    # Save detailed results
    evaluation_summary = {
        'individual_runs': [
            {
                'run_id': r['run_id'],
                'accuracy': r['accuracy'],
                'f1_macro': r['f1_macro'],
                'model_path': r['model_path']
            } for r in results
        ],
        'averages': {
            'accuracy_mean': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'f1_macro_mean': float(np.mean(f1_scores)),
            'f1_macro_std': float(np.std(f1_scores))
        },
        'average_confusion_matrix': avg_cm.tolist(),
        'detailed_results': results  # Full results for further analysis
    }
    
    with open('child_only_evaluation_results.json', 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    
    print(f"\nDetailed evaluation results saved to child_only_evaluation_results.json")

if __name__ == "__main__":
    main()
