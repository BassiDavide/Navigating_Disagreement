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

def load_test_data(path):
    """Load test data - format with ParentCommentText and CommentText, combine them"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f]
    
    print(f"Loaded {len(lines)} test entries")
    
    # Extract and combine ParentCommentText + CommentText
    processed_lines = []
    for line in lines:
        # Combine parent and child text (parent first, then child)
        parent_text = line['ParentCommentText']
        child_text = line['CommentText']
        
        # Concatenate with a separator (newline to maintain logical flow)
        combined_text = f"{parent_text}\n{child_text}"
        
        processed_lines.append({
            'text': combined_text,
            'stance': line['Stance_Label']
        })
    
    return processed_lines

def evaluate_model(model_path, test_dataset, tokenizer):
    """Evaluate a single model on the test dataset"""
    print(f"Loading model from {model_path}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Prepare predictions
    predictions = []
    true_labels = []
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for i, example in enumerate(test_dataset):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(test_dataset)}")
                
            # Tokenize (using same max_length as training)
            inputs = tokenizer(example['text'], 
                             truncation=True, padding='max_length', max_length=512,
                             return_tensors='pt')
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            predictions.append(pred)
            true_labels.append(example['stance'])
    
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
    
    # Print label distributions
    print(f"\nTrue label distribution: {np.bincount(true_labels)}")
    print(f"Predicted label distribution: {np.bincount(predictions)}")
    
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
    print("Evaluating Parent+Child Stance Detection Models (No NLI)")
    print(f"Evaluating {NUM_RUNS} trained models...")
    
    # Load test data
    test_dataset = load_test_data(TEST_DATA_PATH)
    print(f"Loaded {len(test_dataset)} test samples")
    
    # Sample first few entries to verify the text combination
    print("\nSample of combined text (first 3 entries):")
    for i, sample in enumerate(test_dataset[:3]):
        print(f"\nSample {i+1}:")
        print(f"Combined text: {sample['text'][:200]}...")
        print(f"Stance: {sample['stance']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    
    results = []
    
    for run_id in range(NUM_RUNS):
        model_path = f"{MODEL_BASE_PATH}_run_{run_id + 1}"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist. Skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"EVALUATING RUN {run_id + 1}/{NUM_RUNS}")
        print(f"{'='*50}")
        
        result = evaluate_model(model_path, test_dataset, tokenizer)
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
    
    with open('parent_child_evaluation_results.json', 'w') as f:
        json.dump(evaluation_summary, f, indent=2, default=str)
    
    print(f"\nDetailed evaluation results saved to parent_child_evaluation_results.json")

if __name__ == "__main__":
    main()
