import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report
import time
import os

print("Starting eval_3.py")
# Configuration
MODEL_PATH = "ADD PATH"
TEST_PATH = "ADD PATH"
OUTPUT_PATH = "ADD PATH"

# Multiple equivalent hypotheses for immigration support
SUPPORT_HYPOTHESES = [
    "The comment supports immigration.",
    "The comment shows a positive stance toward immigration.",
    "The comment endorses immigration.",
    "The author of the comment favors immigration.",
    "The comment expresses support for immigration.",
    "The comment views immigration positively."
]

# NLI to stance mapping
NLI_TO_STANCE = {
    0: 2,  # Entailment -> Favor
    1: 1,  # Neutral -> Neutral
    2: 0   # Contradiction -> Against
}

STANCE_LABELS = ["Against", "Neutral", "Favor"]

print("Loading model and tokenizer...")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    model = model.to("cuda")
    print("Model moved to GPU")
else:
    print("No GPU available, using CPU")

model.eval()

print(f"Model loaded from {MODEL_PATH}")
print(f"Model type: {model.__class__.__name__}")
print(f"Device: {model.device}")

# First, count total examples to process
total_examples = 0
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('Stance_Label') is not None:
            total_examples += 1

print(f"Found {total_examples} examples with stance labels in {TEST_PATH}")

def get_premise(comment, parent):
    """
    Create premise by combining comment with parent comment.
    Adapted for test data format that doesn't include VideoID.
    """
    if parent == '' or parent == comment:
        return f"Comment: {comment}"
    return f"Context: {parent}\nComment: {comment}"

predictions = []
true_labels = []
start_time = time.time()

# Progress reporting settings
progress_interval = max(1, total_examples // 20) 
last_report_time = start_time

with torch.no_grad():
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            comment = item['CommentText']
            parent = item.get('ConText', '')
            label_raw = item.get('Stance_Label')
            
            if label_raw is None:
                continue
            
            # Calculate and report progress
            if (i+1) % progress_interval == 0 or i == 0 or i == total_examples-1:
                current_time = time.time()
                elapsed_time = current_time - start_time
                examples_processed = len(true_labels)
                
                if examples_processed > 0:
                    seconds_per_example = elapsed_time / examples_processed
                    remaining_examples = total_examples - examples_processed
                    estimated_time_remaining = remaining_examples * seconds_per_example
                    
                    # Only update if at least 2 seconds have passed since last report (prevents console spam)
                    if current_time - last_report_time > 2:
                        print(f"Progress: {examples_processed}/{total_examples} examples processed "
                              f"({examples_processed/total_examples*100:.1f}%) - "
                              f"ETA: {estimated_time_remaining/60:.1f} minutes")
                        last_report_time = current_time
            
            gold_stance = int(label_raw)
            true_labels.append(gold_stance)
            premise = get_premise(comment, parent)
            
            # Test with multiple hypotheses and average logits
            all_logits = []
            for hypothesis in SUPPORT_HYPOTHESES:
                inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                output = model(**inputs)
                all_logits.append(output.logits[0])
            
            # Average the logits across all hypotheses
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            nli_pred = avg_logits.argmax().item()
            
            # Convert NLI prediction to stance
            stance_pred = NLI_TO_STANCE[nli_pred]
            
            # Store individual hypothesis predictions for analysis
            hypothesis_preds = []
            for i, hypothesis in enumerate(SUPPORT_HYPOTHESES):
                hyp_nli_pred = all_logits[i].argmax().item()
                hyp_stance_pred = NLI_TO_STANCE[hyp_nli_pred]
                hypothesis_preds.append({
                    "hypothesis": hypothesis,
                    "nli_pred": int(hyp_nli_pred),
                    "stance_pred": int(hyp_stance_pred)
                })
            
            predictions.append({
                "predicted_stance": stance_pred,
                "predicted_nli": nli_pred,
                "gold_stance": gold_stance,
                "text": comment,
                "avg_logits": avg_logits.tolist(),
                "hypothesis_predictions": hypothesis_preds
            })

total_time = time.time() - start_time
print(f"\nEvaluation completed in {total_time/60:.2f} minutes")
print(f"Average time per example: {total_time/len(predictions):.3f} seconds")

# Write the predictions to file
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for p in predictions:
        f.write(json.dumps(p) + "\n")

print(f"Predictions saved to {OUTPUT_PATH}")

# Print classification report
pred_stances = [p["predicted_stance"] for p in predictions]
print("\nPrediction distribution:", np.bincount(pred_stances, minlength=3))
print("True label distribution:", np.bincount(true_labels, minlength=3))
print("\nClassification Report:")
print(classification_report(true_labels, pred_stances, 
                          target_names=STANCE_LABELS, 
                          digits=3))

# Analyze agreement between different hypotheses
agreement_count = 0
for pred in predictions:
    unique_preds = set(hp["stance_pred"] for hp in pred["hypothesis_predictions"])
    if len(unique_preds) == 1:
        agreement_count += 1

print(f"\nHypothesis agreement: {agreement_count}/{len(predictions)} examples ({agreement_count/len(predictions)*100:.1f}%)")
