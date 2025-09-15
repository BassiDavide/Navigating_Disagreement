import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import os
import glob

print("Starting batch inference script...")

# Configuration
MODEL_PATH = "ADD PATH"
INPUT_DIR = "ADD PATH"
INPUT_PATTERN = "*.jsonl"

# Hypothesis for NLI
HYPOTHESIS = "The Comment supports immigration."

# NLI to stance mapping
NLI_TO_STANCE = {
    0: 2,  # Entailment -> Favor
    1: 1,  # Neutral -> Neutral
    2: 0   # Contradiction -> Against
}

STANCE_LABELS = ["Against", "Neutral", "Favor"]

print("Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if torch.cuda.is_available():
    model = model.to("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU")

model.eval()

def get_child_only_premise(comment):
    return f"Comment: {comment}"

def get_parent_child_premise(comment, parent):
    if parent == '' or parent == comment:
        return f"Comment: {comment}"
    return f"Context: {parent}\nComment: {comment}"

def process_file(input_path, output_path):
    """Process a single JSONL file and save labeled results."""
    print(f"\nProcessing: {input_path}")
    
    # Count total examples
    with open(input_path, 'r', encoding='utf-8') as f:
        total_examples = sum(1 for line in f)
    
    print(f"Found {total_examples} examples to label")
    
    labeled_data = []
    file_start_time = time.time()
    
    with torch.no_grad():
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                comment = item['CommentText']
                parent = item.get('ParentCommentText', '')
                
                if (i+1) % 100 == 0:
                    print(f"Progress: {i+1}/{total_examples} examples")
                
                # Create premises
                child_premise = get_child_only_premise(comment)
                parent_child_premise = get_parent_child_premise(comment, parent)
                
                # Child-only prediction
                child_inputs = tokenizer(child_premise, HYPOTHESIS, 
                                        return_tensors='pt', truncation=True, max_length=512)
                if torch.cuda.is_available():
                    child_inputs = {k: v.to("cuda") for k, v in child_inputs.items()}
                child_output = model(**child_inputs)
                child_nli_pred = child_output.logits[0].argmax().item()
                child_stance_pred = NLI_TO_STANCE[child_nli_pred]
                
                # Parent-child prediction
                parent_inputs = tokenizer(parent_child_premise, HYPOTHESIS, 
                                         return_tensors='pt', truncation=True, max_length=512)
                if torch.cuda.is_available():
                    parent_inputs = {k: v.to("cuda") for k, v in parent_inputs.items()}
                parent_output = model(**parent_inputs)
                parent_nli_pred = parent_output.logits[0].argmax().item()
                parent_stance_pred = NLI_TO_STANCE[parent_nli_pred]
                
                # Add predictions to original item
                item['predicted_stance_child_only'] = child_stance_pred
                item['predicted_stance_parent_child'] = parent_stance_pred
                item['predicted_stance_label_child_only'] = STANCE_LABELS[child_stance_pred]
                item['predicted_stance_label_parent_child'] = STANCE_LABELS[parent_stance_pred]
                
                labeled_data.append(item)
    
    file_time = time.time() - file_start_time
    print(f"File processed in {file_time/60:.2f} minutes")
    
    # Save labeled data
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in labeled_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Labeled data saved to {output_path}")
    
    # Print prediction distribution for this file
    child_only_counts = {label: 0 for label in STANCE_LABELS}
    parent_child_counts = {label: 0 for label in STANCE_LABELS}
    
    for item in labeled_data:
        child_only_counts[item['predicted_stance_label_child_only']] += 1
        parent_child_counts[item['predicted_stance_label_parent_child']] += 1
    
    print("Prediction distribution for this file:")
    print("Child-only method:", child_only_counts)
    print("Parent-child method:", parent_child_counts)
    
    return len(labeled_data)

# Find all input files
input_pattern = os.path.join(INPUT_DIR, INPUT_PATTERN)
input_files = glob.glob(input_pattern)

if not input_files:
    print(f"No files found matching pattern: {input_pattern}")
    exit(1)

print(f"Found {len(input_files)} files to process:")
for file_path in input_files:
    print(f"  - {file_path}")

total_processed = 0
overall_start_time = time.time()

# Process each file
for input_path in input_files:
    # Create output filename with "Anno_" prefix
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    output_filename = f"Anno_{input_filename}"
    output_path = os.path.join(input_dir, output_filename)
    
    # Process the file
    file_count = process_file(input_path, output_path)
    total_processed += file_count

total_time = time.time() - overall_start_time
print(f"\n=== SUMMARY ===")
print(f"Total files processed: {len(input_files)}")
print(f"Total examples labeled: {total_processed}")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Average time per file: {(total_time/len(input_files))/60:.2f} minutes")
