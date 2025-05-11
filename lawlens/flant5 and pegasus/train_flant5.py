import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from datasets import Dataset
import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize
import gc

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# GPU memory monitoring
def print_gpu_utilization():
    print("GPU memory allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("GPU memory cached:", torch.cuda.memory_reserved() / 1024**2, "MB")

# Custom callback for logging
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")

# Load and preprocess data
def load_data(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    return Dataset.from_pandas(df)

# Tokenization function
def preprocess_function(examples, tokenizer, max_input_length=1024, max_target_length=450):
    inputs = [doc for doc in examples["text"]]
    targets = [summary for summary in examples["summary"]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding="max_length"
    )
    
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Compute metrics
def compute_metrics(eval_preds):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    preds, labels = eval_preds
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    rouge_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(pred, label)
        rouge_scores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    # Calculate average scores
    avg_scores = {
        'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
        'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
        'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
    }
    
    return avg_scores

def main():
    print("Starting training process...")
    print_gpu_utilization()
    
    # Load model and tokenizer
    model_name = "google/flan-t5-large"  # Using large model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Load datasets
    train_dataset = load_data("lawlens/dataset/train.csv")
    test_dataset = load_data("lawlens/dataset/test.csv")
    
    print("Preprocessing datasets...")
    # Preprocess datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    print("Setting up training arguments...")
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./legal_summarization_model",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL"
    )
    
    print("Setting up data collator...")
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )
    
    print("Initializing trainer...")
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[LoggingCallback()]
    )
    
    print("Starting training...")
    # Train the model
    trainer.train()
    
    print("Saving model...")
    # Save the model
    trainer.save_model("./legal_summarization_model_final")
    tokenizer.save_pretrained("./legal_summarization_model_final")
    
    print("Training completed!")
    print_gpu_utilization()

if __name__ == "__main__":
    main() 