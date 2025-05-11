import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import gc

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_model(model_path):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def generate_summary(text, model, tokenizer, min_length=150, max_length=450):
    # Clear memory before generation
    clear_memory()
    
    print("Tokenizing input text...")
    # Tokenize input text
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    
    print("Generating summary...")
    # Generate summary with optimized parameters
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=5,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    print("Decoding summary...")
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    print("Post-processing summary...")
    # Post-process summary to ensure it's within length constraints
    sentences = sent_tokenize(summary)
    word_count = len(summary.split())
    
    if word_count > max_length:
        print(f"Summary too long ({word_count} words), truncating...")
        # Remove sentences from the end until we're within the max length
        while word_count > max_length and len(sentences) > 1:
            sentences.pop()
            summary = " ".join(sentences)
            word_count = len(summary.split())
    
    print(f"Final summary length: {word_count} words")
    return summary

def main():
    # Load the trained model
    model_path = "./legal_summarization_model_final"
    model, tokenizer = load_model(model_path)
    
    # Example legal text
    text = """
    [Your legal text here]
    """
    
    print("\nGenerating summary...")
    summary = generate_summary(text, model, tokenizer)
    
    print("\nGenerated Summary:")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    print(f"\nSummary length: {len(summary.split())} words")

if __name__ == "__main__":
    main() 