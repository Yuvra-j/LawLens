LawLens - Legal Document Summarizer

LawLens is a tool designed to help legal professionals and researchers quickly understand lengthy legal case documents. It uses a fine-tuned version of the FLAN-T5 model to generate high-quality summaries of complex legal texts.

Project Purpose:

This project aims to solve the problem of time-consuming manual analysis of legal case files by offering AI-powered summarization. LawLens can assist in quickly getting to the essence of long legal judgments or filings.

Features:

- Summarizes long legal documents in plain language
- Fine-tuned FLAN-T5-Large model (or Base for constrained environments)
- Outputs summaries of up to 512 words
- Optimized to run on GPUs with 14 GB VRAM
- Dataset used: d0r1h/ILC (Indian Legal Cases)

Project Structure:

- data/ → Contains training and test samples in JSON or CSV
- model/ → Fine-tuning scripts, configs, checkpoints
- src/ → Preprocessing and summarization pipeline
- outputs/ → Generated summaries, logs, and results
- notebooks/ → Training and evaluation notebooks (optional)

Technologies Used:

- Python 3.10+
- Hugging Face Transformers
- Datasets (HF datasets library)
- PyTorch
- Accelerate (for memory-efficient training)
- TQDM, Panda
