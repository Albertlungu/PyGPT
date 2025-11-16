import os
import numpy as np
import sys
import jax
import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset

import src.utils.config
from embeddings.embeddings import EmbeddingLayer
from tokenizer.tokenizer_class import BPETokenizer
from training.train import Trainer

def train():
    """Train a new model from scratch with JAX architecture."""

    print("This code is running.")
    # Load tokenizer
    with open("artifacts/tokenizer/tokenizer_general_knowledge.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # max_lines = 10000
    # dataset = load_dataset("tatsu-lab/alpaca")

    # train_data = dataset["train"]
    # train_data = train_data.select(range(max_lines))
    
    # with open("tokenizer_training_data/alpaca_sample_utf8.txt", "w", encoding="utf-8") as f:
    #     for ex in train_data:
    #         text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {ex['output']}\n"
    #         f.write(text + "\n")

    print(jax.devices())
    print(jax.default_backend())


    # Load complete documents (separated by double newlines)
    # This ensures related content stays together
    with open("training_data/general_knowledge.txt", "r") as f:
        content = f.read()

    # Split by double newlines to get complete instruction-response pairs
    training_texts = [doc.strip() for doc in content.split('\n\n') if doc.strip()]

    print("="*60)
    print("Appended training texts to list")
    print("="*60)

    trainer = Trainer(
        tokenizer=tokenizer,
        training_data=training_texts,
        lr=5e-4,
        num_blocks=8,
        num_heads=8,
        max_seq_length=256  # Limit sequence length to avoid memory issues
    )

    # Print model architecture summary
    print("="*60)
    print("MODEL SUMMARY")
    trainer.print_model_summary()
    print("="*60)

    print("Training model.")
    train_time = time.time()

    # Train with automatic checkpointing
    # FAST TEST CONFIGURATION
    trainer.train(
        epochs=10,       # Reduced from 10 to 2 epochs
        batch_size=32,  
        checkpoint_path="artifacts/training_logs/jax_gen_kn.pkl",
        save_every=1    # Save every 2 epochs
    )

    end_train = time.time() - train_time
    print("Finished training model.")
    print("="*60)
    print(f"Train time: {end_train:.4f}s")

    # Test generation
    prompt = "What is 5+5?"
    generated_text = trainer.generate(prompt, max_length=500)
    print("="*60)
    print("Generated text:")
    print(generated_text[0])
    print("="*60)

def extend():

    with open("artifacts/tokenizer/tokenizer_general_knowledge.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Load complete documents (separated by double newlines)
    with open("training_data/general_knowledge.txt", "r", encoding="utf-8") as f:
        content = f.read()

    training_texts = [doc.strip() for doc in content.split('\n\n') if doc.strip()]


    trainer = Trainer(
        tokenizer,
        training_texts,
        lr=5e-4,  # Match the original training learning rate
        num_blocks=8,  # Must match checkpoint!
        num_heads=8,   # Must match checkpoint!
        max_seq_length=256
    )

    trainer.extend_training(
        checkpoint_path="artifacts/training_logs/jax_gen_kn_2025-11-15_20-02-17.pkl",
        epochs=50,
        batch_size=32,
        save_every=1
    )


def main():
    """Load a trained model and generate text."""

    with open("artifacts/tokenizer/tokenizer_general_knowledge.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Create trainer with same architecture as checkpoint
    dummy_input = ["dummy"]
    trainer = Trainer(
        tokenizer,
        dummy_input,
        lr=5e-4,
        num_blocks=8,  # Must match checkpoint!
        num_heads=8   # Must match checkpoint!
    )

    checkpoint_path = "artifacts/training_logs/jax_gk_12epochs.pkl"

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("You need to train a new model first!")
        print("Run: train() function to create a new JAX checkpoint")
        return

    # Test multiple prompts
    prompts = [
        "Design an algorithm to detect plagiarism in academic papers.",
        "Prove that 2 squared is equal to 4.",
        "Write a convincing argument in favor of using GPT models.",
        "Who invented the first successful automobile?",
        "Summarize how quantum computing works."
    ]

    for prompt in prompts:
        print("="*60)
        print(f"Prompt: {prompt}")
        print("="*60)

        generated_text = trainer.generate(
            prompt,
            max_length=100,
            temperature=1,
            top_k=30,
            repetition_penalty=1.5,
            debug=False
        )

        print(f"Generated: '{generated_text[0]}'")
        print()

if __name__ == "__main__":
    start = time.time()

    main_or_train = input("M/T/E? ")
    if main_or_train.lower() == 't':
        train()
    elif main_or_train.lower() == 'e':
        extend()
    else:
        main()

    end = time.time()
    print("="*60)
    print(f"Total execution time: {end-start:.4f} s")
    print("="*60)