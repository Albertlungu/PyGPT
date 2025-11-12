import os
# CRITICAL: Set this BEFORE importing JAX anywhere!
# Force JAX to use CPU (Apple Metal GPU support is buggy and not production-ready)
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import sys
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
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    max_lines = 1000
    dataset = load_dataset("tatsu-lab/alpaca")

    train_data = dataset["train"]
    train_data = train_data.select(range(max_lines))

    training_texts = []
    with open("tokenizer_training_data/alpaca_sample_utf8.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            training_texts.append(line.strip())

    print("="*60)
    print("Appended training texts to list")
    print("="*60)

    # NEW: Include num_blocks and num_heads
    trainer = Trainer(
        tokenizer=tokenizer,
        user_input=training_texts,
        lr=1e-4,
        num_blocks=4,  # Stack 4 transformer blocks
        num_heads=8    # 8 attention heads per block
    )

    # Print model architecture summary
    print("="*60)
    print("MODEL SUMMARY")
    trainer.print_model_summary()
    print("="*60)

    print("Training model.")
    train_time = time.time()

    # Train with automatic checkpointing
    trainer.train(
        epochs=10,
        batch_size=50,
        checkpoint_path="artifacts/training_logs/jax_training_latest.pkl",
        save_every=10
    )

    end_train = time.time() - train_time
    print("Finished training model.")
    print("="*60)
    print(f"Train time: {end_train:.4f}s")

    # Test generation
    prompt = "What is 5+5?"
    generated_text = trainer.generate(prompt, max_length=50)
    print("="*60)
    print("Generated text:")
    print(generated_text)
    print("="*60)

def main():
    """Load a trained model and generate text."""

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Create trainer with same architecture as checkpoint
    dummy_input = ["dummy"]
    trainer = Trainer(
        tokenizer,
        dummy_input,
        lr=1e-4,
        num_blocks=4,  # Must match checkpoint!
        num_heads=8    # Must match checkpoint!
    )

    # Load the JAX checkpoint (NOT the old NumPy one!)
    checkpoint_path = "artifacts/training_logs/jax_training_latest.pkl"

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("You need to train a new model first!")
        print("Run: train() function to create a new JAX checkpoint")
        return

    # Generate text
    prompt = "Describe some of the benefits of a vegetarian diet."
    print("="*60)
    print(f"Prompt: {prompt}")
    print("="*60)

    generated_text = trainer.generate(
        prompt,
        max_length=50,
        temperature=0.7,
        top_k=40,
        repetition_penalty=1.5
    )

    print("Generated:")
    print(generated_text)
    print("="*60)

if __name__ == "__main__":
    start = time.time()

    # Choose what to run:
    # Option 1: Train a new model (creates JAX checkpoint)
    train()

    # Option 2: Load existing model and generate
    # main()

    end = time.time()
    print("="*60)
    print(f"Total execution time: {end-start:.4f} s")
    print("="*60)
