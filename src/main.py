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
    with open("artifacts/tokenizer_dolly_15k.pkl", "rb") as f:
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

    with open("artifacts/tokenized_dolly_15k.pkl", "rb") as f:
        tokenized_data = pickle.load(f)

    print("="*60)
    print("Appended training texts to list")
    print("="*60)

    trainer = Trainer(
        tokenizer=tokenizer,
        pretokenized_data=tokenized_data,
        lr=5e-4,
        num_blocks=1,
        num_heads=4
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
        batch_size=4,  
        checkpoint_path="artifacts/training_logs/jax_dolly_15k.pkl",
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

    with open("artifacts/tokenizer_dolly_15k.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    training_texts = []
    with open("training_data/pygpt_training_corpus.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            training_texts.append(line.strip())


    trainer = Trainer(
        tokenizer,
        training_texts,
        lr=1e-3,  # Match the original training learning rate
        num_blocks=4,  # Must match checkpoint!
        num_heads=4   # Must match checkpoint!
    )

    trainer.extend_training(
        checkpoint_path="artifacts/training_logs/jax_32k_corpus_2025-11-14_11-23-51.pkl",
        epochs=10,
        batch_size=32,
        save_every=1
    )


# @jax.jit(device=jax.devices("cpu")[0])
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
        lr=1e-3,
        num_blocks=4,  # Must match checkpoint!
        num_heads=4   # Must match checkpoint!
    )

    checkpoint_path = "artifacts/training_logs/jax_32k_corpus_2025-11-14_11-23-51.pkl"

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("You need to train a new model first!")
        print("Run: train() function to create a new JAX checkpoint")
        return

    # Test multiple prompts
    prompts = [
        "What is your name?",
        "What is 5+5?",
        "Document the steps needed to deploy a machine learning model in an Android application.",
        "The capital of France is",
        "Rewrite the sentence 'The cat chased the mouse' in the past tense."
    ]

    for prompt in prompts:
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