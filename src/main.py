import os
# import numpy as np
import sys
# import jax
# import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
# from datasets import load_dataset

import src.utils.config
from embeddings.embeddings import EmbeddingLayer
from src.tokenizer.tiktoken_tokenizer import TikToken
from tokenizer.tokenizer_class import BPETokenizer
from training.train import Trainer

def load_BPE(tokenizer_path:str) -> None:
    """
    Loading the custom BPE tokenizer

    Args:
        tokenizer_path (str): Path to the pickled tokenizer file
    """
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

def save_token_ids(output_path:str) -> None:
    """
    Save token ids to a pickled file in order to avoid 10 minutes of prep time during training.
    """

    with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    with open("training_data/alpaca.txt", "r") as f:
        content = f.read()

    # Split by double newlines to get complete instruction-response pairs
    training_texts = [doc.strip() for doc in content.split('\n\n') if doc.strip()]

    token_ids = []
    for text in tqdm(training_texts):
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id)
        token_ids.append(ids)

    with open(output_path, "wb") as f:
        pickle.dump(token_ids, f)

def train() -> None:
    """
    Train a new model from scratch with JAX architecture.
    """

    print("This code is running.")

    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    with open("training_data/tiktoken_alpaca.pkl", "rb") as f:
        token_ids = pickle.load(f)

    print("="*60)
    print("Appended training texts to list")
    print("="*60)

    trainer = Trainer(
        tokenizer=tokenizer,
        token_ids=token_ids,
        lr=6e-4,  # Slightly higher base LR with schedule
        num_blocks=2,
        num_heads=2,
        embedding_dim=128,  # Must be divisible by num_heads
        max_seq_length=256,  # Chunk long sequences to avoid memory issues
        dropout=0.0,
        use_lr_schedule=True,  # Enable warmup + cosine decay
        warmup_steps=500  # Warmup for first 500 steps
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
        epochs=20,       # Reduced from 10 to 2 epochs
        batch_size=32,
        checkpoint_path="artifacts/training_logs/alpaca284.pkl",
        save_every=1,
        prompt="Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"    # Save every 2 epochs
    )

    end_train = time.time() - train_time
    print("Finished training model.")
    print("="*60)
    print(f"Train time: {end_train:.4f}s")

    # Save lightweight model-only checkpoint (no optimizer state)
    print("\nCreating lightweight checkpoint for inference...")
    trainer.save_model_only("artifacts/models/alpaca200_model_only.pkl")
    print("Lightweight checkpoint saved!")

    # Test generation
    prompt = "Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"
    generated_text = trainer.generate(prompt, max_length=200)
    print("="*60)
    print("Generated text:")
    print(generated_text)
    print("="*60)

def extend() -> None:
    """
    Extend model training using the Trainer.extend() function
    """

    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    # Load complete documents (separated by double newlines)
    with open("training_data/general_knowledge.txt", "r", encoding="utf-8") as f:
        content = f.read()

    training_texts = [doc.strip() for doc in content.split('\n\n') if doc.strip()]


    trainer = Trainer(
        tokenizer,
        training_texts,
        lr=5e-4,  # Match the original training learning rate
        num_blocks=12,  # Must match checkpoint!
        num_heads=12,   # Must match checkpoint!
        embedding_dim=768,  # Must match checkpoint!
        max_seq_length=512
    )

    trainer.extend_training(
        checkpoint_path="artifacts/training_logs/jax_gen_kn_2025-11-15_20-02-17.pkl",
        epochs=50,
        batch_size=32,
        save_every=1
    )


def main():
    """
    Load a trained model and generate text.
    """

    print("Loading tokenizer...")
    with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    # Create trainer with same architecture as checkpoint
    dummy_input = ["dummy"]
    trainer = Trainer(
        tokenizer,
        dummy_input,
        lr=5e-4,
        num_blocks=2,  # Must match checkpoint!
        num_heads=2,   # Must match checkpoint!
        embedding_dim=128  # Must match checkpoint!
    )

    # Use model-only checkpoint for faster loading (or fall back to full checkpoint)
    model_only_path = "artifacts/models/alpaca200_model_only.pkl"
    full_checkpoint_path = "artifacts/training_logs/alpaca284_2025-11-22_19-53-01.pkl"

    # Try model-only first (smaller, faster)
    import os
    if os.path.exists(model_only_path):
        checkpoint_path = model_only_path
        print("Loading lightweight model-only checkpoint...")
    else:
        checkpoint_path = full_checkpoint_path
        print("Loading full checkpoint (includes optimizer state)...")

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("You need to train a new model first!")
        print("Run: train() function to create a new JAX checkpoint")
        return


    prompts = [
        "Instruction: Construct an analogy based on the given two words. \nInput: Air, Water. \nOutput: ",
        "Instruction: What is the major contribution of the philosopher Immanuel Kant?\nInput: \nOutput: ",
        "Instruction: Create a list of 20 vocabulary words related to marine animals.\nInput: \nOutput: ",
        "Instruction: Write a description of the Golden Gate Bridge.\nInput: \nOutput: ",
        "Instruction: List three best practices for starting a conversation.\nInput: \nOutput: ",
        "Instruction: Explain the importance of NASA's current mission to Mars.\nInput: \nOutput: "
    ]

    for prompt in prompts:
        print("="*60)
        print(f"Prompt: {prompt}")
        print("="*60)

        generated_text = trainer.generate(
            prompt,
            max_length=150,
            temperature=10,
            top_k=30,
            repetition_penalty=1.5,
            debug=False
        )

        print(f"Generated: '{generated_text}'")
        print()


if __name__ == "__main__":
    print("Hello World - Starting PyGPT")
    start = time.time()

    main_or_train = input("M/T/E/ids? ")
    if main_or_train.lower() == 't':
        train()
    elif main_or_train.lower() == 'e':
        extend()
    elif main_or_train.lower() == "ids":
        save_token_ids("training_data/alpaca_tokenized.pkl")
    else:
        main()

    end = time.time()
    print("="*60)
    print(f"Total execution time: {end-start:.4f} s")
    print("="*60)
