#!/usr/bin/env python3
"""
Quick script to count parameters in the model without training.
"""
import pickle
import sys
sys.path.append('src')

from src.training.train import Trainer
from src.tokenizer.tokenizer_class import BPETokenizer

def main():
    # Load tokenizer
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Create a dummy trainer with minimal data
    dummy_text = ["Hello world"]

    print("Initializing model...")
    trainer = Trainer(tokenizer, dummy_text, lr=1e-4)

    # Print model summary
    trainer.print_model_summary()

    # Manual calculation breakdown
    print("\nMANUAL CALCULATION BREAKDOWN:")
    print("="*60)

    vocab_size = tokenizer.vocab_size
    embed_dim = 512
    ff_dim = embed_dim * 4

    print(f"\n1. EMBEDDING LAYER:")
    print(f"   Embedding matrix: {vocab_size} × {embed_dim} = {vocab_size * embed_dim:,}")

    print(f"\n2. ATTENTION LAYER:")
    print(f"   W_Q: {embed_dim} × {embed_dim} = {embed_dim * embed_dim:,}")
    print(f"   W_K: {embed_dim} × {embed_dim} = {embed_dim * embed_dim:,}")
    print(f"   W_V: {embed_dim} × {embed_dim} = {embed_dim * embed_dim:,}")
    print(f"   W_O: {embed_dim} × {embed_dim} = {embed_dim * embed_dim:,}")
    print(f"   Subtotal: {4 * embed_dim * embed_dim:,}")

    print(f"\n3. FEEDFORWARD LAYER:")
    print(f"   W1: {embed_dim} × {ff_dim} = {embed_dim * ff_dim:,}")
    print(f"   B1: {ff_dim} = {ff_dim:,}")
    print(f"   W2: {ff_dim} × {embed_dim} = {ff_dim * embed_dim:,}")
    print(f"   B2: {embed_dim} = {embed_dim:,}")
    print(f"   Subtotal: {embed_dim * ff_dim + ff_dim + ff_dim * embed_dim + embed_dim:,}")

    print(f"\n4. LAYER NORMALIZATION:")
    print(f"   Gamma_1: {embed_dim} = {embed_dim:,}")
    print(f"   Beta_1:  {embed_dim} = {embed_dim:,}")
    print(f"   Gamma_2: {embed_dim} = {embed_dim:,}")
    print(f"   Beta_2:  {embed_dim} = {embed_dim:,}")
    print(f"   Subtotal: {4 * embed_dim:,}")

    print(f"\n5. OUTPUT LAYER:")
    print(f"   W_out: {embed_dim} × {vocab_size} = {embed_dim * vocab_size:,}")
    print(f"   B_out: {vocab_size} = {vocab_size:,}")
    print(f"   Subtotal: {embed_dim * vocab_size + vocab_size:,}")

    # Total
    total = (vocab_size * embed_dim +
             4 * embed_dim * embed_dim +
             embed_dim * ff_dim + ff_dim + ff_dim * embed_dim + embed_dim +
             4 * embed_dim +
             embed_dim * vocab_size + vocab_size)

    print(f"\n{'='*60}")
    print(f"GRAND TOTAL: {total:,} parameters")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
