#!/usr/bin/env python3
"""
Convert a full checkpoint (with optimizer state) to a model-only checkpoint.
This reduces file size by ~2-3x for faster loading during inference.
"""

import pickle
import sys

def convert_checkpoint(input_path, output_path):
    """Convert full checkpoint to model-only checkpoint."""
    print(f"Loading full checkpoint from {input_path}...")

    with open(input_path, "rb") as f:
        full_checkpoint = pickle.load(f)

    # Create model-only checkpoint (no optimizer state)
    model_only = {
        'embeddings': full_checkpoint['embeddings'],
        'positional_encodings': full_checkpoint['positional_encodings'],
        'stack': full_checkpoint['stack'],
        'output': full_checkpoint['output'],
        'config': full_checkpoint['config']
    }

    print(f"Saving model-only checkpoint to {output_path}...")

    with open(output_path, "wb") as f:
        pickle.dump(model_only, f)

    # Show file sizes
    import os
    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\nDone!")
    print(f"Input size:  {input_size:.1f} MB")
    print(f"Output size: {output_size:.1f} MB")
    print(f"Reduction:   {input_size - output_size:.1f} MB ({(1 - output_size/input_size)*100:.1f}% smaller)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <input.pkl> <output.pkl>")
        print("Example: python convert_checkpoint.py artifacts/training_logs/checkpoint.pkl artifacts/models/model.pkl")
        sys.exit(1)

    convert_checkpoint(sys.argv[1], sys.argv[2])
