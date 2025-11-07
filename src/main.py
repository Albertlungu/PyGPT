import numpy as np
import pickle
from embeddings.embeddings import EmbeddingLayer
from transformer.feed_forward import FeedForward
from transformer.attention import Attention
from tokenizer.tokenizer_class import BPETokenizer

def main():
    # Load tokenizer
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Example texts
    sample_texts = [
        "Hello World. My name is Albert Lungu",
        "What is your name?",
        "I like LLMs"
    ]

    # Convert texts to token ids
    token_ids_list = [tokenizer.encode(text) for text in sample_texts]

    # Determine max sequence length for padding
    max_len = max(len(ids) for ids in token_ids_list)

    # Pad token sequences to same length
    padded_token_ids = np.array([
        ids + [0]*(max_len - len(ids))  # assuming 0 is the padding id
        for ids in token_ids_list
    ])

    batch_size = len(sample_texts)

    # Create embedding layer
    embedding_layer = EmbeddingLayer(
        vocab_size=tokenizer.vocab_size, 
        embedding_dim=256
    )
    # Instantiate Attention
    attention = Attention(padded_token_ids, embedding_layer)

    # Run forward pass
    output = attention.fwd()

    print("Attention output shape:", np.shape(output))
    print("="*60)
    print("Attention forward pass ran successfully")
    print("="*60)

if __name__ == "__main__":
    main()