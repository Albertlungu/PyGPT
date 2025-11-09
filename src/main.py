import numpy as np
import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from embeddings.embeddings import EmbeddingLayer
from transformer.feed_forward import FeedForward
from transformer.single_head_attention import Attention
from tokenizer.tokenizer_class import BPETokenizer
from transformer.transformer_block import TransformerBlock
from transformer.output_layer import OutputLayer

def main():
    # Load tokenizer
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Example texts
    sample_texts = [
        "What is your name?"
    ]

    # Convert texts to token ids
    token_ids = [tokenizer.encode(text) for text in sample_texts]

    batch_size = len(sample_texts)

    # Create embedding layer
    embedding_layer = EmbeddingLayer(
        vocab_size=tokenizer.vocab_size, 
        embedding_dim=256
    )
    embeddings = embedding_layer.fwd(token_ids)

    # Transformer blocks
    transformer_block = TransformerBlock(token_ids, embedding_layer)
    transformer_output = transformer_block.fwd()

    output_layer = OutputLayer(embedding_layer)
    logits = output_layer.fwd(transformer_output)

    next_token = output_layer.predict_next_token(transformer_output, temperature=0.5)
    print(tokenizer.decode([int(next_token[0])]))

if __name__ == "__main__":
    main()