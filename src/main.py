import numpy as np
import pickle
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from embeddings.embeddings import EmbeddingLayer
from transformer.feed_forward import FeedForward
from transformer.single_head_attention import Attention
from tokenizer.tokenizer_class import BPETokenizer
from transformer.transformer_block import TransformerBlock
from transformer.output_layer import OutputLayer
from training.train import Trainer

def main():

    print("This code is running.")
    # Load tokenizer
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    # Example texts
    # training_texts = [
    #     "What is your name?",
    #     "Hello, how are you?",
    #     "This is a simple example text."

    # ]

    max_lines = 100
    training_texts = []

    with open("tokenizer_training_data/all_wiki_text.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=max_lines, desc = "Loading texts...")):
            if i >= max_lines:
                break
            training_texts.append(line.strip())

    print("="*60) 
    print("Appended training texts to list")
    print("="*60)

    trainer = Trainer(
        tokenizer=tokenizer,
        user_input=training_texts,
        lr = 1e-4
    )

    print("="*60)
    print("Training model.")
    trainer.train(epochs=300)
    print("Finished training model.")
    print("="*60)

    print("Saving checkpoints.")
    trainer.save_checkpoint("artifacts/training_logs.pkl")
    print("Saved checkpoints.")
    print("="*60)

    # Convert texts to token ids
    token_ids = [tokenizer.encode(text) for text in training_texts]

    batch_size = len(training_texts)

    # Create embedding layer
    embedding_layer = EmbeddingLayer(
        vocab_size=tokenizer.vocab_size, 
        embedding_dim=256
    )
    embeddings = embedding_layer.fwd(token_ids)

    prompt = "Once upon a time: "
    generated_text = trainer.generate(prompt, max_length = 50)
    print("Generated: \n", generated_text)

if __name__ == "__main__":
    main()