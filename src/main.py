import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.utils.config
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
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

    max_lines = 10
    training_texts =[]

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
    trainer.train(epochs=10)
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
    start = time.time()
    main()
    end = time.time()
    print("="*60)
    print(f"Execution time: {end-start:.4f} s")








# # Wrap your main logic in a function with max_lines parameter
# def run_training(max_lines):
#     # Load tokenizer
#     with open("artifacts/tokenizer.pkl", "rb") as f:
#         tokenizer = pickle.load(f)
#         tokenizer._ensure_vocab()

#     training_texts = []
#     with open("tokenizer_training_data/all_wiki_text.txt", "r", encoding="utf-8") as f:
#         for i, line in enumerate(tqdm(f, total=max_lines, desc=f"Loading {max_lines} lines...")):
#             if i >= max_lines:
#                 break
#             training_texts.append(line.strip())

#     trainer = Trainer(
#         tokenizer=tokenizer,
#         user_input=training_texts,
#         lr=1e-4
#     )

#     start_time = time.time()
#     trainer.train(epochs=1)
#     end_time = time.time()

#     return end_time - start_time


# # Run loop for 10 increments, starting with max_lines = 2
# results = []

# for i in range(30):
#     max_lines = 2 + i * 2
#     print("="*60)
#     print(f"Running training with max_lines = {max_lines}")
#     exec_time = run_training(max_lines)
#     results.append((max_lines, exec_time))
#     print(f"Execution time: {exec_time:.4f} s")
#     print("="*60)

# # Print results in a table
# print("\nSummary of execution times:")
# print(f"{'Max Lines':>10} | {'Time (s)':>10}")
# print("-"*25)
# for max_lines, t in results:
#     print(f"{max_lines:>10} | {t:>10.4f}")

# max_lines_list, times_list = zip(*results)  # unzip into two lists

# plt.figure(figsize=(8, 5))
# plt.plot(max_lines_list, times_list, marker='o', linestyle='-', color='blue')
# plt.title("Execution Time vs Max Lines")
# plt.xlabel("Max Lines")
# plt.ylabel("Time (seconds)")
# plt.grid(True)
# plt.xticks(max_lines_list)
# plt.tight_layout()
# plt.show()