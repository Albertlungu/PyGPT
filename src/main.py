import numpy as np
import sys, os
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

    max_lines = 1000
    dataset = load_dataset("tatsu-lab/alpaca")

    train_data = dataset["train"]
    train_data = train_data.select(range(max_lines))

    # with open("tokenizer_training_data/alpaca_sample_utf8.txt", "w", encoding="utf-8") as f:
    #     for ex in train_data:
    #         text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {ex['output']}\n"
    #         f.write(text + "\n")


    training_texts = []
    with open("tokenizer_training_data/alpaca_sample_utf8.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            training_texts.append(line.strip())

    training_data = train_data.select(range(max_lines))

    # with open("tokenizer_training_data/all_wiki_text.txt", "r", encoding="utf-8") as f:
    #     for i, line in enumerate(tqdm(f, total=max_lines, desc = "Loading texts...")):
    #         if i >= max_lines:
    #             break
    #         training_texts.append(line.strip())

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
    train_time = time.time()
    trainer.train(epochs=13, batch_size=50)
    end_train = time.time() - train_time
    print("Finished training model.")
    print("="*60)

    print("Saving checkpoints.")
    check_time = time.time()
    trainer.save_checkpoint("artifacts/training_logs/training_logs_1000l_10_11_2025_10:27pm.pkl")
    end_check = time.time() - check_time
    print("Saved checkpoints.")
    print("="*60)


    print(f"Train time: {end_train:.4f}\nCheckpoint time: {end_check:.4f}")

    # Convert texts to token ids
    token_ids = [tokenizer.encode(text) for text in training_texts]

    batch_size = len(training_texts)

    # Create embedding layer
    embedding_layer = EmbeddingLayer(
        vocab_size=tokenizer.vocab_size, 
        embedding_dim=256
    )
    embeddings = embedding_layer.fwd(token_ids)

    prompt = "What is 5+5?"
    generated_text = trainer.generate(prompt, max_length = 50)
    print("Generated: \n", generated_text)

def main():
    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    
    dummy_input = ["dummy"]
    trainer = Trainer(tokenizer, dummy_input)

    trainer.load_checkpoint("artifacts/training_logs/training_logs_1000l_10_11_2025_10:27pm.pkl")

    prompt = "Describe some of the benefits of a vegetarian diet."
    generated_text = trainer.generate(prompt, max_length=10, temperature=0.7, top_k=40, repetition_penalty= 5)
    print("Length generated: \n", generated_text)

if __name__ == "__main__":
    start = time.time()
    # train()
    end = time.time()
    print("="*60)
    print(f"Execution time: {end-start:.4f} s")

    main()








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