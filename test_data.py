import pickle
from src.tokenizer.tokenizer_class import BPETokenizer

with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
    tokenizer = pickle.load(f)
    tokenizer._ensure_vocab()

with open("training_data/alpaca_tokenized.pkl", "rb") as f:
    alpaca=pickle.load(f)

print(tokenizer.decode(alpaca[-1]))