import tiktoken
import pickle
from tqdm import tqdm

class TikToken:
    def __init__(self):
        self.enc = tiktoken.get_encoding("r50k_base")
        self.vocab_size = self.enc.max_token_value + 1  # +1 because max_token_value is 0-indexed
        self.eos_token_id = self.enc.eot_token
        # Use 0 as padding token ID (common convention)
        self.padding_token_id = 0

    def encode(self, text):
        """
        Encode text to token IDs.

        Args:
            text (str): Text to encode

        Returns:
            list: List of token IDs
        """
        return self.enc.encode(text)

    def decode(self, token_ids):
        """
        Decode token IDs to text.

        Args:
            token_ids (list): List of token IDs

        Returns:
            str: Decoded text
        """
        return self.enc.decode(token_ids)
    
def tokenize_data(input_path, output_path):
    with open(input_path, "r") as f:
        data = f.read()
    
    training_texts = [doc.strip() for doc in data.split('\n\n') if doc.strip()]

    tokenizer = TikToken()
    token_ids = []

    for i in tqdm(training_texts):
        ids = tokenizer.encode(i)
        ids.append(tokenizer.eos_token_id)
        token_ids.append(ids)
    
    with open(output_path, "wb") as f:
        pickle.dump(token_ids, f)


def main():
    tokenizer = TikToken()

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"Padding token ID: {tokenizer.padding_token_id}")

    # Test encoding and decoding
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

def test_tokenized(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    tokenizer = TikToken()

    print(tokenizer.decode(data[-1]))
    


if __name__ == "__main__":
    tokenize_data("training_data/alpaca.txt", "training_data/tiktoken_alpaca.pkl")
    main()
    test_tokenized("training_data/tiktoken_alpaca.pkl")
    pass