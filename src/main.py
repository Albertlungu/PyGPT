import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import xml.etree.ElementTree as ET
# import re
from tqdm import tqdm
import pickle
from embeddings.embeddings import EmbeddingLayer
from transformer.feed_forward import FeedForward
from transformer.attention import Attention
from transformer.transformer_block import TransformerBlock
from tokenizer.tokenizer_class import BPETokenizer

"""To remember for future: distinction between tokens and ids:

    tokens => text encoded into utf-8 bytes. For example, "Hello World" would return <<b'Hello World'>> (tokens)

    ids ==> tokens turned into numbers (the things that I merged in tokenizer_class.py), where each char has a specific id attributed to it. ids = tokenizer.encode_to_ids(input_message) for input_message = 'hello world' will return [104, 542, 298, 620, 108, 100] (list of ids)
"""

def main():
    """Hey I'm testin' here!"""

    with open("artifacts/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    sample_texts = [
    "Hello World. My name is Albert Lungu",
    "What is your name?",
    "I like LLMs",
]
    token_ids = [tokenizer.encode(i) for i in sample_texts]
    
    embedding_layer = EmbeddingLayer(vocab_size = tokenizer.vocab_size)
    ffn = FeedForward(embedding_layer, token_ids)
    
    print(np.shape(ffn.output))
    

    print("="*60)
    print("Code Ran Succesfully")
    print("="*60)


if __name__ == '__main__':
    main()