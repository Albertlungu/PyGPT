import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from user_input import UserInput
# import xml.etree.ElementTree as ET
# import re
# from tqdm import tqdm
import pickle
from tokenizers.tokenizer_class import BPETokenizer

"""To remember for future: distinction between tokens and ids:

    tokens => text encoded into utf-8 bytes. For example, "Hello World" would return <<b'Hello World'>> (tokens)

    ids ==> tokens turned into numbers (the things that I merged in tokenizer_class.py), where each char has a specific id attributed to it. ids = tokenizer.encode_to_ids(input_message) for input_message = 'hello world' will return [104, 542, 298, 620, 108, 100] (list of ids)
"""


if __name__ == '__main__':

    # input_message = UserInput().user_input

    with open('artifacts/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()
    
    message = UserInput().user_input
    print("original message: ", message)
    print("encoded message: ", tokenizer.encode_to_ids(message))
    for i in tokenizer.encode_to_ids(message):
        print(tokenizer.vocab[i])