import numpy as np
import xml.etree.ElementTree as ET

# Debug: Check if file exists and can be parsed
xml_path = '/Users/albertlungu/Desktop/Personal VSCode/Python Chatbot/tokenizer_training_data/enwiki-latest-pages-articles-multistream1.xml-p1p41242'
print(f"Attempting to parse XML file: {xml_path}")

try:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print("XML file parsed successfully.")
except Exception as e:
    print(f"Error parsing XML file: {e}")


# class BPETokenizer:
#     def __init__(self, input=1000):
#         """
#         Initializes a BPETokenizer object.

#         Sets a fixed vocab size to 1000, similar to tokenizer_ex.py.
#         If input is None, vocab size is set to 1000.
#         """
#         self.vocab_size = int(np.sqrt(len(input)))
#         self.base_vocab_size = 256
#         self.merges = {}

#     def get_stats(self, input):
#         """
#         Given a text, returns a dictionary of pair counts.
#         The key is a tuple of two adjacent characters, and the value is the count of that pair.
#         """
#         counts = {}
#         for pair in zip(input, input[1:]):
#             counts[pair] = counts.get(pair, 0) + 1
#         return counts

#     def merge(self, input, pair, idx):
#         """
#         Merge a pair of adjacent ids in a list of ids to a single idx.

#         Args:
#             input (list): The list of ids to merge.
#             pair (tuple): The pair of ids to merge.
#             idx (int): The id to replace the pair with.

#         Returns:
#             list: The list of ids with the pair merged.
#         """
#         new_input = []
#         i = 0
#         while i < len(input):
#             if i < len(input) - 1 and input[i] == pair[0] and input[i + 1] == pair[1]:
#                 new_input.append(idx)
#                 i += 2
#             else:
#                 new_input.append(input[i])
#                 i += 1
#         return new_input
    
#     def make_merges(self, input):
#         num_merges = self.vocab_size - self.base_vocab_size
#         merges = {}
#         input = list(input)
#         base_vocab_start = self.base_vocab_size
#         for i in range(num_merges):
#             stats = self.get_stats(input)
#             if not stats:
#                 break
#             pair = max(stats, key=stats.get)
#             idx = base_vocab_start + i
#             print(f"merging {pair} into a new token: {idx}")  
#             input = self.merge(input, pair, idx)
#             merges[pair] = idx
#         self.merges = merges
#         return input

# if __name__ == "__main__":
#     # input = open("tokenizer_training_data/text1.txt", "r").read()
#     # print(input)
#     tokens = input.encode("utf-8")
#     # for text in input:
#     #     tokens.append(text.encode("utf-8"))

#     tokenizer = BPETokenizer(tokens)
#     print(tokens)

#     ids = tokenizer.make_merges(tokens) 

#     print("Original tokens length:", len(tokens))
#     print("Final ids length:", len(ids))
#     print(f"Compression ratio: {len(tokens) / len(ids):.2f}X")
#     # pass