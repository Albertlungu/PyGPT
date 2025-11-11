<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue">
  <img src="https://img.shields.io/badge/NumPy-1.26-orange">
  <img src="https://img.shields.io/badge/License-GPLv3-blue">
  <img src="https://img.shields.io/badge/AI-LLM-lightblue">
</p>


# PyGPT — A Learning LLM Project

## Overview
It's a GPT-like LLM that uses a relatively small wikipedia dump as a source, so you already can deduce that it probably isn't very smart. 
This is used as more of a learning experience for myself to understand the concepts of what AI is, and how an LLM is made. 

## Libraries used
- numpy
- pickle
- sys
- os
- matplotlib

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Tokenization process

To get tokenization training data as a `.txt` file, follow these steps:

1. Visit the [Wikipedia English Language Dump](https://dumps.wikimedia.org/enwiki/)
2. Click on [Latest](https://dumps.wikimedia.org/enwiki/latest/)
3. Use **Cmd+F** to find the file at the following date and time: **292240103**
4. Click on `"enwiki-latest-pages-articles-multistream1.xml-p..>"` to download the file
5. Once in finder system, the file is in a `.bz2` format. Double click the file to expand it into a `.xml` file on MacOS (Windows might be different idrk).
6. Add this to the folder **'tokenizer_training_data'**
7. Create a file called **'all_wiki_text.txt'** inside that same folder
8. Open `tokenizer_class.py`, comment all existing code inside the main loop and run the `extract_wiki_text` function there.
9. Once text file has been fully written, comment the `extract_wiki_text` function and uncomment the main loop.

### Tokenizer Details
- Implements **Byte Pair Encoding (BPE)** algorithm to compress all words into subword tokens.
- Starts with a base vocab size of 256
- Iteratively merges the most frequent adjacent byte pairs (letter or character pairs) until max vocab size is reached

To learn more about BPE, I highly recommend [this video by Andrej Karpathy](https://www.youtube.com/watch?v=zduSFxRajkE) 
- **MASSIVE** thanks to him for his amazing instructional videos.

#### Example Usage:
```python
with open("artifacts/tokenizer.pkl", "rb") as f:
  tokenizer = pickle.load(f)
  tokenizer._ensure_vocab()

text = "hello world"
token_ids = tokenizer.encode(text)
print(token_ids)  # e.g., [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

decoded_text = tokenizer.decode(token_ids)
print(decoded_text)  # "hello world"
```

#### Example of how it works (from [Wikipedia.org](https://en.wikipedia.org/wiki/Byte-pair_encoding#:~:text=The%20original%20BPE%20algorithm%20operates,the%20target%20text%20effectively%20compressed)):

Suppose the data to be encoded is:
```
aaabdaaabac
```
  The byte pair "aa" occurs most often, so it will be replaced by a byte that is not used in the data, such as "Z". Now there is the following data and replacement table:

```
ZabdZabac
Z=aa
```
  Then the process is repeated with byte pair "ab", replacing it with "Y":

```
ZYdZYac
Y=ab
Z=aa
```
  The only literal byte pair left occurs only once, and the encoding might stop here. Alternatively, the process could continue with recursive byte-pair encoding, replacing "ZY" with "X":

```
XdXac
X=ZY
Y=ab
Z=aa
```
  This data cannot be compressed further by byte-pair encoding because there are no pairs of bytes that occur more than once.

To decompress the data, simply perform the replacements in the reverse order.

**Source**: [Wikipedia](https://en.wikipedia.org/wiki/Byte-pair_encoding#:~:text=The%20original%20BPE%20algorithm%20operates,the%20target%20text%20effectively%20compressed)



## How PyGPT Works

```mermaid
flowchart TD
    A[Input Text] --> B[Tokenizer: tokenizer.pkl]
    B --> C[Embeddings + Positional Encodings]
    C --> D[Transformer Block: single head attention + feed forward]
    D --> E[Output Layer]
    E --> F[Predicted Tokens → Decoded Text]

    %% Training loop connections (dashed lines)
    B -.-> G[train.py: prepare batches]
    E -.-> H[loss_function.py: compute loss]
    H -.-> I[Backpropagation: update parameters in embeddings, transformer block, and output layer]

    %% Node colors
    style A fill:#C19A00,stroke:#8B7500,stroke-width:2px
    style B fill:#C19A00,stroke:#8B7500,stroke-width:2px
    style C fill:#1E3A8A,stroke:#1E40AF,stroke-width:2px
    style D fill:#15803D,stroke:#166534,stroke-width:2px
    style E fill:#B91C1C,stroke:#991B1B,stroke-width:2px
    style F fill:#C19A00,stroke:#8B7500,stroke-width:2px    
    style G fill:#6B21A8,stroke:#4C1D95,stroke-width:1px,stroke-dasharray: 5 5
    style H fill:#6B21A8,stroke:#4C1D95,stroke-width:1px,stroke-dasharray: 5 5
    style I fill:#6B21A8,stroke:#4C1D95,stroke-width:1px,stroke-dasharray: 5 5
```



## Project status
- [x] Tokenizer
- [x] Embedding Layer + Positional Encodings
- [x] Feed Forward Layer
- [x] Attention module (single head)
- [x] Transformer Block
- [x] Output Layer
- [x] Loss Function
- [x] Training
- [ ] Use JAX to use GPU for faster processing times - create different branch in git
- [ ] Multi-head attention
- [ ] Stack transformer blocks
- [ ] **MAYBE**: Change tokenizer from BPE to WordPiece or SentencePiece
- [ ] Optimization (optional)
- [ ] Add readme files for each part of the model for understanding
- [ ] React + Next.js + Tailwind CSS Frontend


O(n) = 1.1 × 10⁷ × max_lines# trigger cache refresh
