<style>
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 16px;
}
</style>


This file is just to help me understand positional encoding and to remind myself if I forget. 

*[See this website for more details](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)*

**What is positional encoding?**
- Positional encoding is a way of representing the position of an object in the input sequence in a way that the model can understand and use both its meaning and its context.
    - This is necessary because the order of words in a sentence matters.
- For future reference, when I refer to dimension, it means the length of a vector

The formulas for positional encoding at a certain position are the following:

$$P(k, 2i) = \text{sin}(\frac{k}{n^{2i/d}})$$

<p style="text-align:center; font-weight: bold">Or, alternatively, for odd indices: </p>

$$P(k, 2i + 1) = \text{cos}(\frac{k}{n^{2i/d}})$$

Where:

𝑘: Position of an object in the input sequence, 0 ≤ 𝑘 <𝐿/2
- 𝐿 is the length of the input sequence
    - For example, in the input sequence “hello” (5 characters), 𝐿 = 5, and if 𝑘 = 0, the targeted character is "h"

𝑑: Dimension of the output embedding space
- i.e. the size of the embedding vector
    - The size of an embedding vector refers to a vector as such: *$$<0.1, 0.513, -0.008, ..., -0.4>$$* with 𝑑 being the amount of numbers in that vector (often 128, 256, 512, 768, etc.)

𝑃⁡(𝑘,𝑗): Position function for mapping a position 𝑘 in the input sequence to index (𝑘,𝑗) of the positional matrix

𝑛: User-defined scalar, set to 10,000 by the authors of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762).
- Basically, it controls the frequency and wavelength of the sine and cosine functions representing the positional encoding
    - Frequencies and wavelengths help to represent the position of the object in the input sequence and its context

𝑖: Used for mapping to column indices 0 ≤ 𝑖 <𝑑/2, with a single value of 𝑖 maps to both sine and cosine functions
- This last one is in between 0 and d/2 because the first half of the output is the sin values and the second half is the cos values