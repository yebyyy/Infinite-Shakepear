# Infinite Shakespear
This is a decoder only transformer, similar to a GPT structure that generates Shakespear-like text.

The Theoretical part of the model follows the decoder structure explained in the famous [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper.

Other than the basic **Multihead Self Attention** and **Feedforward** Structure, also implemented **LayerNorm**, **Residual Connections** from [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385), and **Dropout** from [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

`lecture.ipynb` follows the instruction by Andrej Karpathy's lecture [Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=471s)

`Final.py` is the final Decoder transformer model with 10 million parameters and `input.txt` as all the shakespear work input.
