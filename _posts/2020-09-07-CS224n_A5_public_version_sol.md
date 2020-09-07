---
layout:     post
title:      CS224n_A5_public_version_sol
subtitle:   CS224n_作业5_2020版答案
date:       2020-09-07
author:     SC
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - NLP
---


## Written部分答案
2020cs224n作业5比2019版本改变挺多，所以在这里记录一下自己的答案（still in process，仅供参考）。

Problem 1.
(a) We learned in class that recurrent neural architectures can operate over variable length input (i.e., the shape of the model parameters is independent of the length of the input sentence). Is the same true of convolutional architectures? Write one sentence to explain why or why not.

Solution: This is true for conv nets as well because we could apply embeddings and use linear transformation to resize input shapes.


(b) In 1D convolutions, we do padding, i.e. we add some zeros to both sides of our input, so that the kernel sliding over the input can be applied to at least one complete window.
In this case, if we use the kernel size k = 5, what will be the size of the padding (i.e. the additional number of zeros on each side) we need for the 1-dimensional convolution, such that there exists at least one window for all possible values of mword in our dataset? Explain your reasoning.

Solution: Because the kernel size is 5, the length of $x'_{padded}$ has to be at least 5 to obtain a complete window sliding. For example, if the input is just a letter "a", then we need to pad 1 zero on each side of this input, plus the "start/end" tokens padded to the beginning and the end of the word, we have "start"+"0"+"a"+"0"+"end", a size 5 window. 


(c ) In step 4, we introduce a Highway Network with $x_{highway} = x_{gate}⊙x_{proj} + (1 − x_{gate})⊙x_{conv\_out}$. Use one or two sentences to explain why this behavior is useful in character embeddings.
Based on the definition of $x_{gate} = σ(W_{gate}x_{conv\_out} + b_{gate})$, do you think it is better to initialize $b_{gate}$ to be negative or positive? Explain your reason briefly.

Sol: This makes the model more complicate and able to capture more information, as $x_{proj}$ represents the linear projection of the word as whole, while $x_{conv\_out}$ is just simply the output from the character model.
It'd better be initialized as positive to increase the weight of $x_{proj}$, so that we could incorporate more word-level information.

(d) In Lecture 10, we briefly introduced Transformers, a non-recurrent sequence (or sequence-to-sequence) model with a sequence of attention-based transformer blocks. Describe 2 advantages of a Transformer encoder over the LSTM-with-attention encoder in our NMT model (which we used in both Assignment 4 and Assignment 5)

Sol: First, as it's non-recurrent (Transformers process all the words simultaneously while LSTM does this one by one), parallel computing utilizing gpu is possible. This could be extremely helpful to efficient computation. Second, the attention mechanism sees all words at once and does not really biased towards closer or more distant words, thus it generates a more natural long-term relationship among words.
