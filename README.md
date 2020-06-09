# Character level Language modelling

### principle goal of the experiments in this repo:

Train models to predict token embeddings from character encoded text.<br>
If we can **find meaningful chunks of characters** (words or tokens) 
and if from those characters we can **predict embeddings** for a pretrained language model, 
then we should be able to use those embeddings in the same ways they are used in any language task.

This work is motivated by many papers which find small amounts of success using large amounts of compute when trying to get character level language models to work well on anything other than next character prediction. I hope to leverage token-based pretrained models to reach state-of-the-art performance using characters as input. From there I hope to exceed token-level performance, by extending the effective vocabulary size, making models more robust to misspellings and uncommon words & names, and lastly increase performance of models for languages in which tokenization is much more difficult than in English.

## Overview
There are 3 things we'll need to do to predict a sequence of word embeddings from a sequence of characters.

1. Find the blocks of characters that make up words. In this case a block is a fixed length and only valid its first character is the first character of a word or word piece from the model vocab.
2. Isolate those blocks from the adjacent blocks that do not contain valid words.
3. predict a word embeddings for each character block.

For the experiments in this repo I'll be using the [huggingface distilbert-base-uncased](https://huggingface.co/transformers/model_doc/distilbert.html) model and lowercase vocab/tokenizer. I'll be referring to 'word pieces' mostly just as 'words'

## 1. Predicting the first character of a word
This module has the job of predicting whether or not a character is the first character of a word.
We'll train this module on character encoded text with the objective of predicting whether or not the bert tokenizer would have split the sequence at each character. The [token_start_pred](token_start_pred) directory contains the model and experiment code for training this module.
<p align="center">
  <img src="images/char_lm_word_attn_rotated.png">
</p>

## 2. Isolate the character blocks that start with words
This module takes in the attention vector from the previous module in order to construct a switchboard which will pass only valid word blocks to the next module. This mechanism works like the mechanisms in a [Neural Teuring Machine](https://arxiv.org/abs/1410.5401) for selecting memory addresses or the SortCut application of the [Sparse Sinkhorn Attention module](https://arxiv.org/abs/2002.11296): it uses matrix multiplication to select certain which rows from the input matrix to output as unchanged rows in a smaller matrix. Differentiable/Rule based state machines, LSTMs, and convolutional models all perform similarly average on this task. Experiments for this module on its own have not been included yet, but the switchboards can be inspected in the full model in [char_seq_to_emb_seq](char_seq_to_emb_seq).

<p align="center">
  <img src="images/char_lm_switchboard_rotated.png">
</p>

## 3. Predicting words from characters
A word block must start with the first character of a word or word piece in order for it to be used by this module.<br>
Predicting a word's embedding from characters is fairly straightforward, but this module needs to be a bit more robust since a word block may contain more than just the first word of interest. This can be done with a fairly small feed-forward network as shown in the [single_emb_pred](single_emb_pred) directory. At first glance this model may not appear to be a simple FFNN becuase it is set up to operate on unfolded character blocks, similar to how convolution operations are computed. However in these experiments only a single character block is used at a time. This is done so that the feed-forward network can efficiently be applied to a sequence of word blocks to calculate an embedding for each one.<br>

<p align="center">
  <img src="images/char_lm_emb_pred_single.png">
</p>

## Putting it all together: so far so meh.
<p align="center">
  <img src="images/char_lm_full_labeled.png">
</p>

### Isn't creating neural networks to predict embeddings always going to be slower and less accurate than an embedding lookup?
For sure. The first pass of these experiments is just to match the performance of token-embedding-lookup based models. Once that is achieved we can see about making the models more robust to misspellings, analyze their performance on unseen token sequences, and generally see what other kinds of additional information can be passed from characters to the rest of the model.
<br>

### problems with the word start prediction & switchboard attention for word block selection:
<p align="center">
  <img src="images/leaky_switchboard.png">
</p>

There are 2 main problems with the performance of this model. Without any hand tuning of the word start attention activation function, values significantly different than 0 or 1 may be passed to the switchboard. This 'uncertainty' in the attention causes the switchboard to select multiple inputs for a single output and weigh them according to the attention values. This breaks the predictions at the point of the uncertainty and all future predictions.  As can be seen in the image above the attention starts of sharp, but leaks and becomes softer and spread out. 
Similarly, In the case where we have a target sequence of embeddings we want to predict, any additional or missed word start prediction causes a mismatch in the rest of the embeddings in the output and no useful signal can be obtained for the end of that sequence. I have set up experiments in which only the masked embeddings are to be predicted, but this, even for token-based inputs, leads to a reduction in performance. Summing the reductions in performance at each step in this process ends up yielding models that do not achieve the desired goal.

###
