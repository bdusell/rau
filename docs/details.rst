Details
=======

Command-Line Interface
----------------------

The main command-line interface to Rau is the command ``rau``, which is
automatically installed as part of the ``rau`` package. It has sub-commands for
two tasks: language modeling (``lm``) and sequence-to-sequence transduction
(``ss``). Each task has sub-commands that correspond to three pipeline stages:

1. take pre-tokenized plaintext data and prepare it in a way that makes it more
   efficient to load later
2. take prepared data and train a new model on it from scratch
3. use a trained model to process some prepared data

For language modeling, the three sub-commands are

* ``rau lm prepare``
* ``rau lm train``
* ``rau lm evaluate`` (compute cross-entropy and perplexity)

For sequence-to-sequence transduction, the three sub-commands are

* ``rau ss prepare``
* ``rau ss train``
* ``rau ss translate`` (translate input sequences to output sequences)

For details on how to use these commands, run them with ``-h`` to see their help
messages.

Features and Limitations
------------------------

This section lists some of Rau's best features and known limitations. This
side-by-side comparison of Rau's pros and cons may help you decide if Rau is a
good fit for your needs.

Features
^^^^^^^^

#. Provides a flexible Python API for building neural network architectures by
   composing simpler ones. In particular, it provides an abstract base class
   called ``Unidirectional`` that represents a unidirectional sequential neural
   network, which makes it effortless to modify or compose sequential neural
   network architectures. The ``Unidirectional`` class supports both
   timestep-parallel training and autoregressive decoding. If you have two
   ``Unidirectional`` models that support both of these modes, you can compose
   them into a model that feeds the outputs of the first model as inputs to the
   second, and the composite model will also support both modes efficiently, for
   free. See :doc:`composable-neural-networks`.
#. The RNN and LSTM use learned initial hidden states.
#. None of the architectures have upper limits on sequence length. This includes
   the transformer, which uses sinusoidal positional encodings that can be
   extended arbitrarily. You can train on short sequences and evaluate on
   arbitrarily long sequences.
#. PyTorch uses two bias terms in the recurrent layers of the RNN and LSTM.
   However, only one is required, and the second one is redundant. Including the
   second term only serves to effectively double the learning rate of the bias
   term at the cost of adding additional parameters to the model. This means
   that RNNs and LSTMs can have speciously high parameter counts, which is
   undesirable if you are trying to compare different models with comparable
   parameter counts. Rau takes care to remove these redundant bias parameters,
   resulting in better parameter counts.
#. Supports minibatching with padding. For the sake of efficiency, Rau groups
   sequences of similar length together to reduce the number of padding tokens,
   and it enforces upper limits on the number of tokens in a minibatch.
#. Padding is handled correctly, in the sense that there is mathematically no
   difference between processing :math:`N` sequences in a single minibatch with
   padding and processing the same `N` sequences individually while accumulating
   their gradients. Minibatching is simply an implementation detail that
   increases speed.
#. Padding tokens do not take up space in the vocabulary or in the embedding
   matrix of the model. That is, there is no integer ID in the vocabulary that
   is devoted to padding. Instead, Rau dynamically figures out integer IDs to
   use for padding that don't conflict with other tokens. They are an
   implementation detail that is entirely hidden from the user. Language models
   and decoders never assign probability to padding tokens and are unaware that
   padding tokens exist.
#. Everything is efficiently vectorized and supports both CPU and GPU modes.
#. Rau is very fast for small model sizes and small dataset sizes, even on CPU.
   An example of a "small" language modeling experiment would be a model with
   about 128k parameters and a dataset of about 100k sequences up to length 40.
   Rau can train hundreds of small models to convergence in under 20 minutes on
   a scientific computing cluster using only CPU nodes—no GPUs! This is very
   useful for researchers who train neural networks on small, synthetic
   experiments.
#. It is not tied to a particular tokenization algorithm, because it does not
   implement tokenization at all. It is compatible with datasets preprocessed by
   external tokenization tools, such as SentencePiece.
#. The dataset format is deliberately simple: plaintext consisting of one
   sequence per line, where each sequence consists of whitespace-separated
   tokens.
#. Offers different ways of handling UNK tokens. You can declare a particular
   token, such as ``<unk>``, to represent a catch-all UNK token. Or, you can
   disable UNK tokens entirely and treat out-of-vocabulary tokens as errors.
#. Beam search is parallelized across beam elements (but not minibatch
   elements).
#. Beam search terminates as soon as EOS is the top beam element, rather than
   waiting for the beam to fill up with EOS. This is correct because the a beam
   element can never have a descendant with higher probability than itself. The
   latter approach is only required if the scores can increase, e.g., when using
   certain kinds of length normalization.

Limitations
^^^^^^^^^^^

#. The only tasks implemented are language modeling and sequence-to-sequence
   generation. Generation from language models has not been implemented,
   although it might be in the future.
#. The only architectures available for language modeling are the simple RNN,
   LSTM, and transformer.
#. The only architecture available for sequence-to-sequence generation is the
   transformer.
#. The only algorithm currently implemented for generating outputs is beam
   search. In the future, other generation algorithms such as ancestral
   sampling, greedy decoding, and constrained ancestral sampling may be added.
#. Beam search is not parallelized across minibatch elements.
#. Due to limitations in the API for PyTorch's transformer implementation,
   decoding for transformers is very inefficient. At every step of decoding, all
   of the hidden representations are re-computed from scratch, and the model
   generates outputs for all previous timesteps, even though only the most
   recent one is needed. It does not implement what is commonly called "KV
   caching." The only things that are cached are the input embeddings. This
   might be fixed in the future.
#. It does not include tokenization and detokenization in the pipeline. You need
   to handle tokenization and detokenization yourself.
#. It slurps the entire training set into memory during training, so it will run
   out of memory on large datasets (~1m sequences). This might be fixed in the
   future.
#. Training cannot be stopped and restarted, so it cannot recover from crashes.
   This feature might be added in the future.

Technical Details
-----------------

This section is for people who want to understand the low-level details of Rau,
including details of the neural network architectures, training algorithm, and
decoding algorithms. This may be useful for researchers who need to be mindful
of these details and describe them in their papers, or for people who are just
deciding if Rau is up to snuff.

* All language models and decoders operate exclusively on whole sequences
  ending in EOS, without truncation, and without assigning any probability to
  tokens that cannot be generated, namely padding and BOS. This means that,
  mathematically, Rau's language models always define tight language models,
  i.e., probability distributions over the set of all strings of tokens.
  Training examples are never truncated, split across multiple minibatches, or
  shifted to different positions. This is in contrast to other setups that
  treat the training data as one long sequence and split it into chunks of
  fixed size.
* The RNN and LSTM use learned initial hidden states.
* During training, checkpoints are taken every :math:`N` examples, where
  :math:`N` can be configured by ``--examples-per-checkpoint``. At each
  checkpoint, the model is evaluated on the validation set. The model's
  performance on the validation set controls the learning rate schedule and
  early stopping.
* When training ends, the parameters of the best checkpoint have been saved to
  disk.
* Parameters can be optimized using either simple gradient descent or Adam.
  This can be configured with ``--optimizer``.
* An initial learning rate can be set with ``--initial-learning-rate``. The
  learning rate is reduced every time the validation performance does not
  improve after a certain number of epochs, which can be configured with
  ``--learning-rate-patience``. It is reduced by multiplying it by a number in
  :math:`(0, 1)`, which can be configured with
  ``--learning-rate-decay-factor``.
* Training stops early when the validation performance does not improve after a
  certain number of epochs, which can be configured with
  ``--early-stopping-patience``.
