Rau
===

Rau (rhymes with "now") is a Python module and command-line tool that provides
PyTorch implementations of neural network-based language modeling and
sequence-to-sequence generation. It is primarily suited for academic
researchers. Out of the box, it provides implementations of the simple
recurrent neural network (RNN), long short-term memory (LSTM), and transformer
architectures. It includes extensible Python APIs and command-line tools for
data preprocessing, training, and evaluation. It is very easy to get started
with the command-line tools quickly if you provide your data as pretokenized
(space-separated) plaintext.

There are, of course, many excellent implementations of neural network training
out in the world for you to choose from, but I think Rau does have some neat
features and avoids some pitfalls that I have not seen handled well in other
codebases. To see if using Rau is a good idea for you, see
`Technical Details`_, `Features`_, and `Limitations`_.

Rau has been used in the following papers:

* Alexandra Butoi, Ghazal Khalighinejad, Anej Svete, Josef Valvoda, Ryan Cotterell, Brian DuSell. `Training Neural Networks as Recognizers of Formal Languages. <https://openreview.net/forum?id=aWLQTbfFgV>`_ ICLR 2025.
* Taiga Someya, Anej Svete, Brian DuSell, Timothy J. O'Donnell, Mario Giulianelli, Ryan Cotterell. Information Locality as an Inductive Bias for Neural Language Models. ACL 2025.

Rau is based on code that was originally used for
`adding stack data structures to LSTMs and transformers <https://github.com/bdusell/stack-attention>`_.
A version of Rau that includes implementations of stack-augmented neural
network architectures can be found on
`this branch <https://github.com/bdusell/rau/tree/differentiable-stacks>`_.

How to Use Rau
--------------

Rau can be used to train neural networks from scratch on pretokenized data,
save them to disk, and re-load them to process pretokenized data. For example,
it can be used to train a transformer language model and then calculate its
perplexity on a set of test data. Or, it can be used to train a transformer
encoder-decoder on sequence-to-sequence data and then translate a set of source
sequences to outputs using beam search.

For language modeling, data simply needs to be formatted with one example
sequence per line, where each example is a sequence of whitespace-separated
tokens. For sequence-to-sequence transduction, it is formatted the same way,
but with separate files for source and target sequences. Rau will automatically
take care of building a vocabulary mapping token strings to integer IDs, adding
BOS and EOS symbols as needed, grouping examples into batches, adding
padding, and undoing all of these transformations at inference time. The end
user only needs to deal with sequences of whitespace-separated tokens.

Quickstart
----------

Clone the repo::

    git clone git@github.com:bdusell/rau.git
    cd rau

Optional: To simplify software installation, set up the pre-made Docker
container defined in this repo. To do this, install
`Docker <https://www.docker.com/get-started>`_
and the
`NVIDIA Container Toolkit <https://www.docker.com/get-started>`_
(for GPU support), then run this script in order to start a shell inside of the
container::

    bash scripts/docker_shell.bash --build

(If you don't have an NVIDIA GPU, don't install the NVIDIA Container Toolkit,
and run the above command with the flag ``--cpu`` added.)

Install Python dependencies. This can be done by installing the package manager
`Poetry <https://python-poetry.org/docs/#installation>`_
(it's already installed in the Docker container) and running this script::

    bash scripts/install_python_packages.bash

Start a shell inside the Python virtual environment using Poetry::

    bash scripts/poetry_shell.bash

You are now ready to use Rau.
    
Below are some quick examples of setting up pipelines for language modeling and
sequence-to-sequence transduction. We will use the pretokenized datasets from
`McCoy et al., 2020 <https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00304/43542/Does-Syntax-Need-to-Grow-on-Trees-Sources-of>`_;
their simplicity and small size make them convenient for our purposes.

Language Modeling
^^^^^^^^^^^^^^^^^

For this example, we'll train a transformer language model on simple
declarative sentences in English (the data comes from the source side of the
question formation task of McCoy et al., 2020).

Download the dataset::

    mkdir language-modeling-example
    curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.train | sed 's/[a-z]\+\t.*//' > language-modeling-example/main.tok
    mkdir language-modeling-example/datasets
    mkdir language-modeling-example/datasets/validation
    curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.dev | sed 's/[a-z]\+\t.*//' > language-modeling-example/datasets/validation/main.tok
    mkdir language-modeling-example/datasets/test
    curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.test | sed 's/[a-z]\+\t.*//' > language-modeling-example/datasets/test/main.tok

Note the directory structure:

* ``language-modeling-example/``: a directory representing this dataset

  * ``main.tok``: the pretokenized training set
  * ``datasets/``: additional datasets that should be processed with the
    training set's vocabulary

    * ``validation/``

      * ``main.tok``: the pretokenized validation set

    * ``test/``

      * ``main.tok``: the pretokenized test set

Now, "prepare" the data by figuring out the vocabulary of the training data and
converting all tokens to integers::

    python src/rau/tasks/language_modeling/prepare_data.py \
      --training-data language-modeling-example \
      --more-data validation \
      --more-data test \
      --never-allow-unk

The flag ``--training-data`` refers to the directory containing our dataset.
The flag ``--more-data`` indicates the name of a directory under
``language-modeling-example/datasets`` to prepare using the vocabulary of the
training data. The flag ``--never-allow-unk`` indicates that the training data
does not contain a designated unknown (UNK) token, and out-of-vocabulary tokens
should be treated as errors at inference time.

Note the new files generated:

* ``language-modeling-example/``

  * ``main.prepared``: the prepared training set
  * ``main.vocab``: the vocabulary of the training data
  * ``datasets/``

    * ``validation/``

      * ``main.prepared``: the prepared validation set

    * ``test/``

      * ``main.prepared``: the prepared test set

Now, train a transformer language model::

    python src/rau/tasks/language_modeling/train.py \
      --training-data language-modeling-example \
      --architecture transformer \
      --num-layers 6 \
      --d-model 64 \
      --num-heads 8 \
      --feedforward-size 256 \
      --dropout 0.1 \
      --init-scale 0.1 \
      --max-epochs 10 \
      --max-tokens-per-batch 2048 \
      --optimizer Adam \
      --initial-learning-rate 0.01 \
      --gradient-clipping-threshold 5 \
      --early-stopping-patience 2 \
      --learning-rate-patience 1 \
      --learning-rate-decay-factor 0.5 \
      --examples-per-checkpoint 50000 \
      --output saved-language-model   

This saves a transformer language model to the directory
``saved-language-model``.

Finally, calculate the perplexity of this language model on the test set::

    python src/rau/tasks/language_modeling/evaluate.py \
      --load-model saved-language-model \
      --training-data language-modeling-example \
      --input test \
      --batching-max-tokens 2048

Sequence-to-Sequence
^^^^^^^^^^^^^^^^^^^^

For this example, we'll train a transformer encoder-decoder on the question
formation task of McCoy et al. (2020), which involves converting a declarative
sentence in English to question form.

Download the dataset::

    mkdir sequence-to-sequence-example
    curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.train > sequence-to-sequence-example/train.tsv
    cut -f 1 < sequence-to-sequence-example/train.tsv > sequence-to-sequence-example/source.tok
    cut -f 2 < sequence-to-sequence-example/train.tsv > sequence-to-sequence-example/target.tok
    mkdir sequence-to-sequence-example/datasets
    mkdir sequence-to-sequence-example/datasets/validation
    curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.dev > sequence-to-sequence-example/validation.tsv
    cut -f 1 < sequence-to-sequence-example/validation.tsv > sequence-to-sequence-example/datasets/validation/source.tok
    cut -f 2 < sequence-to-sequence-example/validation.tsv > sequence-to-sequence-example/datasets/validation/target.tok
    mkdir sequence-to-sequence-example/datasets/test
    curl -s https://raw.githubusercontent.com/tommccoy1/rnn-hierarchical-biases/master/data/question.test | head -100 > sequence-to-sequence-example/test.tsv
    cut -f 1 < sequence-to-sequence-example/test.tsv > sequence-to-sequence-example/datasets/test/source.tok
    cut -f 2 < sequence-to-sequence-example/test.tsv > sequence-to-sequence-example/datasets/test/target.tok
    rm sequence-to-sequence-example/{train,validation,test}.tsv

Note the directory structure:

* ``sequence-to-sequence-example/``: a directory representing this dataset

  * ``source.tok``: the source side of the pretokenized training set
  * ``target.tok``: the target side of the pretokenized training set
  * ``datasets/``: additional datasets that should be processed with the
    training set's vocabulary

    * ``validation/``

      * ``source.tok``: the source side of the pretokenized validation set
      * ``target.tok``: the target side of the pretokenized validation set

    * ``test/``

      * ``source.tok``: the source side of the pretokenized test set
      * ``target.tok``: the target side of the pretokenized test set

Now, "prepare" the data by figuring out the vocabulary of the training data and
converting all tokens to integers::

    python src/rau/tasks/sequence_to_sequence/prepare_data.py \
      --training-data sequence-to-sequence-example \
      --vocabulary-types shared \
      --more-data validation \
      --more-source-data test \
      --never-allow-unk

The flag ``--training-data`` refers to the directory containing our dataset.
The flag ``--vocabulary-types shared`` means that the script will generate a
single vocabulary that is shared by both the source and target sides. This
makes it possible to tie source and target embeddings. The flag ``--more-data``
indicates the name of a directory under
``sequence-to-sequence-example/datasets`` to prepare using the vocabulary of
the training data (both the source and target sides will be prepared). The flag
``--more-source-data`` does the same thing, but it only prepares the source
side (only the source side is necessary for generating translations on a test
set). The flag ``--never-allow-unk`` indicates that the training data does not
contain a designated unknown (UNK) token, and out-of-vocabulary tokens should
be treated as errors at inference time.

Note the new files generated:

* ``language-modeling-example/``

  * ``source.shared.prepared``
  * ``target.shared.prepared``
  * ``shared.vocab``: a shared vocabulary of tokens that appear in either the
    source or target side of the training set
  * ``datasets/``

    * ``validation/``

      * ``source.shared.prepared``
      * ``target.shared.prepared``

    * ``test/``

      * ``source.shared.prepared``
      * ``target.shared.prepared``

Now, train a transformer encoder-decoder model::

    python src/rau/tasks/sequence_to_sequence/train.py \
      --training-data sequence-to-sequence-example \
      --vocabulary-type shared \
      --num-encoder-layers 6 \
      --num-decoder-layers 6 \
      --d-model 64 \
      --num-heads 8 \
      --feedforward-size 256 \
      --dropout 0.1 \
      --init-scale 0.1 \
      --max-epochs 10 \
      --max-tokens-per-batch 2048 \
      --optimizer Adam \
      --initial-learning-rate 0.01 \
      --label-smoothing-factor 0.1 \
      --gradient-clipping-threshold 5 \
      --early-stopping-patience 2 \
      --learning-rate-patience 1 \
      --learning-rate-decay-factor 0.5 \
      --examples-per-checkpoint 50000 \
      --output saved-sequence-to-sequence-model

This saves a model to the directory ``saved-sequence-to-sequence-model``.

Finally, translate the source sequences in the test data using beam search::

    python src/rau/tasks/sequence_to_sequence/translate.py \
      --load-model saved-sequence-to-sequence-model \
      --input sequence-to-sequence-example/datasets/test/source.shared.prepared \
      --beam-size 4 \
      --max-target-length 50 \
      --batching-max-tokens 256 \
      --shared-vocabulary-file sequence-to-sequence-example/shared.vocab

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

Features
--------

#. Provides a flexible Python API for building neural network architectures by
   composing simpler ones. In particular, it provides an abstract base class
   called ``Unidirectional`` that represents a unidirectional sequential neural
   network, which makes it effortless to modify or compose sequential neural
   network architectures. The ``Unidirectional`` class supports both
   timestep-parallel training and autoregressive decoding. If you have two
   ``Unidirectional`` models that support both of these modes, you can compose
   them into a model that feeds the outputs of the first model as inputs to the
   seconds, and the composite model will also support both modes efficiently,
   for free. See `Composable Neural Networks`_.
#. The RNN and LSTM use learned initial hidden states.
#. None of the architectures have upper limits on sequence length. This
   includes the transformer, which uses sinusoidal positional encodings that
   can be extended arbitrarily. You can train on short sequences and evaluate
   on arbitrarily long sequences.
#. PyTorch uses two bias terms in the recurrent layers of the RNN and LSTM.
   However, only one is required, and the second one is redundant. Including
   the second term only serves to effectively double the learning rate of the
   bias term at the cost of adding additional parameters to the model. This
   means that RNNs and LSTMs can have speciously high parameter counts, which
   is undesirable if you are trying to compare different models with comparable
   parameter counts. Rau takes care to remove these redundant bias parameters,
   resulting in better parameter counts.
#. Supports minibatching with padding. For the sake of efficiency, Rau groups
   sequences of similar length together to reduce the number of padding tokens,
   and it enforces upper limits on the number of tokens in a minibatch.
#. Padding is handled correctly, in the sense that there is mathematically no
   difference between processing :math:`N` sequences in a single minibatch with
   padding and processing the same `N` sequences individually while
   accumulating their gradients. Minibatching is simply an implementation
   detail that increases speed.
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
   implement tokenization at all. It is compatible with datasets preprocessed
   by external tokenization tools, such as SentencePiece.
#. The dataset format is deliberately simple: plaintext consisting of one
   sequence per line, where each sequence consists of whitespace-separated
   tokens.
#. Offers different ways of handling UNK tokens. You can declare a particular
   token, such as ``<unk>``, to represent a catch-all UNK token. Or, you can
   disable UNK tokens entirely and treat out-of-vocabulary tokens as errors.
#. Beam search is parallelized across beam elements (but not minibatch
   elements).
#. Beam search terminates as soon as EOS is the top beam element, rather than
   waiting for the beam to fill up with EOS. This is correct because the
   a beam element can never have a descendant with higher probability than
   itself. The latter approach is only required if the scores can increase,
   e.g., when using certain kinds of length normalization.

Composable Neural Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^

This section is yet to be written.

Limitations
-----------

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
   decoding for transformers is very inefficient. At every step of decoding,
   all of the hidden representations are re-computed from scratch, and the
   model generates outputs for all previous timesteps, even though only the
   most recent one is needed. It does not implement what is commonly called "KV
   caching." The only things that are cached are the input embeddings. This
   might be fixed in the future.
#. It does not include tokenization and detokenization in the pipeline. You
   need to handle tokenization and detokenization yourself.
#. It slurps the entire training set into memory during training, so it will
   run out of memory on large datasets (~1m sequences). This might be fixed in
   the future.
#. Training cannot be stopped and restarted, so it cannot recover from crashes.
   This feature might be added in the future.

What does the name "Rau" mean?
------------------------------

The name is pronounced /ɹaʊ/ (rhymes with "now"). It's named after a
`magical mask <https://biomediaproject.com/bmp/data/sites/bionicle/2001/kanohi-noble.html>`_
that gives the person who wears it the ability to translate languages.
