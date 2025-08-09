Getting Started
===============

Installation
------------

To install the differentiable-stacks branch of Rau directly from GitHub with
pip, do one of the following:

.. code-block:: sh

    pip install git+https://github.com/bdusell/rau.git@differentiable-stacks

.. code-block:: sh

    pip install git+ssh://git@github.com/bdusell/rau.git@differentiable-stacks

Examples
--------

Below are some quick examples of setting up pipelines for language modeling and
sequence-to-sequence transduction. We will use the pretokenized datasets from
:cite:t:`mccoy-etal-2020-syntax`; their simplicity and small size make them
convenient for our purposes.

Language Modeling
^^^^^^^^^^^^^^^^^

For this example, we'll train a transformer language model on simple
declarative sentences in English (the data comes from the source side of the
question formation task of :cite:t:`mccoy-etal-2020-syntax`).

Download the dataset:

.. code-block:: sh

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
converting all tokens to integers:

.. code-block:: sh

    rau lm prepare \
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

Now, train a transformer language model:

.. code-block:: sh

    rau lm train \
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

Calculate the perplexity of this language model on the test set:

.. code-block:: sh

    rau lm evaluate \
      --load-model saved-language-model \
      --training-data language-modeling-example \
      --input test

Randomly sample 10 sequences from the trained language model:

.. code-block:: sh

    rau lm generate \
      --load-model saved-language-model \
      --training-data language-modeling-example \
      --num-samples 10

Sequence-to-Sequence
^^^^^^^^^^^^^^^^^^^^

For this example, we'll train a transformer encoder-decoder on the question
formation task of :cite:t:`mccoy-etal-2020-syntax`, which involves converting a
declarative sentence in English to question form.

Download the dataset:

.. code-block:: sh

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
converting all tokens to integers:

.. code-block:: sh

    rau ss prepare \
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

* ``sequence-to-sequence-example/``

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

Now, train a transformer encoder-decoder model:

.. code-block:: sh

    rau ss train \
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

Finally, translate the source sequences in the test data using beam search:

.. code-block:: sh

    rau ss translate \
      --load-model saved-sequence-to-sequence-model \
      --input sequence-to-sequence-example/datasets/test/source.shared.prepared \
      --beam-size 4 \
      --max-target-length 50 \
      --batching-max-tokens 256 \
      --shared-vocabulary-file sequence-to-sequence-example/shared.vocab
