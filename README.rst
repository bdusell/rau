Rau
===

Rau (rhymes with "cow") is a Python module that provides PyTorch
implementations of neural language modeling and sequence-to-sequence
transduction. Out of the box, it provides implementations of the simple
recurrent neural network (RNN), long short-term memory (LSTM), and transformer
architectures, as well as data preparation, training loops, and evaluation
scripts for each task. Rau is designed to make it easy to modify the underlying
neural network architectures and training loops, while providing some
reasonable defaults.

Does the world need another implementation of neural network training? Perhaps
not, but Rau does have some neat tricks I haven't seen used in other codebases.
It will likely be particularly useful for people who want to tinker with new
sequential neural network architectures. It was originally based on
`code for adding stack data structures to LSTMs and transformers <https://github.com/bdusell/stack-attention>`_.

To see if using Rau is a good idea for you, see `Should You Use Rau?`_.

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

Install Python dependencies. This can be done by installing the package manager
`Poetry <https://python-poetry.org/docs/#installation>`_
(it's already installed in the Docker container) and running this script::

    bash scripts/install_python_packages.bash

Start a shell inside the Python virtual environment using Poetry::

    poetry shell

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

Should You Use Rau?
-------------------

TODO

What does the name "Rau" mean?
------------------------------

The name is pronounced /ɹaʊ/ (rhymes with "cow"). It's named after a
`magical mask <https://biomediaproject.com/bmp/data/sites/bionicle/2001/kanohi-noble.html>`_
that gives the person who wears it the ability to translate languages.
