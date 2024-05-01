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

For this example, we'll train a language model on simple declarative sentences
in English (the data comes from the source side of the question formation task
from McCoy et al., 2020).

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
      --training-data language-modeling-example \
      --input test \
      --load-model saved-language-model \
      --batching-max-tokens 2048

Sequence-to-Sequence
^^^^^^^^^^^^^^^^^^^^

TODO

Should You Use Rau?
-------------------

TODO

What does the name "Rau" mean?
------------------------------

The name is pronounced /ɹaʊ/ (rhymes with "cow"). It's named after a
`magical mask <https://biomediaproject.com/bmp/data/sites/bionicle/2001/kanohi-noble.html>`_
that gives the person who wears it the ability to translate languages.
