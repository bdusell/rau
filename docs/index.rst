Rau
===

Rau (rhymes with "now") is a Python module and command-line tool that provides
PyTorch implementations of neural network-based language modeling and
sequence-to-sequence generation. It is primarily suited for academic
researchers. Out of the box, it provides implementations of the simple
recurrent neural network (RNN), long short-term memory (LSTM), and transformer
architectures. It includes extensible Python APIs and command-line tools for
data preprocessing, training, and evaluation. It is very easy to get started
with the command-line tools if you provide your data in plaintext as lines of
space-separated tokens.

There are, of course, many excellent implementations of neural network training
out in the world for you to choose from, but I think Rau does have some neat
features and avoids some pitfalls that I have not seen handled well in other
codebases. To see if using Rau is a good idea for you, see :doc:`details`.

Rau has been used in the following papers:

* Training Neural Networks as Recognizers of Formal Languages :cite:p:`butoi-etal-2025-training`
* Information Locality as an Inductive Bias for Neural Language Models :cite:p:`someya-etal-2025-information`

Rau is based on code that was originally used for
`adding stack data structures to LSTMs and transformers <https://github.com/bdusell/stack-attention>`_.
A version of Rau that includes implementations of stack-augmented neural
network architectures can be found on
`this branch <https://github.com/bdusell/rau/tree/differentiable-stacks>`_.

What Can Rau Be Used For?
-------------------------

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

What Does the Name "Rau" Mean?
------------------------------

The name is pronounced /ɹaʊ/ (rhymes with "now"). It's named after a
`magical mask <https://biomediaproject.com/bmp/data/sites/bionicle/2001/kanohi-noble.html>`_
that gives the person who wears it the ability to translate languages.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   details
   composable-neural-networks
   rau.generation
   rau.models
   rau.tasks
   rau.tools.torch
   rau.unidirectional
   rau.vocab
   miscellaneous-tools
   development
   bibliography
