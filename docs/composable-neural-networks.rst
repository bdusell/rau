Composable Neural Networks
==========================

For those using the Python API of Rau, a useful feature that the library
provides is the ability to easily create new neural network modules by composing
simpler modules with the ``|`` operator, so that the output of one is used as
the input to the other. (The choice of ``|`` as the composition operator is
meant to evoke piping in shell languages.) If ``A`` and ``B`` are
:py:class:`~torch.nn.Module`\ s and ``A`` is also an instance of Rau's
:py:class:`~rau.tools.torch.compose.BasicComposable` class, then the expression
``A | B`` creates a new :py:class:`~torch.nn.Module` whose ``()`` operator
passes its input to ``A``, feeds the output of ``A`` as input to ``B``, and
returns the output of ``B``. This module is also an instance of
:py:class:`~rau.tools.torch.compose.BasicComposable`, so you can easily create a
pipeline of more than two modules like ``A | B | C | D | ...``. You can make any
:py:class:`~torch.nn.Module` an instance of
:py:class:`~rau.tools.torch.compose.BasicComposable` by wrapping it in
:py:class:`~rau.tools.torch.compose.Composable`.

.. code-block:: python

    import torch
    from torch.nn import Linear
    from rau.tools.torch.compose import Composable

    # Create a simple pipeline of Linear modules.
    # We only need to wrap the first module in Composable.
    M = Composable(Linear(3, 7)) | Linear(7, 5) | Linear(5, 11)

    # Feed an input to the composed module.
    x = torch.ones(3)
    y = M(x)

    # The output is the size of the output from the last module.
    assert y.size() == (11,)

This saves you the trouble of defining a custom :py:class:`~torch.nn.Module`
subclass that implements this pipeline.

The full API is documented in :py:mod:`rau.tools.torch.compose`.

Composable Sequential Neural Networks
-------------------------------------

This composition feature is especially useful when dealing with sequential
neural networks in Rau. Rau uses an abstraction for sequential neural networks
called :py:class:`~torch.unidirectional.Unidirectional`. A
:py:class:`~torch.unidirectional.Unidirectional` is a
:py:class:`~torch.nn.Module` that receives a variable-length sequence of
:py:class:`~torch.Tensor`\ s as input and produces an output
:py:class:`~torch.Tensor` for each input :py:class:`~torch.Tensor`. Moreover,
each output :py:class:`~torch.Tensor` may **not** have any data dependencies on
future inputs. As usual, a :py:class:`~torch.unidirectional.Unidirectional` has
a ``()`` operator, which receives the inputs stacked into a single
:py:class:`~torch.Tensor` along dimension 1 (the batch dimension is 0) and
returns the outputs similarly stacked into a single :py:class:`~torch.Tensor`.

.. code-block:: python

    import torch
    from rau.models.transformer.unidirectional_encoder import (
        get_unidirectional_transformer_encoder
    )

    # This instantiates a causally-masked transformer encoder (also
    # known as a "decoder-only" transformer). It is an instance of
    # Unidirectional.
    M = get_unidirectional_transformer_encoder(
        # This module will receive a sequence of tensors of size 3 as
        # input.
        input_vocabulary_size=5,
        # This module will produce a sequence of tensors of size 5 as
        # output.
        output_vocabulary_size=3,
        # Turn off dropout in order ot make the outputs deterministic
        # for this example.
        dropout=0,
        # The remaining arguments are not relevant for this example.
        tie_embeddings=False,
        num_layers=5,
        d_model=32,
        num_heads=4,
        feedforward_size=64,
        use_padding=False
    )
    # Batch size.
    B = 7
    # Sequence length.
    n = 11

    # Create a batch of sequences of integer inputs in the range [0, 5) of
    # length n. These are the "tokens" given to the transformer encoder.
    x = torch.randint(5, (B, n))

    # Use the () operator to get an output sequence of vectors.
    # The argument include_first=False tells the module that we do not
    # want it to attempt to produce an output before reading the first
    # input. This is not possible for transformers, but it is for RNNs,
    # which have an initial hidden state. For transformers, an output
    # corresponding to an initial BOS input serves the same purpose.
    y = M(x, include_first=False)
    assert y.size() == (B, n, 3)

It *also* has an :py:meth:`~torch.unidirectional.Unidirectional.initial_state`
method that returns a :py:class:`~torch.unidirectional.Unidirectional.State`
object, which can be used to receive inputs and return outputs iteratively using
its :py:class:`~torch.unidirectional.Unidirectional.State.next` and
:py:class:`~torch.unidirectional.Unidirectional.State.output` methods.

.. code-block:: python

    from torch.testing import assert_close

    state = M.initial_state(batch_size=B)
    # Call .next() to feed a new input to the current state and produce
    # a new state.
    state = state.next(x[:, 0])
    # Call .output() to get the output tensor of this state.
    # Because transformers have no initial output vector before reading
    # any inputs, calling .output() before .next() would have raised an
    # error.
    y1 = state.output()
    # The output of this state is a single vector of size 3 and is
    # equivalent to the first element of the output of ().
    assert y1.size() == (B, 3)
    assert_close(y1, y[:, 0])
    # Do the same thing for a second iteration.
    state = state.next(x[:, 1])
    y2 = state.output()
    assert y2.size() == (B, 3)
    assert_close(y2, y[:, 1])

These two modes are useful in different scenarios. The ``()`` method can be
overridden to parallelize computation across the sequence dimension, making it
more efficient than the iterative mode. This makes the ``()`` method useful for
training, where future inputs are always known in advance. The iterative mode is
useful when future inputs are *not* known in advance, namely when generating
sequences from language models or decoders in machine translation systems.

:py:class:`~torch.unidirectional.Unidirectional`\ s can also be composed with
the ``|`` operator. If ``A`` and ``B`` are both
:py:class:`~torch.unidirectional.Unidirectional`\ s, then the expression ``A |
B`` returns another :py:class:`~torch.unidirectional.Unidirectional` that feeds
its inputs to ``A``, feeds the outputs of ``A`` as inputs to ``B``, and returns
the outputs of ``B``. Like ``A`` and ``B``, the
:py:class:`~torch.unidirectional.Unidirectional` returned by ``A | B`` also
supports both ``()`` and iterative modes. If ``A`` and ``B`` implement their
``()`` and iterative modes efficiently, then ``A | B`` gives you a composed
module that implements both modes efficiently for free.

The full API is documented in :doc:`rau.unidirectional`.

Argument Routing
----------------

What if you try to compose modules that require multiple arguments? For example,
if you have a module ``A`` that takes no keyword arguments, a module ``B`` that
requires a keyword argument ``foo``, and a module ``C`` that requires keyword
arguments ``bar`` and ``baz``, how do you invoke ``A | B | C``? Rau handles this
by allowing you to add tags to modules that signal which modules should receive
which arguments.

.. code-block:: python

    # Create a pipeline where individual modules have been tagged.
    M = A | B.tag('b') | C.tag('c')
    x = torch.rand(B, n, A_input_size)
    y = M(
        # x will be passed as input to A, whose output will be passed
        # as input B, whose output will be passed as input to C, whose
        # output will be returned as y.
        x,
        # tag_kwargs is a dict that maps tags to dicts of keyword
        # arguments. The keyword argument foo=123 will be passed to B,
        # and the keywords bar=456 and baz=789 will be passed to C.
        tag_kwargs=dict(
            b=dict(foo=123),
            c=dict(
                bar=456,
                baz=789
            )
        )
    )

You can make this more succinct by designating at most one module in a pipeline
as the "main" module, which will receive any extra positional or keyword
arguments. This is useful when wrapping a module with input and output layers.

.. code-block:: python

    # Create a pipeline where B is tagged with 'b' and C is the main
    # module.
    M = A | B.tag('b') | C.main()
    x = torch.rand(B, n, A_input_size)
    y = M(
        x,
        bar=456,
        baz=789,
        tag_kwargs=dict(
            b=dict(foo=123)
        )
    )
