import dataclasses
from collections.abc import Callable
from typing import Any, Optional

import torch

from rau.unidirectional import Unidirectional

class UnidirectionalBuiltinRNN(Unidirectional):
    """Wraps a built-in PyTorch RNN class in the :py:class:`Unidirectional`
    API."""

    RNN_CLASS: type

    def __init__(self,
        input_size: int,
        hidden_units: int,
        layers: int,
        dropout: Optional[float],
        bias: bool,
        use_extra_bias: bool,
        **kwargs: Any
    ):
        """
        :param input_size: The size of the input vectors to the RNN.
        :param hidden_units: The number of hidden units in each layer.
        :param layers: The number of layers in the RNN.
        :param dropout: The amount of dropout applied in between layers. If
            ``layers`` is 1, then this value is ignored.
        :param bias: Whether to use bias terms.
        :param use_extra_bias: The built-in PyTorch implementations of RNNs
            include redundant bias terms, resulting in more parameters than
            necessary. If this is true, the extra bias terms are kept.
            Otherwise, they are removed.
        :param kwargs: Additional arguments passed to the RNN constructor,
            such as ``nonlinearity``.
        """
        if dropout is None or layers == 1:
            dropout = 0.0
        super().__init__()
        self.rnn = self.RNN_CLASS(
            input_size,
            hidden_units,
            num_layers=layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
            bias=bias,
            **kwargs
        )
        if bias and not use_extra_bias:
            remove_extra_bias_parameters(self.rnn)
        self._hidden_units = hidden_units
        self._layers = layers

    def _initial_tensors(self, batch_size: int) -> tuple[Any, torch.Tensor]:
        raise NotImplementedError

    def _apply_to_hidden_state(self,
        hidden_state: Any,
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> Any:
        raise NotImplementedError

    @dataclasses.dataclass
    class State(Unidirectional.State):

        rnn: 'UnidirectionalBuiltinRNN'
        hidden_state: Any
        _output: torch.Tensor

        def next(self, input_tensor):
            # input_tensor : batch_size x input_size
            # unsqueezed_input : batch_size x 1 x input_size
            unsqueezed_input = input_tensor.unsqueeze(1)
            unsqueezed_output, new_hidden_state = self.rnn.rnn(
                unsqueezed_input,
                self.hidden_state)
            # unsqueezed_output : batch_size x 1 x hidden_units
            return self.rnn.State(
                self.rnn,
                new_hidden_state,
                unsqueezed_output.squeeze(1))

        def output(self):
            return self._output

        def batch_size(self):
            return self._output.size(0)

        def transform_tensors(self, func):
            return self.rnn.State(
                self.rnn,
                self.rnn._apply_to_hidden_state(
                    self.hidden_state,
                    # The builtin hidden states always have batch size as
                    # dim 1 instead of dim 0. So, temporarily move it to dim 0
                    # when func is called for compatibility.
                    # The builtin hidden states also need to be contiguous when
                    # they are used as inputs to the builtin module.
                    lambda x: func(x.transpose(0, 1)).transpose(0, 1).contiguous()
                ),
                func(self._output))

        def fastforward(self, input_sequence):
            """This method is overridden to use the builtin RNN class
            efficiently."""
            input_tensors = input_sequence.transpose(0, 1)
            output_sequence, state = self.forward(
                input_tensors,
                return_state=True,
                include_first=False)
            return state

        def outputs(self, input_sequence, include_first):
            """This method is overridden to use the builtin RNN class
            efficiently."""
            return self.forward(
                input_sequence,
                return_state=False,
                include_first=include_first)

        def forward(self, input_sequence, return_state, include_first):
            """This method is overridden to use the builtin RNN class
            efficiently."""
            # input_sequence : batch_size x sequence_length x input_size
            # self.output() : batch_size x hidden_units
            # Handle empty sequences, since the built-in RNN module does not
            # handle empty sequences (I checked).
            if input_sequence.size(1) == 0:
                first_output = self.output()
                if include_first:
                    output_sequence = first_output[:, None, :]
                else:
                    batch_size, hidden_units = first_output.size()
                    output_sequence = first_output.new_empty(batch_size, 0, hidden_units)
                if return_state:
                    return output_sequence, self
                else:
                    return output_sequence
            # output_sequence : batch_size x sequence_length x hidden_units
            # The type and size of new_hidden_state depends on the RNN unit.
            # For torch.nn.RNN, it's a tensor whose size is
            # num_layers x batch_size x hidden_units, where
            # new_hidden_state[-1] is the last layer and is equal to
            # output_sequence[:, -1].
            output_sequence, new_hidden_state = self.rnn.rnn(
                input_sequence,
                self.hidden_state)
            # output_sequence : batch_size x sequence_length x hidden_units
            if include_first:
                first_output = self.output()
                output_sequence = torch.cat([
                    first_output[:, None, :],
                    output_sequence
                ], dim=1)
            if return_state:
                # last_output : batch_size x hidden_units
                last_output = output_sequence[:, -1, :]
                state = self.rnn.State(self.rnn, new_hidden_state, last_output)
                return output_sequence, state
            else:
                return output_sequence

    def initial_state(self,
        batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> Unidirectional.State:
        if args or kwargs:
            raise ValueError
        hidden_state, output = self._initial_tensors(batch_size)
        return self.State(self, hidden_state, output)

def remove_extra_bias_parameters(module: torch.nn.Module):
    pairs = [
        (name, param)
        for name, param in module.named_parameters()
        if name.startswith('bias_hh_l')
    ]
    for name, param in pairs:
        # It looks like PyTorch's RNN modules keep some internal references to
        # the original parameter tensors even if the corresponding properties
        # are deleted. So, set each parameter to zero in-place and disable the
        # gradient so it never changes.
        param.data.zero_()
        param.requires_grad = False
        # Now, overwrite the property with param.data, which is a constant
        # zero tensor that is not a parameter. This ensures that the property
        # still exists but doesn't get counted as a parameter.
        # Register it as a buffer, which ensures that if the other parameters
        # are moved to a different device, it will be moved as well.
        delattr(module, name)
        module.register_buffer(name, param.data, persistent=False)
