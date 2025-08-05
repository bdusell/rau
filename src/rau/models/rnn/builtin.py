import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from rau.unidirectional import Unidirectional, ForwardResult
from rau.unidirectional.util import unwrap_output_tensor

class UnidirectionalBuiltinRNN(Unidirectional):
    """Wraps a built-in PyTorch RNN class in the :py:class:`Unidirectional`
    API."""

    _RNN_CLASS: type

    def __init__(self,
        input_size: int,
        hidden_units: int,
        layers: int,
        dropout: float | None,
        bias: bool,
        use_extra_bias: bool,
        **kwargs: Any
    ) -> None:
        r"""
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
        self.rnn = self._RNN_CLASS(
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

    def _initial_hidden_state(self, batch_size: int) -> Any:
        raise NotImplementedError

    def _apply_to_hidden_state(self,
        hidden_state: Any,
        func: Callable[[torch.Tensor], torch.Tensor]
    ) -> Any:
        raise NotImplementedError

    def _hidden_state_to_output(self, hidden_state: Any) -> torch.Tensor:
        raise NotImplementedError

    def _hidden_state_to_batch_size(self, hidden_state: Any) -> int:
        raise NotImplementedError

    @dataclasses.dataclass
    class State(Unidirectional.State):

        rnn: 'UnidirectionalBuiltinRNN'
        hidden_state: Any

        def next(self, input_tensor: torch.Tensor) -> Unidirectional.State:
            # input_tensor : batch_size x input_size
            # unsqueezed_input : batch_size x 1 x input_size
            unsqueezed_input = input_tensor.unsqueeze(1)
            _, new_hidden_state = self.rnn.rnn(
                unsqueezed_input,
                self.hidden_state
            )
            return dataclasses.replace(
                self,
                hidden_state=new_hidden_state
            )

        def output(self) -> torch.Tensor:
            return self.rnn._hidden_state_to_output(self.hidden_state)

        def forward(self,
            input_sequence: torch.Tensor,
            include_first: bool,
            return_state: bool = False,
            return_output: bool = True
        ) -> torch.Tensor | ForwardResult:
            # input_sequence : batch_size x sequence_length x input_size
            # The built-in RNN modules do not handle empty input sequences, so
            # handle that as a special case here.
            if input_sequence.size(1) == 0:
                if return_output:
                    first_output = self.output()
                    batch_size, hidden_units = first_output.size()
                    if include_first:
                        output_sequence = first_output.unsqueeze(1)
                    else:
                        output_sequence = first_output.new_empty(batch_size, 0, hidden_units)
                else:
                    output_sequence = None
                if return_state:
                    state = self
                else:
                    state = None
            else:
                # output_sequence : batch_size x sequence_length x hidden_units
                # The type and size of new_hidden_state depends on the RNN unit.
                # For torch.nn.RNN, it's a tensor whose size is
                # num_layers x batch_size x hidden_units, where
                # new_hidden_state[-1] is the last layer and is equal to
                # output_sequence[:, -1].
                output_sequence, new_hidden_state = self.rnn.rnn(
                    input_sequence,
                    self.hidden_state
                )
                if return_output:
                    if include_first:
                        output_sequence = torch.concat([
                            self.output().unsqueeze(1),
                            output_sequence
                        ], dim=1)
                else:
                    output_sequence = None
                if return_state:
                    state = dataclasses.replace(
                        self,
                        hidden_state=new_hidden_state
                    )
                else:
                    state = None
            return unwrap_output_tensor(ForwardResult(
                output=output_sequence,
                extra_outputs=[],
                state=state
            ))

        def batch_size(self) -> int:
            return self.rnn._hidden_state_to_batch_size(self.hidden_state)

        def transform_tensors(self,
            func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Unidirectional.State:
            return dataclasses.replace(
                self,
                hidden_state=self.rnn._apply_to_hidden_state(
                    self.hidden_state,
                    # The builtin hidden states always have batch size as
                    # dim 1 instead of dim 0. So, temporarily move it to dim 0
                    # when func is called, for compatibility.
                    # The builtin hidden states also need to be contiguous when
                    # they are used as inputs to the builtin module.
                    lambda x: func(x.transpose(0, 1)).transpose(0, 1).contiguous()
                )
            )

    def initial_state(self, batch_size: int) -> Unidirectional.State:
        return self.State(self, self._initial_hidden_state(batch_size))

def remove_extra_bias_parameters(module: torch.nn.Module) -> None:
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
