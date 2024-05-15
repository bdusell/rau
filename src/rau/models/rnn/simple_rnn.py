from typing import Literal, Optional

import torch

from .builtin import UnidirectionalBuiltinRNN

class SimpleRNN(UnidirectionalBuiltinRNN):
    """A simple RNN wrapped in the :py:class:`Unidirectional` API."""

    def __init__(self,
        input_size: int,
        hidden_units: int,
        layers: int=1,
        dropout: Optional[float]=None,
        nonlinearity: Literal['tanh', 'relu']='tanh',
        bias: bool=True,
        learned_hidden_state: bool=False,
        use_extra_bias: bool=False
    ):
        """
        :param input_size: The size of the input vectors to the RNN.
        :param hidden_units: The number of hidden units in each layer.
        :param layers: The number of layers in the RNN.
        :param dropout: The amount of dropout applied in between layers. If
            ``layers`` is 1, then this value is ignored.
        :param nonlinearity: The non-linearity applied to hidden units. Either
            ``'tanh'`` or ``'relu'``.
        :param bias: Whether to use bias terms.
        :param learned_hidden_state: Whether the initial hidden state should be
            a learned parameter. If true, the initial hidden state will be the
            result of passing learned parameters through the activation
            function. If false, the initial state will be zeros.
        :param use_extra_bias: The built-in PyTorch implementation of the RNN
            includes redundant bias terms, resulting in more parameters than
            necessary. If this is true, the extra bias terms are kept.
            Otherwise, they are removed.
        """
        super().__init__(
            input_size=input_size,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            nonlinearity=nonlinearity,
            bias=bias,
            use_extra_bias=use_extra_bias
        )
        self.learned_hidden_state = learned_hidden_state
        if learned_hidden_state:
            self.initial_hidden_state_inputs = torch.nn.Parameter(torch.zeros(layers, hidden_units))
            if nonlinearity == 'tanh':
                self.activation_function = torch.tanh
            else:
                self.activation_function = torch.nn.functional.relu

    RNN_CLASS = torch.nn.RNN

    def _initial_tensors(self, batch_size):
        # The initial tensor is a tensor of all the hidden states of all layers
        # before the first timestep.
        # Its size needs to be num_layers x batch_size x hidden_units, where
        # index 0 is the first layer and -1 is the last layer.
        # Note that the batch dimension is always the second dimension even
        # when batch_first=True.
        if self.learned_hidden_state:
            h = self.activation_function(
                self.initial_hidden_state_inputs
            )[:, None, :].repeat(1, batch_size, 1)
        else:
            h = torch.zeros(
                self._layers,
                batch_size,
                self._hidden_units,
                device=next(self.parameters()).device
            )
        return h, h[-1]

    def _apply_to_hidden_state(self, hidden_state, func):
        return func(hidden_state)
