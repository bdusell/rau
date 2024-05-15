from typing import Optional

import torch

from .builtin import UnidirectionalBuiltinRNN

class LSTM(UnidirectionalBuiltinRNN):
    """An LSTM wrapped in the :py:class:`Unidirectional` API."""

    def __init__(self,
        input_size: int,
        hidden_units: int,
        layers: int=1,
        dropout: Optional[float]=None,
        bias: bool=True,
        learned_hidden_state: bool=False,
        use_extra_bias: bool=False
    ):
        """
        :param input_size: The size of the input vectors to the LSTM.
        :param hidden_units: The number of hidden units in each layer.
        :param layers: The number of layers in the LSTM.
        :param dropout: The amount of dropout applied in between layers. If
            ``layers`` is 1, then this value is ignored.
        :param bias: Whether to use bias terms.
        :param learned_hidden_state: Whether the initial hidden state should be
            a learned parameter. If true, the initial hidden state will be the
            result of passing learned parameters through the tanh activation
            function. If false, the initial state will be zeros. The initial
            memory cell is always zeros.
        :param use_extra_bias: The built-in PyTorch implementation of the LSTM
            includes redundant bias terms, resulting in more parameters than
            necessary. If this is true, the extra bias terms are kept.
            Otherwise, they are removed.
        """
        super().__init__(
            input_size=input_size,
            hidden_units=hidden_units,
            layers=layers,
            dropout=dropout,
            bias=bias,
            use_extra_bias=use_extra_bias
        )
        self.learned_hidden_state = learned_hidden_state
        if learned_hidden_state:
            self.initial_hidden_state_inputs = torch.nn.Parameter(torch.zeros(layers, hidden_units))

    RNN_CLASS = torch.nn.LSTM

    def _initial_tensors(self, batch_size):
        c = torch.zeros(
            self._layers,
            batch_size,
            self._hidden_units,
            device=next(self.parameters()).device
        )
        if self.learned_hidden_state:
            h = torch.tanh(
                self.initial_hidden_state_inputs
            )[:, None, :].repeat(1, batch_size, 1)
        else:
            h = c
        return (h, c), h[-1]

    def _apply_to_hidden_state(self, hidden_state, func):
        return tuple(map(func, hidden_state))
