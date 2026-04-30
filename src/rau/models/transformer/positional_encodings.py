import math

import torch

NEG_LOG_10K = -math.log(10000.0)

def sinusoidal_positional_encodings(sequence_length, d_model, device):
    # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    if sequence_length == 0 or d_model == 0:
        return torch.empty((sequence_length, d_model), device=device)
    # position : sequence_length x 1
    position = torch.arange(sequence_length, device=device).unsqueeze(1)
    # div_term : 1 x d_model/2 or 1 x d_model/2+1, depending on whether d_model
    # is odd.
    div_term = torch.exp(
        torch.arange(0, d_model + d_model % 2, 2, device=device) *
        (NEG_LOG_10K / d_model)
    ).unsqueeze(0)
    # pe : sequence_length x d_model
    pe = torch.empty(sequence_length, d_model, device=device)
    # TODO I'm sure sin and cos can be parallelized by simply changing the
    # phase of sin.
    pe[:, 0::2] = torch.sin(position * div_term)
    # Truncate div_term in case d_model is odd.
    pe[:, 1::2] = torch.cos(position * div_term[:, :d_model//2])
    return pe

class SinusoidalPositionalEncodingCacher(torch.nn.Module):
    r"""A module that caches a tensor of sinusoidal positional encodings.

    This module can dynamically resize the cached tensor as needed, but it is
    **highly recommended** to set a maximum size up-front at the beginning of
    your program using :py:meth`get_encodings` (for example, by looping through
    the training data) and then disable dynamic resizing using
    :py:meth:`set_allow_realloation` to avoid CUDA memory fragmentation.
    Otherwise, you may run out of memory in a way that is very hard to debug.
    """

    _d_model: int
    _allow_reallocation: bool

    def __init__(self, d_model: int) -> None:
        r"""
        :param d_model: The :math:`d_\mathrm{model}` of the positional
            encodings.
        """
        super().__init__()
        self._d_model = d_model
        self._set_cache_size_with_device(0, None)
        self._allow_reallocation = True

    def clear(self) -> None:
        r"""Clear the cache."""
        self._set_cache_size(0)

    def _set_cache_size(self, length: int) -> None:
        self._set_cache_size_with_device(length, self.encodings.device)

    def _set_cache_size_with_device(self, length, device):
        self._set_encodings(sinusoidal_positional_encodings(
            length,
            self._d_model,
            device
        ))

    def _set_encodings(self, tensor):
        self.register_buffer('encodings', tensor, persistent=False)

    def get_encodings(self, sequence_length: int) -> torch.Tensor:
        r"""Get a tensor of positional encodings of the requested size.

        :param sequence_length: Get positional encodings up to this length.
        :return: A tensor of positional encodings of the requested size.
        """
        if sequence_length > self.encodings.size(0):
            if not self._allow_reallocation:
                raise ValueError(
                    'reallocation of the positional encoding cache has been '
                    'intentionally disabled with set_allow_reallocation(False)'
                )
            self._set_cache_size(sequence_length)
        return self.encodings[:sequence_length]

    def set_allow_reallocation(self, value: bool) -> None:
        r"""Set whether reallocating the tensor dynamically based on requested
        sizes should be enabled. By default, it is enabled. If it is disabled,
        requesting a size bigger than the currently cached tensor will cause an
        error. After setting a maximum size with :py:meth:`get_encodings`, the
        advantage of disabling it is that it will treat requests for bigger
        sizes (which would imply that the way you determined the maximum length
        has a bug) as errors rather than silently allowing them to cause memory
        fragmentation.

        :param value: Whether to allow reallocation.
        """
        self._allow_reallocation = value
