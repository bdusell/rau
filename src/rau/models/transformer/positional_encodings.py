import math

import torch

def sinusoidal_positional_encodings(sequence_length, d_model, device):
    # Based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    if sequence_length == 0 or d_model == 0:
        return torch.empty((sequence_length, d_model), device=device)
    # TODO This doesn't work when d_model is odd.
    position = torch.arange(sequence_length, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device) *
        (-math.log(10000.0) / d_model)
    )
    pe = torch.empty(sequence_length, d_model, device=device)
    # TODO I'm sure sin and cos can be parallelized by simply changing the
    # phase of sin.
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
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

    def __init__(self) -> None:
        super().__init__()
        self._set_cache_size_with_device((0, 0), None)
        self._allow_reallocation = True

    def clear(self) -> None:
        r"""Clear the cache."""
        self._set_cache_size((0, 0))

    def _set_cache_size(self, size):
        self._set_cache_size_with_device(size, self.encodings.device)

    def _set_cache_size_with_device(self, size, device):
        sequence_length, d_model = size
        self._set_encodings(sinusoidal_positional_encodings(
            sequence_length,
            d_model,
            device
        ))

    def _set_encodings(self, tensor):
        self.register_buffer('encodings', tensor, persistent=False)

    def get_encodings(self, sequence_length: int, d_model: int) -> torch.Tensor:
        r"""Get a tensor of positional encodings of the requested size.

        :param sequence_length: Get positional encodings up to this length.
        :param d_model: The :math:`d_\mathrm{model}` of the positional
            encodings.
        :return: A tensor of positional encodings of the requested size.
        """
        query_size = (sequence_length, d_model)
        cache_size = self.encodings.size()
        if not all(a <= b for a, b in zip(query_size, cache_size)):
            if not self._allow_reallocation:
                raise ValueError(
                    'reallocation of the positional encoding cache has been '
                    'intentionally disabled with set_allow_reallocation(False)'
                )
            # Make sure never to decrease the cached sequence_length or
            # d_model to avoid flip-flopping.
            new_size = tuple(max(a, b) for a, b in zip(query_size, cache_size))
            self._set_cache_size(new_size)
        return self.encodings[:sequence_length, :d_model]

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
