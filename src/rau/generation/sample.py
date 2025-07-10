import random
from collections.abc import Iterable

import torch

from rau.unidirectional import Unidirectional

def sample(
    initial_state: Unidirectional.State,
    eos_symbol: int,
    max_length: int,
    device: torch.device
) -> list[list[int]]:
    r"""Given a state of an autoregressive language model containing any number
    of batch elements, generate a sequence for each element using ancestral
    sampling.
    """
    batch_size = initial_state.batch_size()
    return [
        sample_single(
            initial_state.transform_tensors(lambda x: x[i:i+1, ...]),
            eos_symbol,
            max_length,
            device
        )
        for i in range(batch_size)
    ]

def sample_single(
    initial_state: Unidirectional.State,
    eos_symbol: int,
    max_length: int | None = None,
    generator: torch.Generator | None = None
) -> Iterable[int]:
    if max_length < 1:
        return
    if initial_state.batch_size() != 1:
        raise ValueError
    state = initial_state
    t = 0
    while True:
        # output_probs : batch_size x output_vocab_size
        output_probs = torch.nn.functional.softmax(
            state.output().squeeze(0),
            dim=0
        )
        # next_symbol : 1
        next_symbol = torch.multinomial(
            output_probs,
            num_samples=1,
            generator=generator
        )
        next_symbol_int = next_symbol.item()
        if next_symbol_int == eos_symbol:
            break
        yield next_symbol_int
        if max_length is not None and t >= max_length - 1:
            break
        state = state.next(next_symbol)
        t += 1