from collections.abc import Iterable

import torch

from rau.generation import sample
from rau.unidirectional import Unidirectional

from test_beam_search import RandomState

def sample_single(
    initial_state: Unidirectional.State,
    eos_symbol: int,
    max_length: int | None = None,
    generator: torch.Generator | None = None
) -> Iterable[int]:
    if max_length is not None and max_length < 1:
        return
    if initial_state.batch_size() != 1:
        raise ValueError
    state = initial_state
    t = 0
    while True:
        # output_probs : output_vocab_size
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

def test_batched_matches_single():
    batch_size = 1
    num_samples = 1
    output_size = 7
    eos = output_size - 1
    max_length = 50
    initial_state = RandomState(list(range(batch_size)), output_size)
    for seed in range(10):
        generator = torch.manual_seed(seed)
        expected = list(sample_single(
            initial_state,
            eos_symbol=eos,
            max_length=max_length,
            generator=generator
        ))
        generator = torch.manual_seed(seed)
        ((result,),) = sample(
            initial_state,
            eos_symbol=eos,
            num_samples=1,
            max_length=max_length,
            generator=generator
        )
        assert result == expected
