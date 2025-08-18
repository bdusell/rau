from collections.abc import Iterable

import torch

from rau.unidirectional import Unidirectional
from rau.generation import decode_greedily

from test_beam_search import RandomState

def decode_greedily_single(
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
        # output_logits : output_vocab_size
        output_logits = state.output().squeeze(0)
        # next_symbol : ()
        next_symbol = torch.argmax(output_logits, dim=0)
        next_symbol_int = next_symbol.item()
        if next_symbol_int == eos_symbol:
            break
        yield next_symbol_int
        if max_length is not None and t >= max_length - 1:
            break
        state = state.next(next_symbol.unsqueeze(0))
        t += 1

def test_batched_matches_reference():
    batch_size = 17
    output_size = 7
    eos = output_size - 1
    max_length = 50
    initial_state = RandomState(list(range(batch_size)), output_size)
    result = decode_greedily(
        initial_state,
        eos_symbol=eos,
        max_length=max_length
    )
    for i, result_i in enumerate(result):
        result_i = list(result_i)
        expected_i = list(decode_greedily_single(
            initial_state=initial_state.transform_tensors(lambda x: x[i:i+1]),
            eos_symbol=eos,
            max_length=max_length,
        ))
        assert result_i == expected_i
