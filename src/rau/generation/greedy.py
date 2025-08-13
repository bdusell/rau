from collections.abc import Iterable

import torch

from rau.unidirectional import Unidirectional

def decode_greedily(
    initial_state: Unidirectional.State,
    eos_symbol: int,
    max_length: int
) -> list[list[int]]:
    r"""Given a state of an autoregressive language model or decoder containing
    any number of batch elements, generate a sequence for each element using
    greedy decoding.

    :param initial_state: A state of an autoregressive decoder or language model
        from which decoding starts, containing any number of batch elements. A
        separate sequence will be decoded for each of the initial batch
        elements. Note that this does not actually need to be the *initial*
        state of a decoder; decoding can start from any state.
    :param eos_symbol: Identifier of a designated end-of-sequence (EOS) symbol
        that indicates that the model should stop generating symbols for a
        sequence.
    :param max_length: A hard upper limit on the number of symbols in the
        generated sequences.
    :return: A list of generated sequences, one per batch element in the initial
        state.
    """
    batch_size = initial_state.batch_size()
    return [
        decode_greedily_single(
            initial_state.transform_tensors(lambda x: x[i:i+1, ...]),
            eos_symbol,
            max_length
        )
        for i in range(batch_size)
    ]

def decode_greedily_single(
    initial_state: Unidirectional.State,
    eos_symbol: int,
    max_length: int | None = None
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
