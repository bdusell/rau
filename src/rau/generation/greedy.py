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
    initial_batch_size = initial_state.batch_size()
    # symbols : [ current_batch_size of ints in [0, output_vocab_size) ] x longest_output_length
    symbols = []
    # symbol_batch_indexes : [ current_batch_size of ints in [0, initial_batch_size) ] x longest_output_length
    symbol_batch_indexes = []
    if max_length is None or max_length > 0:
        state = initial_state
        curr_symbol_batch_indexes = None
        t = 0
        while True:
            # output_logits : current_batch_size x output_vocab_size
            output_logits = state.output()
            # next_symbol : current_batch_size of ints in [0, output_vocab_size)
            next_symbol = torch.argmax(output_logits, dim=1)
            next_is_not_eos = (next_symbol != eos_symbol)
            # Slice out the generated symbols that are not EOS.
            next_non_eos_symbol = next_symbol[next_is_not_eos]
            if next_non_eos_symbol.size(0) == 0:
                # All output sequences have terminated. Stop now.
                break
            # Remember the symbols generated.
            symbols.append(next_non_eos_symbol)
            # Remember which batch elements the generated symbols belong to.
            if curr_symbol_batch_indexes is None:
                # Wait until here to initialize so we know which device to put
                # it on.
                curr_symbol_batch_indexes = torch.arange(initial_batch_size, device=output_logits.device)
            curr_symbol_batch_indexes = curr_symbol_batch_indexes[next_is_not_eos]
            symbol_batch_indexes.append(curr_symbol_batch_indexes)
            # Stop now if the maximum length is reached.
            if max_length is not None and t >= max_length - 1:
                break
            # Reslice the state so that only non-terminated batch elements are
            # continued.
            state = state.transform_tensors(lambda x: x[next_is_not_eos, ...])
            state = state.next(next_non_eos_symbol)
            t += 1
    results = [[] for _ in range(initial_batch_size)]
    for symbols_t, symbol_batch_indexes_t in zip(symbols, symbol_batch_indexes):
        for symbol, batch_index in zip(symbols_t.tolist(), symbol_batch_indexes_t.tolist()):
            results[batch_index].append(symbol)
    return results
