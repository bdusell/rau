from collections.abc import Iterable

import torch

from rau.unidirectional import Unidirectional

def sample(
    initial_state: Unidirectional.State,
    eos_symbol: int,
    num_samples: int = 1,
    max_length: int | None = None,
    generator: torch.Generator | None = None
) -> list[list[list[int]]]:
    r"""Given a state of an autoregressive language model or decoder containing
    any number of batch elements, randomly generate ``num_samples`` sequences
    for each element using ancestral sampling. Sampling is parallelized across
    batch elements and samples.

    :param initial_state: A state of an autoregressive decoder or language model
        from which decoding starts, containing any number of batch elements. A
        separate sequence will be decoded for each of the initial batch
        elements. Note that this does not actually need to be the *initial*
        state of a decoder; decoding can start from any state.
    :param eos_symbol: Identifier of a designated end-of-sequence (EOS) symbol
        that indicates that the model should stop generating symbols for a
        sequence.
    :param num_samples: Number of samples per batch element.
    :param max_length: A hard upper limit on the number of symbols in the
        generated sequences.
    :param generator: Optional random number generator to make sampling
        deterministic.
    :return: Randomly generated sequences for each batch element. This will be a
        list of lists of sequences, where each batch element has a list
        containing ``num_samples`` sequences.
    """
    initial_batch_size = initial_state.batch_size()
    # Create num_samples copies of each batch element in the state.
    initial_state = initial_state.transform_tensors(lambda x: torch.repeat_interleave(x, num_samples, dim=0))
    # Let current_num_active be, at a given point in time, the number of
    # sequences that are still actively being generated across all batch
    # elements and requested samples. Initially,
    # current_num_active = initial_batch_size * num_samples, and it can
    # decrease as individual samples generate EOS and terminate.
    # symbols : [ current_num_active of ints in [0, output_vocab_size) ] x longest_output_length
    symbols = []
    # symbol_indexes : [ current_num_active x 2 of ints ] x longest_output_length
    # symbol_indexes[:, 0] contains ints in [0, initial_batch_size)
    # symbol_indexes[:, 1] contains ints in [0, num_samples)
    symbol_indexes = []
    if max_length is None or max_length > 0:
        state = initial_state
        curr_symbol_indexes = None
        t = 0
        while True:
            # output_probs : current_num_active x output_vocab_size
            output_probs = torch.nn.functional.softmax(state.output(), dim=1)
            # next_symbol : current_num_active of ints in [0, output_vocab_size)
            next_symbol = torch.multinomial(
                output_probs,
                num_samples=1,
                generator=generator
            ).squeeze(1)
            next_is_not_eos = (next_symbol != eos_symbol)
            # Slice out the generated symbols that are not EOS.
            next_non_eos_symbol = next_symbol[next_is_not_eos]
            if next_non_eos_symbol.size(0) == 0:
                # All output sequences have terminated. Stop now.
                break
            # Remember the symbols generated.
            symbols.append(next_non_eos_symbol)
            # Remember which batch elements and samples the generated symbols
            # belong to.
            if curr_symbol_indexes is None:
                # Wait until here to initialize so we know which device to put
                # it on.
                curr_symbol_indexes = torch.cartesian_prod(
                    torch.arange(initial_batch_size, device=output_probs.device),
                    torch.arange(num_samples, device=output_probs.device)
                )
            curr_symbol_indexes = curr_symbol_indexes[next_is_not_eos, ...]
            symbol_indexes.append(curr_symbol_indexes)
            # Stop now if the maximum length is reached.
            if max_length is not None and t >= max_length - 1:
                break
            # Reslice the state so that only non-terminated batch elements are
            # continued.
            state = state.transform_tensors(lambda x: x[next_is_not_eos, ...])
            state = state.next(next_non_eos_symbol)
            t += 1
    results = [
        [[] for _ in range(num_samples)]
        for _ in range(initial_batch_size)
    ]
    for symbols_t, symbol_indexes_t in zip(symbols, symbol_indexes):
        for symbol, (batch_index, sample_index) in zip(symbols_t.tolist(), symbol_indexes_t.tolist()):
            results[batch_index][sample_index].append(symbol)
    return results
