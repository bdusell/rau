from typing import Optional

import torch

def pad_sequences(
    sequences: list[list[int]],
    device: torch.device,
    pad: int,
    bos: Optional[int]=None,
    eos: Optional[int]=None,
    return_lengths: bool=False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    max_length = max(map(len, sequences))
    add_bos = bos is not None
    bos_offset = int(add_bos)
    add_eos = eos is not None
    sequence_length = bos_offset + max_length + int(add_eos)
    result = torch.full(
        (batch_size, sequence_length),
        pad,
        dtype=torch.long,
        device=device
    )
    if add_bos:
        result[:, 0] = bos
    for i, sequence in enumerate(sequences):
        end_pos = bos_offset + len(sequence)
        result[i, bos_offset:end_pos] = sequence
        if add_eos:
            result[i, end_pos] = eos
    if return_lengths:
        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            device=device
        )
        result = result, lengths
    return result
