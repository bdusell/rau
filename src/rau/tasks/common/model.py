import torch

def pad_sequences(sequences, device, pad, bos=None, eos=None):
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
        result[i, bos_offset:bos_offset+len(sequence)] = sequence
        if add_eos:
            result[i, bos_offset+len(sequence)] = eos
    return result
