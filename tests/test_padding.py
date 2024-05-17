import random

import torch

from rau.tasks.common.model import pad_sequences

def test_pad_sequences():
    generator = random.Random(123)
    alphabet_size = 10
    e = alphabet_size
    b = alphabet_size + 1
    p = alphabet_size + 2
    device = torch.device('cpu')
    sequences = [
        [5, 7, 1, 7, 0, 0, 6],
        [8, 9, 2, 4, 1],
        [8, 3, 0, 1, 3, 2, 2, 6],
        [],
        [7]
    ]
    sequences = [torch.tensor(x) for x in sequences]
    result = pad_sequences(sequences, device, pad=p, eos=e)
    assert torch.all(result == torch.tensor([
        [5, 7, 1, 7, 0, 0, 6, e, p],
        [8, 9, 2, 4, 1, e, p, p, p],
        [8, 3, 0, 1, 3, 2, 2, 6, e],
        [e, p, p, p, p, p, p, p, p],
        [7, e, p, p, p, p, p, p, p]
    ]))
    result = pad_sequences(sequences, device, pad=p, bos=b, eos=e)
    assert torch.all(result == torch.tensor([
        [b, 5, 7, 1, 7, 0, 0, 6, e, p],
        [b, 8, 9, 2, 4, 1, e, p, p, p],
        [b, 8, 3, 0, 1, 3, 2, 2, 6, e],
        [b, e, p, p, p, p, p, p, p, p],
        [b, 7, e, p, p, p, p, p, p, p]
    ]))
    result, lengths = pad_sequences(sequences, device, pad=p, eos=e, return_lengths=True)
    assert torch.all(result == torch.tensor([
        [5, 7, 1, 7, 0, 0, 6, e, p],
        [8, 9, 2, 4, 1, e, p, p, p],
        [8, 3, 0, 1, 3, 2, 2, 6, e],
        [e, p, p, p, p, p, p, p, p],
        [7, e, p, p, p, p, p, p, p]
    ]))
    assert torch.all(lengths == torch.tensor([7, 5, 8, 0, 1]))
    result, lengths = pad_sequences(sequences, device, pad=p, bos=b, eos=e, return_lengths=True)
    assert torch.all(result == torch.tensor([
        [b, 5, 7, 1, 7, 0, 0, 6, e, p],
        [b, 8, 9, 2, 4, 1, e, p, p, p],
        [b, 8, 3, 0, 1, 3, 2, 2, 6, e],
        [b, e, p, p, p, p, p, p, p, p],
        [b, 7, e, p, p, p, p, p, p, p]
    ]))
    assert torch.all(lengths == torch.tensor([7, 5, 8, 0, 1]))
