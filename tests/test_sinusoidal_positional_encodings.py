import pytest
import torch

from rau.models.transformer.positional_encodings import sinusoidal_positional_encodings

@pytest.mark.parametrize('d_model', [128])
def test_dmodel_sizes(d_model) -> None:
    sequence_length = 50
    device = torch.device('cpu')
    r = sinusoidal_positional_encodings(sequence_length, d_model, device)
    assert r.size() == (sequence_length, d_model)

def test_consistency() -> None:
    device = torch.device('cpu')
    d_model = 128
    a_length = 50
    a = sinusoidal_positional_encodings(a_length, d_model, device)
    assert a.size() == (a_length, d_model)
    b_length = 100
    b = sinusoidal_positional_encodings(b_length, d_model, device)
    assert b.size() == (b_length, d_model)
    torch.testing.assert_close(b[:a_length], a)
