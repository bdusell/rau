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
    a_length = 4
    a_d_model = 4
    a = sinusoidal_positional_encodings(a_length, a_d_model, device)
    print(a)
    b_length = 10
    b_d_model = 4
    b = sinusoidal_positional_encodings(b_length, b_d_model, device)
    print(b)
    torch.testing.assert_close(b[:a_length,:a_d_model], a)
