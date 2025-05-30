from pytest import approx
import torch

from rau.tools.torch.layer import Layer, MultiLayer

def test_layer():
    batch_size = 5
    input_size = 7
    output_size = 11
    x = torch.ones(batch_size, input_size)
    layer = Layer(input_size, output_size, torch.nn.Softmax(dim=1))
    y = layer(x)
    assert y.size() == (batch_size, output_size)
    for a in y:
        a_sum = a.sum().item()
        assert a_sum == approx(1)
        for b in y:
            torch.testing.assert_close(a, b)

def test_multi_layer():
    batch_size = 5
    input_size = 7
    output_size = 11
    n = 13
    x = torch.ones(batch_size, input_size)
    layer = MultiLayer(input_size, output_size, n, torch.nn.Softmax(dim=2))
    y = layer(x)
    assert y.size() == (batch_size, n, output_size)
    for a in y:
        for aa in a:
            aa_sum = aa.sum().item()
            assert aa_sum == approx(1)
