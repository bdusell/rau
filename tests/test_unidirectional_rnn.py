import pytest
import torch

from rau.models import SimpleRNN, LSTM

@pytest.mark.parametrize('ModelClass', [SimpleRNN, LSTM])
def test_forward_matches_iterative(ModelClass):
    batch_size = 5
    sequence_length = 13
    input_size = 7
    hidden_units = 17
    generator = torch.manual_seed(123)
    model = ModelClass(
        input_size=input_size,
        hidden_units=hidden_units,
        layers=3
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=True)
    assert forward_output.size() == (batch_size, sequence_length + 1, hidden_units)
    state = model.initial_state(batch_size)
    output = state.output()
    assert output.size() == (batch_size, hidden_units)
    torch.testing.assert_close(output, forward_output[:, 0])
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, hidden_units)
        torch.testing.assert_close(output, forward_output[:, i+1])

@pytest.mark.parametrize('ModelClass', [SimpleRNN, LSTM])
def test_empty_inputs(ModelClass):
    batch_size = 5
    sequence_length = 0
    input_size = 7
    hidden_units = 17
    generator = torch.manual_seed(123)
    model = ModelClass(
        input_size=input_size,
        hidden_units=hidden_units,
        layers=3
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, 0, hidden_units)
    forward_output = model(input_sequence, include_first=True)
    assert forward_output.size() == (batch_size, 1, hidden_units)
    torch.testing.assert_close(forward_output[:, 0], model.initial_state(batch_size).output())

@pytest.mark.parametrize('ModelClass', [SimpleRNN, LSTM])
def test_fastforward(ModelClass):
    batch_size = 5
    sequence_length = 13
    input_size = 7
    hidden_units = 17
    generator = torch.manual_seed(123)
    model = ModelClass(
        input_size=input_size,
        hidden_units=hidden_units,
        layers=3
    )
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, input_size), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, hidden_units)
    state = model.initial_state(batch_size)
    state = state.fastforward(input_sequence)
    output = state.output()
    assert output.size() == (batch_size, hidden_units)
    torch.testing.assert_close(output, forward_output[:, -1])
