import torch

from rau.models.transformer.unidirectional_encoder import UnidirectionalTransformerEncoderLayers

def test_layers_forward_matches_iterative():
    batch_size = 5
    sequence_length = 13
    d_model = 32
    generator = torch.manual_seed(123)
    model = UnidirectionalTransformerEncoderLayers(
        num_layers=3,
        d_model=d_model,
        num_heads=8,
        feedforward_size=64,
        dropout=0,
        use_final_layer_norm=True
    )
    for param in model.parameters():
        param.data.uniform_(generator=generator)
    input_sequence = torch.rand((batch_size, sequence_length, d_model), generator=generator)
    forward_output = model(input_sequence, include_first=False)
    assert forward_output.size() == (batch_size, sequence_length, d_model)
    state = model.initial_state(batch_size)
    for i in range(sequence_length):
        input_tensor = input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i])
    state = model.initial_state(batch_size)
    state = state.fastforward(input_sequence[:, :3])
    result = state.forward(
        input_sequence[:, 3:7],
        include_first=False,
        return_state=True
    )
    torch.testing.assert_close(
        result.output,
        forward_output[:, 3:7]
    )
    state = result.state
    state_forward_output = state.forward(
        input_sequence[:, 7:],
        include_first=True
    )
    torch.testing.assert_close(
        state_forward_output,
        forward_output[:, 6:]
    )
