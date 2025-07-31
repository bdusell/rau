import random

import torch

from rau.unidirectional import Unidirectional
from rau.models.stack_nn.transformer.cross_attention import (
    CrossAttention,
    CrossAttentionUnidirectional
)

def test_cross_attention_module():
    batch_size = 11
    num_heads = 5
    d_model = num_heads * 7
    source_sequence_length = 13
    target_sequence_length = 17
    generator = torch.manual_seed(123)
    module = CrossAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.0
    )
    for p in module.parameters():
        p.data.uniform_(generator=generator)
    encoder_output_sequence = torch.rand((batch_size, source_sequence_length, d_model), generator=generator)
    decoder_input_sequence = torch.rand((batch_size, target_sequence_length, d_model), generator=generator)
    decoder_output_sequence = module(decoder_input_sequence, encoder_output_sequence)
    assert decoder_output_sequence.size() == (batch_size, target_sequence_length, d_model)
    source_is_padding_mask = torch.zeros((batch_size, source_sequence_length), dtype=torch.bool)
    decoder_output_sequence = module(decoder_input_sequence, encoder_output_sequence, source_is_padding_mask)
    assert decoder_output_sequence.size() == (batch_size, target_sequence_length, d_model)
    python_generator = random.Random(123)
    for mask_vector in source_is_padding_mask:
        pos = python_generator.randrange(len(mask_vector))
        mask_vector[pos:] = True
    decoder_output_sequence = module(decoder_input_sequence, encoder_output_sequence, source_is_padding_mask)
    assert decoder_output_sequence.size() == (batch_size, target_sequence_length, d_model)

def test_cross_attention_unidirectional():
    batch_size = 11
    num_heads = 5
    d_model = num_heads * 7
    source_sequence_length = 13
    target_sequence_length = 17
    generator = torch.manual_seed(123)
    module = CrossAttentionUnidirectional(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.0
    )
    for p in module.parameters():
        p.data.uniform_(generator=generator)
    encoder_output_sequence = torch.rand((batch_size, source_sequence_length, d_model), generator=generator)
    decoder_input_sequence = torch.rand((batch_size, target_sequence_length, d_model), generator=generator)
    source_is_padding_mask = torch.zeros((batch_size, source_sequence_length), dtype=torch.bool)
    python_generator = random.Random(123)
    for mask_vector in source_is_padding_mask:
        pos = python_generator.randrange(len(mask_vector))
        mask_vector[pos:] = True
    forward_output = module(
        decoder_input_sequence,
        encoder_sequence=encoder_output_sequence,
        encoder_is_padding_mask=source_is_padding_mask,
        include_first=False
    )
    assert forward_output.size() == (batch_size, target_sequence_length, d_model)
    state = module.initial_state(
        batch_size,
        encoder_sequence=encoder_output_sequence,
        encoder_is_padding_mask=source_is_padding_mask
    )
    for i in range(target_sequence_length):
        input_tensor = decoder_input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i])

class StatefulUnidirectional(Unidirectional):

    def initial_state(self, batch_size):
        return self.State(self, None)

    class State(Unidirectional.State):

        def __init__(self, parent, input_tensor):
            super().__init__()
            self.parent = parent
            self.input_tensor = input_tensor

        def next(self, input_tensor):
            return self.parent.State(self.parent, input_tensor)

        def output(self):
            assert self.input_tensor is not None
            return self.input_tensor

def test_composed_cross_attention_unidirectional():
    batch_size = 11
    num_heads = 5
    d_model = num_heads * 7
    source_sequence_length = 13
    target_sequence_length = 17
    generator = torch.manual_seed(123)
    module = (
        StatefulUnidirectional() |
        CrossAttentionUnidirectional(
            d_model=d_model,
            num_heads=num_heads,
            dropout=0.0
        ).main()
    )
    for p in module.parameters():
        p.data.uniform_(generator=generator)
    encoder_output_sequence = torch.rand((batch_size, source_sequence_length, d_model), generator=generator)
    decoder_input_sequence = torch.rand((batch_size, target_sequence_length, d_model), generator=generator)
    source_is_padding_mask = torch.zeros((batch_size, source_sequence_length), dtype=torch.bool)
    python_generator = random.Random(123)
    for mask_vector in source_is_padding_mask:
        pos = python_generator.randrange(len(mask_vector))
        mask_vector[pos:] = True
    forward_output = module(
        decoder_input_sequence,
        encoder_sequence=encoder_output_sequence,
        encoder_is_padding_mask=source_is_padding_mask,
        include_first=False
    )
    assert forward_output.size() == (batch_size, target_sequence_length, d_model)
    state = module.initial_state(
        batch_size,
        encoder_sequence=encoder_output_sequence,
        encoder_is_padding_mask=source_is_padding_mask
    )
    for i in range(target_sequence_length):
        input_tensor = decoder_input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, d_model)
        torch.testing.assert_close(output, forward_output[:, i])
