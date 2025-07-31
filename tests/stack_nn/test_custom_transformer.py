import argparse
import random

import torch

from rau.tools.torch.init import smart_init, uniform_fallback
from rau.vocab import ToStringVocabularyBuilder, ToIntVocabularyBuilder
from rau.tasks.common.data_preparation import get_token_types
from rau.tasks.sequence_to_sequence.vocabulary import (
    SharedVocabularyData,
    get_vocabularies
)
from rau.tasks.sequence_to_sequence.model import SequenceToSequenceModelInterface
from rau.models.stack_nn.transformer.decoder import get_stack_transformer_decoder

def test_custom_matches_builtin():

    batch_size = 7

    generator = random.Random(123)
    def random_sequence():
        length = generator.randint(10, 20)
        return [
            chr(generator.randint(ord('a'), ord('z')))
            for i in range(length)
        ]

    batch = [(random_sequence(), random_sequence()) for i in range(batch_size)]
    source_token_types, source_has_unk = get_token_types((c for x, y in batch for c in x), '<unk>')
    target_token_types, target_has_unk = get_token_types((c for x, y in batch for c in y), '<unk>')
    tokens_in_target = sorted(target_token_types)
    tokens_only_in_source = sorted(source_token_types - target_token_types)
    vocabulary_data = SharedVocabularyData(
        tokens_in_target,
        tokens_only_in_source,
        allow_unk=False
    )
    source_vocab_to_int, _, target_output_vocab_to_int = get_vocabularies(
        vocabulary_data,
        ToIntVocabularyBuilder()
    )
    batch_as_ints = [
        (
            torch.tensor([source_vocab_to_int.to_int(c) for c in x]),
            torch.tensor([target_output_vocab_to_int.to_int(c) for c in y])
        )
        for x, y in batch
    ]
    source_vocab, target_input_vocab, target_output_vocab = get_vocabularies(
        SharedVocabularyData(
            tokens_in_target,
            tokens_only_in_source,
            allow_unk=False
        ),
        ToStringVocabularyBuilder()
    )

    def construct_model(use_builtin):
        parser = argparse.ArgumentParser()
        model_interface = SequenceToSequenceModelInterface(
            use_load=False,
            use_init=True,
            use_output=False,
            require_output=False
        )
        model_interface.add_arguments(parser)
        model_interface.add_forward_arguments(parser)
        if use_builtin:
            extra_args = [
                '--architecture', 'transformer',
                '--num-encoder-layers', '3',
                '--num-decoder-layers', '3'
            ]
        else:
            extra_args = [
                '--architecture', 'stack-transformer',
                '--stack-transformer-encoder-layers', '3',
                '--stack-transformer-decoder-layers', '3'
            ]
        args = parser.parse_args([
            *extra_args,
            '--d-model', '32',
            '--num-heads', '4',
            '--feedforward-size', '128',
            '--dropout', '0.2',
            '--init-scale', '0.1',
            '--device', 'cpu'
        ])
        saver = model_interface.construct_saver(args, vocabulary_data)
        return model_interface, saver
    builtin_model_interface, builtin_saver = construct_model(True)
    custom_model_interface, custom_saver = construct_model(False)
    def construct_generator():
        return torch.manual_seed(123)
    def init_model(model, generator):
        smart_init(model, generator, fallback=uniform_fallback(0.1))

    generator = construct_generator()
    init_model(builtin_saver.model, generator)

    generator = construct_generator()
    init_model(custom_saver.model, generator)

    device = torch.device('cpu')

    torch.manual_seed(42)
    model_input, target_output = builtin_model_interface.prepare_batch(batch_as_ints, device)
    builtin_logits = builtin_model_interface.get_logits(builtin_saver.model, model_input)

    torch.manual_seed(42)
    model_input, target_output = custom_model_interface.prepare_batch(batch_as_ints, device)
    custom_logits = custom_model_interface.get_logits(custom_saver.model, model_input)

    torch.testing.assert_close(builtin_logits, custom_logits)

def test_custom_decoder():
    input_vocabulary_size = 17
    output_vocabulary_size = 19
    num_heads = 5
    d_model = num_heads * 7 * 2
    batch_size = 3
    source_sequence_length = 11
    target_sequence_length = 13
    model = get_stack_transformer_decoder(
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        shared_embeddings=None,
        positional_encoding_cacher=None,
        layers=[('transformer', (1,)), ('superposition', (10,)), ('transformer', (1,))],
        d_model=d_model,
        num_heads=num_heads,
        feedforward_size=d_model * 4,
        dropout=0,
        use_padding=False
    )
    generator = torch.manual_seed(123)
    for p in model.parameters():
        p.data.uniform_(generator=generator)
    encoder_output_sequence = torch.rand((batch_size, source_sequence_length, d_model), generator=generator)
    decoder_input_sequence = torch.randint(input_vocabulary_size, (batch_size, target_sequence_length), generator=generator)
    source_is_padding_mask = torch.zeros((batch_size, source_sequence_length), dtype=torch.bool)
    python_generator = random.Random(123)
    for mask_vector in source_is_padding_mask:
        pos = python_generator.randrange(len(mask_vector))
        mask_vector[pos:] = True
    forward_output = model(
        decoder_input_sequence,
        tag_kwargs=dict(
            transformer=dict(
                encoder_sequence=encoder_output_sequence,
                encoder_is_padding_mask=source_is_padding_mask
            )
        ),
        include_first=False
    )
    assert forward_output.size() == (batch_size, target_sequence_length, output_vocabulary_size)
    state = model.initial_state(
        batch_size,
        tag_kwargs=dict(
            transformer=dict(
                encoder_sequence=encoder_output_sequence,
                encoder_is_padding_mask=source_is_padding_mask
            )
        )
    )
    for i in range(target_sequence_length):
        input_tensor = decoder_input_sequence[:, i]
        state = state.next(input_tensor)
        output = state.output()
        assert output.size() == (batch_size, output_vocabulary_size)
        torch.testing.assert_close(output, forward_output[:, i], atol=1e-4, rtol=1e-4)
