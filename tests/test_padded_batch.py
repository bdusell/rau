import argparse
import random

import torch

from rau.tasks.language_modeling.vocabulary import VocabularyData
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import (
    LanguageModelingTrainingLoop,
    add_training_loop_arguments,
    get_training_loop_kwargs,
)

def test_single_matches_batched():
    argv = [
        '--device', 'cpu',
        '--parameter-seed', '123',
        '--architecture', 'transformer',
        '--num-layers', '5',
        '--d-model', '32',
        '--num-heads', '4',
        '--feedforward-size', '64',
        '--dropout', '0',
        '--init-scale', '0.1',
        '--max-epochs', '1',
        '--random-shuffling-seed', '1',
        '--max-tokens-per-batch', '1',
        '--optimizer', 'SGD',
        '--initial-learning-rate', '0.01',
        '--label-smoothing-factor', '0',
        '--gradient-clipping-threshold', '5',
        '--early-stopping-patience', '10',
        '--learning-rate-patience', '5',
        '--learning-rate-decay-factor', '0.5',
        '--examples-per-checkpoint', '1'
    ]

    model_interface = LanguageModelingModelInterface(use_load=False, use_output=False)
    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    args = parser.parse_args(argv)

    device = model_interface.get_device(args)
    training_loop = LanguageModelingTrainingLoop(**get_training_loop_kwargs(parser, args))
    vocabulary_data = VocabularyData(
        tokens=['0', '1'],
        allow_unk=False
    )
    saver = model_interface.construct_saver(args, vocabulary_data)

    generator = random.Random(123)
    lengths = [0, 1, 3, 7, 10, 13, 23]
    vocab_size = len(vocabulary_data.tokens)
    batch = [
        torch.tensor(
            [generator.randrange(vocab_size) for _ in range(length)],
            dtype=torch.int64,
            device=device
        )
        for length in lengths
    ]
    generator.shuffle(batch)
    num_examples = len(batch)

    # Run forward and backward passes on the whole batch.
    saver.model.zero_grad()
    _, batched_loss, _, _, _ = training_loop.get_prepared_batch_and_loss(
        saver,
        model_interface,
        batch
    )
    batched_loss.backward()
    batched_grads = { name : param.grad.clone() for name, param in saver.model.named_parameters() }

    # Run forward and backward passes on the individual examples and accumulate
    # their gradient.
    saver.model.zero_grad()
    for example in batch:
        _, example_loss, _, _, _ = training_loop.get_prepared_batch_and_loss(
            saver,
            model_interface,
            [example]
        )
        example_loss = example_loss / num_examples
        example_loss.backward()
    single_grads = { name : param.grad.clone() for name, param in saver.model.named_parameters() }

    names = batched_grads.keys()
    assert single_grads.keys() == names
    for name in names:
        batched_grad = batched_grads[name]
        single_grad = single_grads[name]
        torch.testing.assert_close(
            batched_grad,
            single_grad,
            msg=f'gradients not equal for parameter {name}'
        )
