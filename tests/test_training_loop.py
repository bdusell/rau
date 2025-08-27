import argparse
import logging
import random

import pytest
import torch

from rau.tools.torch.saver import read_saver
from rau.tasks.common.command import get_logger
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import VocabularyData
from rau.tasks.language_modeling.data import VocabularyContainer
from rau.tasks.common.training_loop import SimulatedTrainingLoopError
from rau.tasks.language_modeling.training_loop import (
    LanguageModelingTrainingLoop,
    add_training_loop_arguments
)
from old_training_loop.lm_training_loop import (
    LanguageModelingTrainingLoop as OldLanguageModelingTrainingLoop,
    add_training_loop_arguments as old_add_training_loop_arguments,
    get_training_loop_kwargs as old_get_training_loop_kwargs
)

def generate_dataset(
    num_examples: int,
    max_length: int,
    generator: random.Random
) -> list[list[int]]:
    return [
        torch.tensor([1] + [0] * generator.randint(0, max_length-1))
        for _ in range(num_examples)
    ]

def test_new_loop_matches_old(tmp_path):
    console_logger = get_logger()
    data_generator = random.Random(123)
    training_data = generate_dataset(
        num_examples=1000,
        max_length=20,
        generator=data_generator
    )
    validation_data = generate_dataset(
        num_examples=100,
        max_length=20,
        generator=data_generator
    )
    vocabulary_data = VocabularyData(
        tokens=['0', '1'],
        allow_unk=False
    )
    common_argv = [
        '--device', 'cpu',
        '--parameter-seed', '123',
        '--architecture', 'transformer',
        '--num-layers', '2',
        '--d-model', '32',
        '--num-heads', '4',
        '--feedforward-size', '32',
        '--dropout', '0',
        '--init-scale', '0.01',
        '--max-epochs', '100',
        '--random-shuffling-seed', '123',
        '--max-tokens-per-batch', '256',
        '--optimizer', 'Adam',
        '--initial-learning-rate', '0.01',
        '--gradient-clipping-threshold', '5',
        '--early-stopping-patience', '4',
        '--learning-rate-patience', '2',
        '--learning-rate-decay-factor', '0.5',
        '--examples-per-checkpoint', '500'
    ]
    reference_model_path = tmp_path / 'reference'
    reference_argv = common_argv + ['--output', str(reference_model_path)]
    reference_model_interface = LanguageModelingModelInterface()
    reference_parser = argparse.ArgumentParser()
    reference_model_interface.add_arguments(reference_parser)
    reference_model_interface.add_forward_arguments(reference_parser)
    old_add_training_loop_arguments(reference_parser)
    reference_args = reference_parser.parse_args(reference_argv)
    reference_training_loop = OldLanguageModelingTrainingLoop(
        **old_get_training_loop_kwargs(reference_parser, reference_args)
    )
    reference_saver = reference_model_interface.construct_saver(
        reference_args,
        vocabulary_data
    )
    reference_vocabulary = VocabularyContainer(
        *reference_model_interface.get_vocabularies(
            vocabulary_data,
            None
        )
    )
    with reference_saver.logger() as reference_event_logger:
        reference_training_loop.run(
            saver=reference_saver,
            model_interface=reference_model_interface,
            training_data=training_data,
            validation_data=validation_data,
            vocabulary=reference_vocabulary,
            console_logger=console_logger,
            event_logger=reference_event_logger
        )
    reference_model = read_saver(
        reference_model_interface.construct_model,
        reference_model_path
    ).model
    model_path = tmp_path / 'model'
    argv = common_argv + ['--output', str(model_path)]
    model_interface = LanguageModelingModelInterface()
    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    args = parser.parse_args(argv)
    saver = model_interface.construct_saver(args, vocabulary_data)
    training_loop_state = LanguageModelingTrainingLoop.get_state(parser, args, saver.model)
    vocabulary = VocabularyContainer(
        *model_interface.get_vocabularies(
            vocabulary_data,
            None
        )
    )
    with saver.logger() as event_logger:
        training_loop_state.run(
            saver=saver,
            model_interface=model_interface,
            training_data=training_data,
            validation_data=validation_data,
            vocabulary=vocabulary,
            console_logger=console_logger,
            event_logger=event_logger,
            show_progress=not args.no_progress
        )
    reference_params = dict(reference_saver.model.named_parameters())
    params = dict(saver.model.named_parameters())
    assert params.keys() == reference_params.keys()
    for name, reference_param in reference_params.items():
        param = params[name]
        torch.testing.assert_close(param, reference_param, msg=lambda msg: f'mismatch in parameter {name}: {msg}')
