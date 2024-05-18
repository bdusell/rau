import argparse
import logging
import sys

import humanfriendly

from rau.tools.torch.profile import get_current_memory
from rau.tasks.language_modeling.data import (
    add_data_arguments,
    load_vocabulary_data,
    load_prepared_data
)
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import (
    add_training_loop_arguments,
    get_training_loop_kwargs,
    LanguageModelingTrainingLoop
)

def main():

    # Configure logging to stdout.
    console_logger = logging.getLogger('main')
    console_logger.addHandler(logging.StreamHandler(sys.stdout))
    console_logger.setLevel(logging.INFO)
    console_logger.info(f'arguments: {sys.argv}')

    model_interface = LanguageModelingModelInterface()

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description=
        'Train a language model.'
    )
    add_data_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    args = parser.parse_args()
    console_logger.info(f'parsed arguments: {args}')

    # Are we training on CPU or GPU?
    device = model_interface.get_device(args)
    console_logger.info(f'device: {device}')
    do_profile_memory = device.type == 'cuda'

    # Configure the training loop.
    training_loop = LanguageModelingTrainingLoop(
        **get_training_loop_kwargs(parser, args)
    )

    # Load the tokens in the vocabulary. This determines the sizes of the
    # embedding and softmax layers in the model.
    vocabulary_data = load_vocabulary_data(args, parser)

    if do_profile_memory:
        memory_before = get_current_memory(device)
    # Construct the model.
    saver = model_interface.construct_saver(args, vocabulary_data)
    # Log some information about the model: parameter random seed, number of
    # parameters, GPU memory.
    if model_interface.parameter_seed is not None:
        console_logger.info(f'parameter random seed: {model_interface.parameter_seed}')
    num_parameters = sum(p.numel() for p in saver.model.parameters())
    console_logger.info(f'number of parameters: {num_parameters}')
    if do_profile_memory:
        model_size_in_bytes = get_current_memory(device) - memory_before
        console_logger.info(f'model size: {humanfriendly.format_size(model_size_in_bytes)}')
    else:
        model_size_in_bytes = None

    # Load the data.
    training_data, validation_data, vocabulary = \
        load_prepared_data(args, parser, vocabulary_data, model_interface)

    # Start logging events to disk.
    with saver.logger() as event_logger:
        event_logger.log('model_info', dict(
            parameter_seed=model_interface.parameter_seed,
            size_in_bytes=model_size_in_bytes,
            num_parameters=num_parameters
        ))
        # Run the training loop.
        training_loop.run(
            saver,
            model_interface,
            training_data,
            validation_data,
            vocabulary,
            console_logger,
            event_logger
        )

if __name__ == '__main__':
    main()
