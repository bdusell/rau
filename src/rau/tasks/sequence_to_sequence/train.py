import argparse
import logging
import sys

import humanfriendly

from rau.tools.torch.profile import get_current_memory
from rau.tasks.sequence_to_sequence.data import add_data_arguments, load_prepared_data
from rau.tasks.sequence_to_sequence.model import SequenceToSequenceModelInterface
from rau.tasks.sequence_to_sequence.training_loop import (
    add_training_loop_arguments,
    get_training_loop_kwargs,
    SequenceToSequenceTrainingLoop
)

def main():

    console_logger = logging.getLogger('main')
    console_logger.addHandler(logging.StreamHandler(sys.stdout))
    console_logger.setLevel(logging.INFO)
    console_logger.info(f'arguments: {sys.argv}')

    model_interface = SequenceToSequenceModelInterface()

    parser = argparse.ArgumentParser(
        description=
        'Train a sequence-to-sequence model.'
    )
    add_data_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    args = parser.parse_args()
    console_logger.info(f'parsed arguments: {args}')

    device = model_interface.get_device(args)
    console_logger.info(f'device: {device}')
    do_profile_memory = device.type == 'cuda'

    training_loop = SequenceToSequenceTrainingLoop(
        **get_training_loop_kwargs(parser, args)
    )

    data = load_prepared_data(args, parser)

    if do_profile_memory:
        memory_before = get_current_memory(device)
    saver = model_interface.construct_saver(
        args,
        source_vocabulary_size=len(data.source_vocab),
        target_input_vocabulary_size=len(data.target_input_vocab),
        target_output_vocabulary_size=len(data.target_output_vocab),
        tie_embeddings=data.vocab_is_shared
    )
    if model_interface.parameter_seed is not None:
        console_logger.info(f'parameter random seed: {model_interface.parameter_seed}')
    num_parameters = sum(p.numel() for p in saver.model.parameters())
    console_logger.info(f'number of parameters: {num_parameters}')
    if do_profile_memory:
        model_size_in_bytes = get_current_memory(device) - memory_before
        console_logger.info(f'model size: {humanfriendly.format_size(model_size_in_bytes)}')
    else:
        model_size_in_bytes = None

    with saver.logger() as event_logger:
        event_logger.log('model_info', dict(
            parameter_seed=model_interface.parameter_seed,
            size_in_bytes=model_size_in_bytes,
            num_parameters=num_parameters
        ))
        training_loop.run(
            saver,
            model_interface,
            data,
            console_logger,
            event_logger
        )

if __name__ == '__main__':
    main()
