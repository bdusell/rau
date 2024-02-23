import argparse
import logging
import sys

import humanfriendly

from rau.tools.torch.profile import get_current_memory
from rau.tasks.language_modeling.data import add_data_arguments, load_prepared_data
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import add_train_arguments, train

def main():

    logger = logging.getLogger('main')
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    logger.info(f'arguments: {sys.argv}')

    model_interface = LanguageModelingModelInterface()

    parser = argparse.ArgumentParser(
        description=
        'Train a language model.'
    )
    add_data_arguments(parser)
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_train_arguments(parser)
    args = parser.parse_args()
    logger.info(f'parsed arguments: {args}')

    device = model_interface.get_device(args)
    logger.info(f'device: {device}')
    do_profile_memory = device.type == 'cuda'

    data = load_prepared_data(args, parser)

    if do_profile_memory:
        memory_before = get_current_memory(device)
    saver = model_interface.construct_saver(
        args,
        input_vocab_size=len(data.input_vocab),
        output_vocab_size=len(data.output_vocab)
    )
    if model_interface.parameter_seed is not None:
        logger.info(f'parameter random seed: {model_interface.parameter_seed}')
    num_parameters = sum(p.numel() for p in saver.model.parameters())
    logger.info(f'number of parameters: {num_parameters}')
    if do_profile_memory:
        model_size_in_bytes = get_current_memory(device) - memory_before
        logger.info(f'model size: {humanfriendly.format_size(model_size_in_bytes)}')
    else:
        model_size_in_bytes = None

    with saver.logger() as events:
        events.log('model_info', dict(
            size_in_bytes=model_size_in_bytes,
            num_parameters=num_parameters
        ))
        train(
            parser,
            args,
            saver,
            data,
            model_interface,
            events,
            logger
        )

if __name__ == '__main__':
    main()
