import argparse
import json
import math
import pathlib
import sys

from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.language_modeling.data import load_vocabularies
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import LanguageModelingTrainingLoop

def main():

    model_interface = LanguageModelingModelInterface(
        use_load=True,
        use_init=False,
        use_output=False,
        require_output=False
    )

    parser = argparse.ArgumentParser(
        description=
        'Evaluate a language model on a dataset. Output the results as JSON.'
    )
    parser.add_argument('--training-data', type=pathlib.Path,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--input',
        help='Name of a dataset in the training data directory that will be '
             'used as input. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input.')
    parser.add_argument('--input-file', type=pathlib.Path,
        help='A .prepared file to be used as input. This overrides '
             '--training-data and --input.')
    parser.add_argument('--vocabulary-file', type=pathlib.Path,
        help='A .vocab file containing the token vocabulary. This overrides '
             '--training-data.')
    parser.add_argument('--batching-max-tokens', type=int, required=True,
        help='The maximum number of tokens allowed per batch.')
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    args = parser.parse_args()

    if args.input_file is not None:
        input_file = args.input_file
    elif args.training_data is not None and args.input is not None:
        input_file = args.training_data / 'datasets' / args.input / 'main.prepared'
    else:
        parser.error('either --training-data and --input or --input-file is required')

    device = model_interface.get_device(args)
    examples = load_prepared_data_file(input_file)
    saver = model_interface.construct_saver(args)
    # TODO The vocabulary is only used here to figure out the padding index,
    # which can be inferred from the size of the model alone.
    vocabs = load_vocabularies(args, parser, model_interface)
    # Create a dummy training loop object so we can reuse the batching and
    # evaluation methods.
    # TODO Refactor this.
    training_loop = LanguageModelingTrainingLoop(
        show_progress=False,
        max_epochs=0,
        random_shuffling_seed=0,
        max_tokens_per_batch=args.batching_max_tokens,
        optimizer='SGD',
        initial_learning_rate=1.0,
        label_smoothing_factor=None,
        gradient_clipping_threshold=None,
        early_stopping_patience=1,
        learning_rate_patience=1,
        learning_rate_decay_factor=0.5,
        examples_per_checkpoint=1
    )
    batches = training_loop.generate_batches(examples, args.batching_max_tokens)
    result = training_loop.evaluate(saver.model, model_interface, vocabs, batches)
    result['perplexity'] = math.exp(result['cross_entropy_per_token'])
    json.dump(result, sys.stdout, indent=2)
    print(file=sys.stdout)

if __name__ == '__main__':
    main()
