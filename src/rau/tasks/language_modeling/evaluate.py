import argparse
import json
import math
import pathlib
import sys

from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.common.training_loop import evaluate
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import generate_batches, evaluate_batch

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
             'input.')
    parser.add_argument('--input',
        help='Name of a dataset in the training data directory that will be '
             'used as input. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input.')
    parser.add_argument('--input-file', type=pathlib.Path,
        help='A .prepared file to be used as input. This overrides '
             '--training-data and --input.')
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

    examples = load_prepared_data_file(input_file)
    saver = model_interface.construct_saver(args)
    batches = generate_batches(examples, args.batching_max_tokens)
    result = evaluate(saver.model, model_interface, batches, evaluate_batch)
    result['perplexity'] = math.exp(result['cross_entropy_per_token'])
    json.dump(result, sys.stdout, indent=2)
    print(file=sys.stdout)

if __name__ == '__main__':
    main()
