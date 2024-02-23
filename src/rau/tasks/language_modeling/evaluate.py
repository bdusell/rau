import argparse
import json
import math
import pathlib
import sys

from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.language_modeling.data import (
    add_vocabulary_arguments,
    load_vocabularies
)
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import generate_batches, evaluate

def main():

    model_interface = LanguageModelingModelInterface(
        use_load=True,
        use_init=False,
        use_output=False,
        require_output=False
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=pathlib.Path, required=True,
        help='A .prepared file to be evaluated with the language model.')
    parser.add_argument('--batching-max-tokens', type=int, required=True,
        help='The maximum number of tokens allowed per batch.')
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_vocabulary_arguments(parser)
    args = parser.parse_args()

    device = model_interface.get_device(args)
    sources = load_prepared_data_file(args.input)
    vocabs = load_vocabularies(args, parser)
    saver = model_interface.construct_saver(args)
    batches = generate_batches(sources, args.batching_max_tokens)
    result = evaluate(saver.model, batches, vocabs, model_interface, device)
    result['perplexity'] = math.exp(result['cross_entropy_per_token'])
    json.dump(result, sys.stdout, indent=2)
    print(file=sys.stdout)

if __name__ == '__main__':
    main()
