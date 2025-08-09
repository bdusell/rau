import json
import math
import pathlib
import sys
from collections.abc import Iterable

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tasks.common.command import Command
from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.common.training_loop import evaluate, DictScoreAccumulator
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import generate_batches, evaluate_batch
from rau.tasks.language_modeling.batching import group_into_batches

class LanguageModelingEvaluateCommand(Command):

    def __init__(self):
        super().__init__()
        self.model_interface = LanguageModelingModelInterface(
            use_load=True,
            use_init=False,
            use_output=False,
            require_output=False
        )

    DESCRIPTION = (
        'Evaluate the likelihood of a dataset under a language model. Output '
        'the results as JSON.'
    )

    def add_arguments(self, parser):
        model_interface = self.model_interface
        parser.add_argument('--training-data', type=pathlib.Path,
            help='A directory containing training data. The file '
                 '<training-data>/datasets/<input>/main.prepared will be used '
                 'as input.')
        parser.add_argument('--input',
            dest='prompt_and_input',
            action='append',
            help='Name of a dataset in the training data directory that will '
                 'be used as input. The file '
                 '<training-data>/datasets/<input>/main.prepared will be used '
                 'as input. This can be given multiple times to process '
                 'multiple datasets.')
        parser.add_argument('--prompt-and-input',
            dest='prompt_and_input',
            nargs=2,
            action='append',
            help='Names of two, parallel datasets in the training data '
                 'directory. The first dataset will be used as prompts to the '
                 'language model. The conditional cross-entropy of the second '
                 'dataset will be computed, conditioned on the prompts from '
                 'the first dataset.')
        parser.add_argument('--output', type=pathlib.Path,
            help='Directory where results will be saved as JSON files. There '
                 'will be one file per input dataset.')
        parser.add_argument('--batching-max-tokens', type=int, default=2048,
            help='The maximum number of tokens allowed per batch.')
        model_interface.add_arguments(parser)
        model_interface.add_forward_arguments(parser)

    def run(self, parser, args):
        model_interface = self.model_interface
        if not args.prompt_and_input:
            parser.error(
                'no datasets were provided; provide one or more of --input or '
                '--prompt-and-input'
            )
        saver = model_interface.construct_saver(args)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        for arg in args.prompt_and_input:
            if isinstance(arg, list):
                prompt_dataset, input_dataset = arg
            else:
                prompt_dataset = None
                input_dataset = arg
            if prompt_dataset is not None:
                prompt_file = get_dataset_file_name(args.training_data, prompt_dataset)
                prompts = load_prepared_data_file(prompt_file)
            input_file = get_dataset_file_name(args.training_data, input_dataset)
            examples = load_prepared_data_file(input_file)
            if prompt_dataset is None:
                batches = generate_batches(examples, args.batching_max_tokens)
                result = evaluate(saver.model, model_interface, batches, evaluate_batch)
            else:
                batches = generate_prompt_batches(prompts, examples, args.batching_max_tokens)
                result = evaluate_conditional_cross_entropy(saver.model, model_interface, batches)
            if args.output is None:
                print_result(result, sys.stdout)
            else:
                output_file = args.output / f'{input_dataset}.json'
                print(f'writing {output_file}')
                with output_file.open('w') as fout:
                    print_result(result, fout)

def get_dataset_file_name(training_data, dataset):
    return training_data / 'datasets' / dataset / 'main.prepared'

def print_result(result, fout):
    json.dump(result, fout)
    print(file=fout)

def generate_prompt_batches(
    prompts: Iterable[torch.Tensor],
    examples: Iterable[torch.Tensor],
    max_tokens: int
) -> Iterable[list[tuple[int, torch.Tensor]]]:
    return group_into_batches(
        [
            (len(prompt), torch.concat([prompt, example], dim=0))
            for prompt, example in zip(prompts, examples, strict=True)
        ],
        is_small_enough=lambda b, n: b * n <= max_tokens,
        get_length=lambda x: len(x[1])
    )

def evaluate_conditional_cross_entropy(
    model: torch.nn.Module,
    model_interface: ModelInterface,
    batches: Iterable[list[tuple[int, torch.Tensor]]]
) -> dict[str, float]:
    pad_index = model_interface.output_padding_index
    device = model_interface.get_device(None)
    accumulator = DictScoreAccumulator()
    model.eval()
    with torch.inference_mode():
        for batch in batches:
            input_tensor, output_tensor = model_interface.prepare_batch([x[1] for x in batch], device)
            # Make sure the in-place modifications to the output tensor don't
            # affect the input tensor.
            input_tensor = input_tensor.clone()
            # In the output tensor only, mask out the prompt with padding
            # tokens so that the prompt won't contribute to the total
            # cross-entropy or the total number of tokens.
            for (prompt_length, _), output_tensor_element in zip(batch, output_tensor, strict=True):
                output_tensor_element[:prompt_length] = pad_index
            batch_score_dict = evaluate_batch(
                model,
                model_interface,
                (input_tensor, output_tensor)
            )
            accumulator.update(batch_score_dict)
    return accumulator.get_value()

if __name__ == '__main__':
    LanguageModelingEvaluateCommand().main()
