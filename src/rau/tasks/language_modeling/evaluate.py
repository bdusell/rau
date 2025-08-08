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
            help='Name of a dataset in the training data directory that will '
                 'be used as input. The file '
                 '<training-data>/datasets/<input>/main.prepared will be used '
                 'as input.')
        parser.add_argument('--input-file', type=pathlib.Path,
            help='A .prepared file to be used as input. This overrides '
                 '--training-data and --input.')
        parser.add_argument('--prompt-dataset',
            help='Optional name of a dataset in the training data directory '
                 'that will be used as prompts to the language model before '
                 'measuring probabilities, so that what is computed will '
                 'consist of conditional probabilities.')
        parser.add_argument('--batching-max-tokens', type=int, required=True,
            help='The maximum number of tokens allowed per batch.')
        model_interface.add_arguments(parser)
        model_interface.add_forward_arguments(parser)

    def run(self, parser, args):
        model_interface = self.model_interface

        if args.input_file is not None:
            input_file = args.input_file
        elif args.training_data is not None and args.input is not None:
            input_file = args.training_data / 'datasets' / args.input / 'main.prepared'
        else:
            parser.error('either --training-data and --input or --input-file is required')

        if args.prompt_dataset is not None:
            prompt_file = args.training_data / 'datasets' / args.prompt_dataset / 'main.prepared'
        else:
            prompt_file = None

        examples = load_prepared_data_file(input_file)
        saver = model_interface.construct_saver(args)

        if prompt_file is not None:
            prompts = load_prepared_data_file(prompt_file)
            result = evaluate_conditional_cross_entropy(
                saver.model,
                model_interface,
                prompts,
                examples,
                args.batching_max_tokens
            )
        else:
            batches = generate_batches(examples, args.batching_max_tokens)
            result = evaluate(saver.model, model_interface, batches, evaluate_batch)
        json.dump(result, sys.stdout, indent=2)
        print(file=sys.stdout)

def evaluate_conditional_cross_entropy(
    model: torch.nn.Module,
    model_interface: ModelInterface,
    prompts: Iterable[torch.Tensor],
    examples: Iterable[torch.Tensor],
    max_tokens: int
) -> dict[str, float]:
    batches = generate_prompt_batches(prompts, examples, max_tokens)
    return evaluate_conditional_cross_entropy_on_batches(model, model_interface, batches)

def generate_prompt_batches(
    prompts: Iterable[torch.Tensor],
    examples: Iterable[torch.Tensor],
    max_tokens: int
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    return group_into_batches(
        [
            (len(prompt), torch.concat([prompt, example], dim=0))
            for prompt, example in zip(prompts, examples, strict=True)
        ],
        is_small_enough=lambda b, n: b * n <= max_tokens,
        get_length=lambda x: len(x[1])
    )

def evaluate_conditional_cross_entropy_on_batches(
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
