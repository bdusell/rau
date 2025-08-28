import json
import math
import pathlib
import sys
from collections.abc import Iterable

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tasks.common.command import Command
from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.common.training_loop import MicroAveragedScoreAccumulator
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.training_loop import generate_batches, evaluate_batch
from rau.tasks.language_modeling.batching import group_into_batches

class LanguageModelingEvaluateCommand(Command):

    def __init__(self):
        super().__init__()
        self.model_interface = LanguageModelingModelInterface(use_load=True)

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
        parser.add_argument('--granularity',
            choices=[
                'dataset',
                'sequence',
                'position',
                'vocabulary',
                'logits'
            ],
            default='dataset',
            help='The level of specificity of the output. Options: '
                 'dataset (default): Output a single, micro-averaged '
                 'cross-entropy score for each dataset, normalized by sequence '
                 'length plus one. '
                 'sequence: Output an unnormalized cross-entropy score for '
                 'every sequence. '
                 'position: Output a cross-entropy score for every position in '
                 'every sequence; this is the negative log probability of the '
                 'correct token at each position. '
                 'vocabulary: Output the negative log probability of every '
                 'token in the vocabulary at every position. '
                 'logits: Output the raw logits for every position.')
        parser.add_argument('--output', type=pathlib.Path, required=True,
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
        match args.granularity:
            case 'dataset':
                process_sequences = process_sequences_dataset
                write_result = write_txt
                extension = 'txt'
            case 'position':
                process_sequences = process_sequences_position
                write_result = write_pt
                extension = 'pt'
            case 'vocabulary':
                process_sequences = process_sequences_vocabulary
                write_result = write_pt
                extension = 'pt'
            case 'logits':
                process_sequences = process_sequences_logits
                write_result = write_pt
                extension = 'pt'
            case _:
                raise NotImplementedError
        saver = model_interface.construct_saver(args)
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
        model = saver.model
        model.eval()
        with torch.inference_mode():
            for arg in args.prompt_and_input:
                if isinstance(arg, list):
                    prompt_dataset, input_dataset = arg
                else:
                    prompt_dataset = None
                    input_dataset = arg
                prompts = load_file(args.training_data, prompt_dataset) if prompt_dataset is not None else None
                sequences = load_file(args.training_data, input_dataset)
                result = process_sequences(
                    model,
                    model_interface,
                    prompts,
                    sequences,
                    args.batching_max_tokens
                )
                output_file = args.output / f'{input_dataset}.{extension}'
                print(f'writing {output_file}')
                write_result(result, output_file)

def get_dataset_file_name(training_data, dataset):
    return training_data / 'datasets' / dataset / 'main.prepared'

def load_file(training_data, dataset):
    return load_prepared_data_file(get_dataset_file_name(training_data, dataset))

def generate_batches(
    prompts: Iterable[torch.Tensor] | None,
    sequences: Iterable[torch.Tensor],
    max_tokens: int,
    include_indexes: bool = False
):
    if prompts is None:
        items = sequences
        get_sequence = lambda x: x
    else:
        items = (
            (len(prompt), torch.concat([prompt, sequence], dim=0))
            for prompt, sequence in zip(prompts, sequences, strict=True)
        )
        get_sequence = lambda x: x[1]
    if include_indexes:
        items = enumerate(items)
        old_get_sequence = get_sequence
        get_sequence = lambda x: old_get_sequence(x[1])
    batches = group_into_batches(
        list(items),
        is_small_enough=lambda b, n: b * n <= max_tokens,
        get_length=lambda x: len(get_sequence(x))
    )
    if include_indexes:
        return ((batch, [get_sequence(x) for x in batch]) for batch in batches)
    else:
        if prompts is None:
            return ((None, batch) for batch in batches)
        else:
            return ((batch, [get_sequence(x) for x in batch]) for batch in batches)

def process_sequences_dataset(
    model: torch.nn.Module,
    model_interface: LanguageModelingModelInterface,
    prompts: list[torch.Tensor] | None,
    sequences: list[torch.Tensor],
    max_tokens: int
) -> float:
    batches = generate_batches(prompts, sequences, max_tokens)
    device = model_interface.get_device(None)
    pad_index = model_interface.output_padding_index
    accumulator = MicroAveragedScoreAccumulator()
    for prompts, sequences in batches:
        input_tensor, output_tensor = model_interface.prepare_batch(sequences, device)
        logits = model_interface.get_logits(model, input_tensor)
        if prompts is not None:
            # In the output tensor only, mask out the prompt with padding
            # tokens so that the prompt won't contribute to the total
            # cross-entropy or the total number of tokens.
            # Since input_tensor has already been used to compute the logits,
            # it doesn't matter if modifying output_tensor also modifies
            # input_tensor.
            for (prompt_length, _), output_tensor_element in zip(prompts, output_tensor, strict=True):
                output_tensor_element[:prompt_length] = pad_index
        cross_entropy = torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1),
            output_tensor,
            ignore_index=pad_index,
            reduction='sum'
        )
        num_symbols = torch.sum(output_tensor != pad_index)
        accumulator.update(cross_entropy.item(), num_symbols.item())
    return accumulator.get_value()

def process_sequences_position(
    model: torch.nn.Module,
    model_interface: LanguageModelingModelInterface,
    prompts: list[torch.Tensor] | None,
    sequences: list[torch.Tensor],
    max_tokens: int
) -> list[torch.Tensor]:
    device = model_interface.get_device(None)
    pad_index = model_interface.output_padding_index
    def get_outputs(sequences):
        input_tensor, output_tensor = model_interface.prepare_batch(sequences, device)
        logits = model_interface.get_logits(model, input_tensor)
        return torch.nn.functional.cross_entropy(
            logits.permute(0, 2, 1),
            output_tensor,
            ignore_index=pad_index,
            reduction='none'
        )
    return process_sequences_token_level(
        prompts,
        sequences,
        max_tokens,
        get_outputs
    )

def process_sequences_vocabulary(
    model: torch.nn.Module,
    model_interface: LanguageModelingModelInterface,
    prompts: list[torch.Tensor] | None,
    sequences: list[torch.Tensor],
    max_tokens: int
) -> list[torch.Tensor]:
    device = model_interface.get_device(None)
    def get_outputs(sequences):
        input_tensor = model_interface.prepare_input_batch(sequences, device)
        logits = model_interface.get_logits(model, input_tensor)
        return -torch.nn.functional.log_softmax(logits, dim=2)
    return process_sequences_token_level(
        prompts,
        sequences,
        max_tokens,
        get_outputs
    )

def process_sequences_logits(
    model: torch.nn.Module,
    model_interface: LanguageModelingModelInterface,
    prompts: list[torch.Tensor] | None,
    sequences: list[torch.Tensor],
    max_tokens: int
) -> list[torch.Tensor]:
    device = model_interface.get_device(None)
    def get_outputs(sequences):
        input_tensor = model_interface.prepare_input_batch(sequences, device)
        return model_interface.get_logits(model, input_tensor)
    return process_sequences_token_level(
        prompts,
        sequences,
        max_tokens,
        get_outputs
    )

def process_sequences_token_level(
    prompts: list[torch.Tensor] | None,
    sequences: list[torch.Tensor],
    max_tokens: int,
    get_outputs
) -> list[torch.Tensor]:
    result = [None] * len(sequences)
    has_prompts = prompts is not None
    batches = generate_batches(prompts, sequences, max_tokens, include_indexes=True)
    for info, sequences in batches:
        outputs = get_outputs(sequences)
        for j, ((i, _), sequence, sequence_outputs) in enumerate(zip(info, sequences, outputs)):
            if has_prompts:
                start_pos = info[j][1][0]
            else:
                start_pos = 0
            result[i] = sequence_outputs[start_pos:len(sequence)+1]
    return result

def write_txt(result, output_file):
    output_file.write_text(str(result))

def write_pt(result, output_file):
    torch.save(result, output_file)

if __name__ == '__main__':
    LanguageModelingEvaluateCommand().main()
