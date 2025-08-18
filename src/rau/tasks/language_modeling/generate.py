import pathlib
import sys

import torch

from rau.tasks.common.command import Command
from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import load_vocabulary_data_from_file
from rau.tasks.language_modeling.batching import group_into_same_length_batches
from rau.generation import sample, decode_greedily, beam_search

class LanguageModelingGenerateCommand(Command):

    def __init__(self):
        super().__init__()
        self.model_interface = LanguageModelingModelInterface(
            use_load=True,
            use_init=False,
            use_output=False,
            require_output=False
        )

    DESCRIPTION = (
        'Generate sequences of tokens from a language model.'
    )

    def add_arguments(self, parser):
        model_interface = self.model_interface
        parser.add_argument('--training-data', type=pathlib.Path,
            help='A directory containing training data. The file '
                 '<training-data>/main.vocab will be used as the vocabulary.')
        parser.add_argument('--vocabulary-file', type=pathlib.Path,
            help='A .vocab file to be used as the vocabulary. This overrides '
                 '--training-data.')
        parser.add_argument('--prompt-datasets', nargs='+',
            help='Optional names of datasets in the training data directory '
                 'that will be used to prompt the language model. The file '
                 '<training-data>/datasets/<prompt-dataset>/main.prepared will '
                 'be used as input. Outputs will be generated for each example '
                 'in each dataset.')
        parser.add_argument('--output', type=pathlib.Path,
            help='When using --prompt-datasets, this is a directory where '
                 'output files will be written. A separate file will be '
                 'written for each prompt dataset. If random mode is used, its '
                 'name will be <output>/<prompt-dataset>.tsv, where multiple '
                 'samples for the same prompt will be separated by tab '
                 'characters. Otherwise, the file name will be '
                 '<output>/<prompt-dataset>.tok.')
        parser.add_argument('--mode',
            choices=['random', 'greedy', 'beam-search'],
            default='random',
            help='Which generation algorithm to use. Choices: '
                 'random: Random sampling or ancestral sampling; '
                 'greedy: Greedy decoding; '
                 'beam-search: Beam search with length normalization.')
        parser.add_argument('--max-length', type=int,
            help='Optional maximum number of tokens generated per sequence. If '
                 'this is not given, generation may run arbitrarily long.')
        parser.add_argument('--num-samples', type=int, default=1,
            help='Number of samples to generate when using random mode.')
        parser.add_argument('--random-seed', type=int,
            help='Optional random seed for random mode.')
        parser.add_argument('--beam-size', type=int,
            help='Beam size for beam search.')
        parser.add_argument('--batching-max-tokens', type=int, default=2048,
            help='The maximum number of tokens allowed per batch when using '
                 'prompts.')
        model_interface.add_arguments(parser)
        model_interface.add_forward_arguments(parser)

    def run(self, parser, args):
        model_interface = self.model_interface

        if args.vocabulary_file is not None:
            vocab_file = args.vocabulary_file
        elif args.training_data is not None:
            vocab_file = args.training_data / 'main.vocab'
        else:
            parser.error('either --training-data or --vocabulary-file is required')
        if args.prompt_datasets and args.output is None:
            parser.error('--output is required when --prompt-datasets is used')
        if args.mode == 'beam-search' and args.beam_size is None:
            parser.error('--beam-size is required for beam search')

        saver = model_interface.construct_saver(args)
        model = saver.model
        device = model_interface.get_device(args)
        eos_index = model_interface.eos_index
        _, vocab = model_interface.get_vocabularies(
            load_vocabulary_data_from_file(vocab_file),
            include_embedding_vocab=False
        )

        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)

        match args.mode:
            case 'random':
                if args.random_seed is not None:
                    generator = torch.Generator(device=device).manual_seed(args.random_seed)
                else:
                    generator = None
                def generate_output(initial_state):
                    return sample(
                        initial_state=initial_state,
                        eos_symbol=eos_index,
                        max_length=args.max_length,
                        num_samples=args.num_samples,
                        generator=generator
                    )
            case 'greedy':
                def generate_output(initial_state):
                    return decode_greedily(
                        initial_state=initial_state,
                        eos_symbol=eos_index,
                        max_length=args.max_length
                    )
            case 'beam-search':
                def generate_output(initial_state):
                    return beam_search(
                        initial_state=initial_state,
                        beam_size=args.beam_size,
                        eos_symbol=eos_index,
                        max_length=args.max_length,
                        device=device
                    )
        has_multiple_outputs = args.mode == 'random'

        model.eval()
        with torch.inference_mode():
            if args.prompt_datasets is not None:
                # Mix all datasets together for more batching opportunities. We
                # will unmix them all later.
                batch_elements = []
                result = {}
                for prompt_dataset in args.prompt_datasets:
                    prompts = load_prepared_data_file(args.training_data / 'datasets' / prompt_dataset / 'main.prepared')
                    batch_elements.extend((prompt_dataset, i, prompt) for i, prompt in enumerate(prompts))
                    result[prompt_dataset] = [None] * len(prompts)
                batches = group_into_same_length_batches(
                    batch_elements,
                    is_small_enough=lambda b, n: b * n <= args.batching_max_tokens,
                    get_length=lambda x: len(x[2])
                )
                # Actually run the model on the batches.
                for batch in batches:
                    # TODO Make this more efficient by adjusting the
                    # ModelInterface so that it can prepare a prompt and add
                    # BOS up front.
                    prompt_tensor = torch.stack([prompt.to(device) for _, _, prompt in batch], dim=0)
                    state = model_interface.get_initial_state(
                        saver.model,
                        batch_size=len(batch),
                        device=device
                    )
                    state = state.fastforward(prompt_tensor)
                    outputs = generate_output(state)
                    for (prompt_dataset, i, _), output in zip(batch, outputs):
                        result[prompt_dataset][i] = output
                # Write out the results in the original order.
                extension = 'tsv' if has_multiple_outputs else 'tok'
                print_output = print_multiple_token_sequences if has_multiple_outputs else print_token_sequence
                for prompt_dataset in args.prompt_datasets:
                    output_file_name = args.output / f'{prompt_dataset}.{extension}'
                    print(f'writing {output_file_name}')
                    with output_file_name.open('w') as fout:
                        for output in result[prompt_dataset]:
                            print_output(vocab, output, fout)
            else:
                initial_state = model_interface.get_initial_state(
                    saver.model,
                    batch_size=1,
                    device=device
                )
                if has_multiple_outputs:
                    def generate_token_sequences():
                        for outputs in generate_output(initial_state):
                            for output in outputs:
                                yield output
                else:
                    def generate_token_sequences():
                        return generate_output(initial_state)
                for output in generate_token_sequences():
                    print_token_sequence(vocab, output, sys.stdout)

def token_sequence_to_str(vocabulary, sequence):
    return ' '.join(map(vocabulary.to_string, sequence))

def print_token_sequence(vocabulary, sequence, fout):
    print(token_sequence_to_str(vocabulary, sequence), file=fout)

def print_multiple_token_sequences(vocabulary, sequences, fout):
    print('\t'.join((token_sequence_to_str(vocabulary, s) for s in sequences)), file=fout)

if __name__ == '__main__':
    LanguageModelingGenerateCommand().main()
