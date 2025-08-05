import pathlib
import sys

import torch

from rau.tasks.common.command import Command
from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import load_vocabulary_data_from_file
from rau.generation.sample import sample_single
from rau.generation.greedy import decode_greedily_single
from rau.generation.beam_search import beam_search_single

class LanguageModelingGenerateCommand(Command):

    def __init__(self):
        super().__init__()
        self.model_interface = LanguageModelingModelInterface(
            use_load=True,
            use_init=False,
            use_output=False,
            require_output=False
        )

    DESCRIPTION = 'Randomly sample strings from a language model.'

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
                 'in each dataset. The name "training" can be used to evaluate '
                 'on the training data.')
        parser.add_argument('--output', type=pathlib.Path,
            help='Optional directory where output files will be written when '
                 'using --prompt-datasets. A separate file will be written for '
                 'each prompt dataset.')
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
            help='Optional random seed for generating random samples.')
        parser.add_argument('--beam-size', type=int,
            help='Beam size for beam search.')
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

        def generate_outputs(initial_state):
            match args.mode:
                case 'random':
                    if args.random_seed is not None:
                        generator = torch.Generator(device=device).manual_seed(args.random_seed)
                    else:
                        generator = None
                    for _ in range(args.num_samples):
                        yield sample_single(
                            initial_state=initial_state,
                            eos_symbol=eos_index,
                            max_length=args.max_length,
                            generator=generator
                        )
                case 'greedy':
                    yield decode_greedily_single(
                        initial_state=initial_state,
                        eos_symbol=eos_index,
                        max_length=args.max_length
                    )
                case 'beam-search':
                    yield beam_search_single(
                        initial_state=initial_state,
                        beam_size=args.beam_size,
                        eos_symbol=eos_index,
                        max_length=args.max_length,
                        device=device
                    )

        model.eval()
        with torch.inference_mode():
            if args.prompt_datasets is not None:
                for prompt_dataset in args.prompt_datasets:
                    input_file_name = args.training_data / 'datasets' / prompt_dataset / 'main.prepared'
                    prompts = load_prepared_data_file(input_file_name)
                    if args.output is not None:
                        output_file_name = args.output / f'{prompt_dataset}.tok'
                        print(f'writing {output_file_name}')
                        output_file = output_file_name.open('w')
                    else:
                        output_file = sys.stdout
                    with output_file:
                        for prompt in prompts:
                            prompt = prompt.to(device)
                            state = model_interface.get_initial_state(
                                saver.model,
                                batch_size=1,
                                device=device
                            )
                            state = state.fastforward(prompt.unsqueeze(0))
                            for s in generate_outputs(state):
                                print(' '.join(map(vocab.to_string, s)), file=output_file)
            else:
                initial_state = model_interface.get_initial_state(
                    saver.model,
                    batch_size=1,
                    device=device
                )
                for s in generate_outputs(initial_state):
                    print(' '.join(map(vocab.to_string, s)))

if __name__ == '__main__':
    LanguageModelingGenerateCommand().main()
