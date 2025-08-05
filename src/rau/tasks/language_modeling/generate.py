import pathlib

import torch

from rau.tasks.common.command import Command
from rau.tasks.language_modeling.model import LanguageModelingModelInterface
from rau.tasks.language_modeling.vocabulary import load_vocabulary_data_from_file
from rau.generation.sample import sample_single
from rau.generation.greedy import decode_greedily_single

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
        parser.add_argument('--mode',
            choices=['random', 'greedy', 'beam-search'],
            default='random',
            help='Which generation algorithm to use. Choices: '
                 'random: Random sampling or ancestral sampling; '
                 'greedy: Greedy decoding; '
                 'beam-search: Beam search with length normalization.')
        parser.add_argument('--num-samples', type=int, default=1,
            help='Number of samples to generate when using random mode.')
        parser.add_argument('--max-length', type=int,
            help='Optional maximum number of tokens generated per sequence. If '
                 'this is not given, generation may run arbitrarily long.')
        parser.add_argument('--random-seed', type=int,
            help='Optional random seed for generating random samples.')
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

        saver = model_interface.construct_saver(args)
        model = saver.model
        device = model_interface.get_device(args)
        eos_index = model_interface.eos_index
        _, vocab = model_interface.get_vocabularies(
            load_vocabulary_data_from_file(vocab_file),
            include_embedding_vocab=False
        )
        model.eval()
        with torch.inference_mode():
            initial_state = model_interface.get_initial_state(
                saver.model,
                batch_size=1,
                device=device
            )
            def generate_outputs():
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
                        pass
            for s in generate_outputs():
                print(' '.join(map(vocab.to_string, s)))

if __name__ == '__main__':
    LanguageModelingGenerateCommand().main()
