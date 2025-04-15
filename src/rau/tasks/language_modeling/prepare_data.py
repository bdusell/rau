import argparse
import pathlib
import sys

import torch

from rau.vocab import ToIntVocabularyBuilder
from rau.tasks.common.data_preparation import (
    add_prepare_data_args,
    validate_prepare_data_args,
    get_token_types_in_file,
    prepare_file
)
from rau.tasks.language_modeling.vocabulary import (
    load_vocabulary_data_from_file,
    build_softmax_vocab
)

def main():

    parser = argparse.ArgumentParser(
        description=
        'Convert tokenized text to a prepared, integerized form that can be '
        'loaded efficiently. Input files (.tok) should have one sequence of '
        'whitespace-separated tokens per line. A prepared output file '
        '(.prepared) and a vocabulary file (.vocab) will be written.'
    )
    parser.add_argument('--training-data', type=pathlib.Path,
        help='An optional directory containing training data. The file '
             '<training-data>/main.tok will be used as input, and the file '
             '<training-data>/main.prepared will be used as output. '
             'The vocabulary will be saved to the file '
             '<training-data>/main.vocab.')
    parser.add_argument('--training-data-files', type=pathlib.Path, nargs=2,
        help='Input .tok file and output .prepared file for the training '
             'data. Overrides --training-data.')
    parser.add_argument('--vocabulary-file', type=pathlib.Path,
        help='A .vocab file where the vocabulary will be saved. Overrides '
             '--training-data. If --training-data is not given, the '
             'vocabulary is loaded from this file instead.')
    parser.add_argument('--more-data', action='append', default=[],
        help='Name of an additional dataset in the training data directory '
             'that will be prepared using the training data. This option can '
             'be passed multiple times. The file '
             '<training-data>/datasets/<more-data>/main.tok will be used as '
             'input, and the file '
             '<training-data>/datasets/<more-data>/main.prepared will be used '
             'as output.')
    parser.add_argument('--more-data-files',
        type=pathlib.Path, nargs=2, action='append', dest='more_data',
        help='An additional pair of input (.tok) and output (.prepared) files '
             'that will be preprocessed using the training data. This option '
             'can be passed multiple times. This is an alternative to '
             '--more-data.')
    parser.add_argument('--always-allow-unk', action='store_true', default=False,
        help='Always allow the vocabulary to include an <unk> token, even if '
             'one does not appear in the training data.')
    parser.add_argument('--never-allow-unk', action='store_true', default=False,
        help='Never allow the vocabulary to include an <unk> token; treat '
             'every token as a normal token in the vocabulary. This is useful '
             'for datasets that already have <unk> preprocessing done.')
    add_prepare_data_args(parser)
    args = parser.parse_args()
    validate_prepare_data_args(parser, args)

    if args.always_allow_unk and args.never_allow_unk:
        parser.error('cannot pass both --always-allow-unk and --never-allow-unk')
    if args.training_data_files is not None:
        training_data_input_file, training_data_output_file = args.training_data_files
    elif args.training_data is not None:
        training_data_input_file = args.training_data / 'main.tok'
        training_data_output_file = args.training_data / 'main.prepared'
    else:
        training_data_input_file = None
        training_data_output_file = None
    if args.vocabulary_file is not None:
        vocab_file = args.vocabulary_file
    elif args.training_data is not None:
        vocab_file = args.training_data / 'main.vocab'
    else:
        parser.error('either --training-data or --vocabulary-file is required')
    prepared_files = []
    if training_data_input_file is not None:
        prepared_files.append((training_data_input_file, training_data_output_file))
    for arg in args.more_data:
        if isinstance(arg, str):
            if args.training_data is not None:
                more_data_dir = args.training_data / 'datasets' / arg
                input_file = more_data_dir / 'main.tok'
                output_file = more_data_dir / 'main.prepared'
            else:
                parser.error('if --more-data is used, then --training-data is required')
        else:
            input_file, output_file = arg
        prepared_files.append((input_file, output_file))

    if training_data_input_file is not None:
        unk_string = None if args.never_allow_unk else args.unk_string
        token_types, has_unk = get_token_types_in_file(training_data_input_file, unk_string)
        allow_unk = (args.always_allow_unk or has_unk) and not args.never_allow_unk
        tokens = sorted(token_types)
        vocab = build_softmax_vocab(tokens, allow_unk, ToIntVocabularyBuilder())
        print(f'token types: {len(token_types)}', file=sys.stderr)
        print(f'vocabulary size: {len(vocab)}', file=sys.stderr)
        print(f'has unk ({unk_string}): {has_unk}', file=sys.stderr)
        print(f'allow unk: {allow_unk}', file=sys.stderr)
        print(f'writing {vocab_file}', file=sys.stderr)
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tokens' : tokens,
            'allow_unk' : allow_unk
        }, vocab_file)
    else:
        vocab_data = load_vocabulary_data_from_file(vocab_file)
        vocab = build_softmax_vocab(
            vocab_data.tokens,
            vocab_data.allow_unk,
            ToIntVocabularyBuilder()
        )
    for pair in prepared_files:
        prepare_file(vocab, pair)

if __name__ == '__main__':
    main()
