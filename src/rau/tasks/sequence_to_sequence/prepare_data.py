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
from rau.tasks.sequence_to_sequence.vocabulary import (
    SharedVocabularyData,
    get_vocabularies
)

def add_vocabulary_type_to_extension(path, vocabulary_type):
    return path.with_suffix(f'.{vocabulary_type}{path.suffix}')

def add_vocabulary_type_to_pairs(pairs, vocabulary_type):
    return [(s, add_vocabulary_type_to_extension(t, vocabulary_type)) for s, t in pairs]

def main():

    parser = argparse.ArgumentParser(
        description=
        'Convert tokenized parallel text to a prepared, integerized form that '
        'can be loaded efficiently. Input files (.tok) should have one '
        'sequence of whitespace-separated tokens per line. Prepared output '
        'files (.prepared) and a vocabulary file (.vocab) will be written.'
    )
    parser.add_argument('--training-data', type=pathlib.Path,
        help='A directory containing training data. The files '
             '<training-data>/source.tok and <training-data>/target.tok will '
             'be used as inputs, and the files '
             '<training-data>/source.<vocabulary-type>.prepared and '
             '<training-data>/target.<vocabulary-type>.prepared will be used '
             'as outputs.')
    parser.add_argument('--training-data-source-files', type=pathlib.Path, nargs=2,
        help='Input .tok file and output .prepared file for the source-side '
             'training data. Overrides --training-data. The vocabulary type '
             'will be added to the extension of the output file name.')
    parser.add_argument('--training-data-target-files', type=pathlib.Path, nargs=2,
        help='Input .tok file and output .prepared file for the target-side '
             'training data. Overrides --training-data. The vocabulary type '
             'will be added to the extension of the output file name.')
    parser.add_argument('--vocabulary-types', choices=['shared', 'separate'], nargs='*', default=[],
        help='Which types of vocabulary to generate: either "shared" or '
             '"separate". If "shared" is used, then a common vocabulary that '
             'is shared by both the source and the target side will be saved '
             'to the file <training-data>/shared.vocab. If "separate" is '
             'used, then separate vocabulary files will be saved to '
             '<training-data>/source.vocab and <training-data>/target.vocab. '
             'Both vocabulary types can be generated at once.')
    parser.add_argument('--shared-vocabulary-file', type=pathlib.Path,
        help='A .vocab file where a shared vocabulary will be saved. '
             'Overrides --training-data.')
    parser.add_argument('--source-vocabulary-file', type=pathlib.Path,
        help='A .vocab file where a separate source vocabulary will be saved. '
             'Overrides --training-data.')
    parser.add_argument('--target-vocabulary-file', type=pathlib.Path,
        help='A .vocab file where a separate target vocabulary will be saved. '
             'Overrides --training-data.')
    parser.add_argument('--more-data', action='append', default=[],
        help='Name of an additional dataset in the training data directory '
             'that will be prepared using the training data. This option can '
             'be passed multiple times. The files '
             '<training-data>/datasets/<more-data>/{source,target}.{tok,<vocabulary-type>.prepared} '
             'will be used as the source/target input/output files.')
    parser.add_argument('--more-source-data', action='append', default=[],
        help='Name of an additional dataset in the training data directory '
             'whose source side will be prepared using the training data. This '
             'option can be passed multiple times. The files '
             '<training-data>/datasets/<more-data>/source.{tok,<vocabulary-type>.prepared} will '
             'be used as the input/output files.')
    parser.add_argument('--more-target-data', action='append', default=[],
        help='Name of an additional dataset in the training data directory '
             'whose target side will be prepared using the training data. This '
             'option can be passed multiple times. The files '
             '<training-data>/datasets/<more-data>/target.{tok,<vocabulary-type>.prepared} will '
             'be used as the input/output files.')
    parser.add_argument('--more-source-data-files', action='append',
        type=pathlib.Path, nargs=2, default=[],
        help='An additional pair of source-side input (.tok) and output '
             '(.prepared) files that will be prepared using the training data. '
             'This option can be passed multiple times. This is an '
             'alternative to --more-data and --more-source-data. The '
             'vocabulary type will be added to the extension of the output '
             'file name.')
    parser.add_argument('--more-target-data-files', action='append',
        type=pathlib.Path, nargs=2, default=[],
        help='An additional pair of target-side input (.tok) and output '
             '(.prepared) files that will be prepared using the training data. '
             'This option can be passed multiple times. This is an '
             'alternative to --more-data and --more-target-data. The '
             'vocabulary type will be added to the extension of the output '
             'file name.')
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
        parser.error(
            'cannot pass both --always-allow-unk and --never-allow-unk')
    if args.training_data_source_files is not None:
        training_data_source_input_file, training_data_source_output_file = \
            args.training_data_source_files
    elif args.training_data is not None:
        training_data_source_input_file = args.training_data / 'source.tok'
        training_data_source_output_file = args.training_data / 'source.prepared'
    else:
        parser.error(
            'either --training-data or --training-data-source-files is '
            'required')
    if args.training_data_target_files is not None:
        training_data_target_input_file, training_data_target_output_file = \
            args.training_data_target_files
    elif args.training_data is not None:
        training_data_target_input_file = args.training_data / 'target.tok'
        training_data_target_output_file = args.training_data / 'target.prepared'
    else:
        parser.error(
            'either --training-data or --training-data-target-files is '
            'required')
    vocabulary_types = set(args.vocabulary_types)
    if args.shared_vocabulary_file is not None:
        vocabulary_types.add('shared')
    if args.source_vocabulary_file is not None or args.target_vocabulary_file is not None:
        vocabulary_types.add('separate')
    if not vocabulary_types:
        parser.error(
            'one of --vocabulary-types, --shared-vocabulary-file, '
            '--source-vocabulary-file, or --target-vocabulary-file is '
            'required')

    unk_string = None if args.never_allow_unk else args.unk_string

    if 'separate' in vocabulary_types:
        raise NotImplementedError(
            'using separate source and target vocabularies has not yet been '
            'implemented')

    if 'shared' in vocabulary_types:

        if args.shared_vocabulary_file is not None:
            vocab_output_file = args.shared_vocabulary_file
        elif args.training_data is not None:
            vocab_output_file = args.training_data / 'shared.vocab'
        else:
            parser.error(
                'either --training-data or --shared-vocabulary-file is '
                'required')
        prepared_source_files = [
            (
                training_data_source_input_file,
                add_vocabulary_type_to_extension(training_data_source_output_file, 'shared')
            )
        ]
        prepared_target_files = [
            (
                training_data_target_input_file,
                add_vocabulary_type_to_extension(training_data_target_output_file, 'shared')
            )
        ]
        more_source_dirs = [*args.more_data, *args.more_source_data]
        more_target_dirs = [*args.more_data, *args.more_target_data]
        if (more_source_dirs or more_target_dirs) and args.training_data is None:
            parser.error(
                'if --more-data, --more-source-data, or '
                '--more-target-data is used, then --training-data is '
                'required')
        for source_dir in more_source_dirs:
            data_dir = args.training_data / 'datasets' / source_dir
            prepared_source_files.append((
                data_dir / 'source.tok',
                data_dir / 'source.shared.prepared'
            ))
        for target_dir in more_target_dirs:
            data_dir = args.training_data / 'datasets' / target_dir
            prepared_target_files.append((
                data_dir / 'target.tok',
                data_dir / 'target.shared.prepared'
            ))
        prepared_source_files.extend(
            add_vocabulary_type_to_pairs(args.more_source_data_files, 'shared'))
        prepared_target_files.extend(
            add_vocabulary_type_to_pairs(args.more_target_data_files, 'shared'))

        source_token_types, source_has_unk = \
            get_token_types_in_file(training_data_source_input_file, unk_string)
        target_token_types, target_has_unk = \
            get_token_types_in_file(training_data_target_input_file, unk_string)
        allow_unk = args.always_allow_unk or source_has_unk or target_has_unk

        tokens_in_target = sorted(target_token_types)
        tokens_only_in_source = sorted(source_token_types - target_token_types)
        embedding_vocab, _, softmax_vocab = get_vocabularies(
            SharedVocabularyData(
                tokens_in_target,
                tokens_only_in_source,
                allow_unk
            ),
            ToIntVocabularyBuilder()
        )

        print(f'vocabulary type: shared')
        print(f'token types in target: {len(tokens_in_target)}', file=sys.stderr)
        print(f'token types only in source: {len(tokens_only_in_source)}', file=sys.stderr)
        print(f'embedding vocabulary size: {len(embedding_vocab)}', file=sys.stderr)
        print(f'softmax vocabulary size: {len(softmax_vocab)}', file=sys.stderr)
        print(f'source has {unk_string}: {source_has_unk}', file=sys.stderr)
        print(f'target has {unk_string}: {target_has_unk}', file=sys.stderr)
        print(f'allow unk: {allow_unk}', file=sys.stderr)
        print(f'writing {vocab_output_file}', file=sys.stderr)
        vocab_output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tokens_in_target' : tokens_in_target,
            'tokens_only_in_source' : tokens_only_in_source,
            'allow_unk' : allow_unk
        }, vocab_output_file)
        for pair in prepared_source_files:
            prepare_file(embedding_vocab, pair)
        for pair in prepared_target_files:
            prepare_file(softmax_vocab, pair)

if __name__ == '__main__':
    main()
