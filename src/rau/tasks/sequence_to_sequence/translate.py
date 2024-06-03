import argparse
import datetime
import logging
import pathlib
import sys

from rau.tools.ticker import TimedTicker
from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.sequence_to_sequence.data import load_vocabularies
from rau.tasks.sequence_to_sequence.model import SequenceToSequenceModelInterface
from rau.tasks.sequence_to_sequence.batching import group_sources_into_batches

def main():

    console_logger = logging.getLogger('main')
    console_logger.addHandler(logging.StreamHandler(sys.stderr))
    console_logger.setLevel(logging.INFO)

    model_interface = SequenceToSequenceModelInterface(
        use_load=True,
        use_init=False,
        use_output=False,
        require_output=False
    )

    parser = argparse.ArgumentParser(
        description=
        'Given a trained model, translate input sequences to output sequences.'
    )
    parser.add_argument('--input', type=pathlib.Path, required=True,
        help='A .prepared file of input sequences.')
    parser.add_argument('--beam-size', type=int, required=True,
        help='The beam size to use for beam search.')
    parser.add_argument('--max-target-length', type=int, required=True,
        help='The maximum allowed length, in tokens, for output sequences.')
    parser.add_argument('--batching-max-tokens', type=int, required=True,
        help='The maximum number of tokens allowed per batch. This puts a '
             'limit on the number of elements included in a single tensor of '
             'input sequences, incude BOS and padding tokens. If a single '
             'example exceeds the limit, it is not discarded, but included in '
             'a batch by itself.')
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    parser.add_argument('--shared-vocabulary-file', type=pathlib.Path,
        help='A .vocab file to be used as a shared source-target vocabulary.')
    parser.add_argument('--source-vocabulary-file', type=pathlib.Path,
        help='A .vocab file to be used as the source vocabulary.')
    parser.add_argument('--target-vocabulary-file', type=pathlib.Path,
        help='A .vocab file to be used as the target vocabulary.')
    parser.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages.')
    args = parser.parse_args()

    show_progress = not args.no_progress

    device = model_interface.get_device(args)
    sources = load_prepared_data_file(args.input)
    vocabs = load_vocabularies(args, parser, model_interface)
    saver = model_interface.construct_saver(args)
    model_interface.on_before_decode(saver, [sources], args.max_target_length)
    batches = list(group_sources_into_batches(
        sources,
        lambda b, s: b * s <= args.batching_max_tokens
    ))
    ordered_outputs = [None] * len(sources)
    if show_progress:
        ticker = TimedTicker(len(batches), 1)
        progress_num_sequences = 0
        start_time = progress_start_time = datetime.datetime.now()
    for batch_no, batch in enumerate(batches):
        source = model_interface.prepare_source([s for i, s in batch], device)
        output = model_interface.decode(
            model=saver.model,
            model_source=source,
            beam_size=args.beam_size,
            max_length=args.max_target_length
        )
        for (i, s), output_sequence in zip(batch, output, strict=True):
            ordered_outputs[i] = output_sequence
        if show_progress:
            progress_num_sequences += len(batch)
            ticker.progress = batch_no + 1
            if ticker.tick():
                progress_duration = datetime.datetime.now() - progress_start_time
                progress_sequences_per_second = progress_num_sequences / progress_duration.total_seconds()
                console_logger.info(
                    f'{ticker.int_percent}% | '
                    f'sequences/s: {progress_sequences_per_second:.2f}'
                )
                progress_num_sequences = 0
                progress_start_time = datetime.datetime.now()
    if show_progress:
        duration = datetime.datetime.now() - start_time
    for output_sequence in ordered_outputs:
        print(' '.join(vocabs.target_output_vocab.to_string(w) for w in output_sequence))
    if show_progress:
        sequences_per_second = len(ordered_outputs) / duration.total_seconds()
        console_logger.info(f'duration: {duration} | sequences/s: {sequences_per_second:.2f} | s/sequence: {1/sequences_per_second:.4f}')

if __name__ == '__main__':
    main()
