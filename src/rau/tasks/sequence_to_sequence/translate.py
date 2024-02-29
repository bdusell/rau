import argparse
import pathlib

from rau.tasks.common.data import load_prepared_data_file
from rau.tasks.sequence_to_sequence.data import load_vocabularies
from rau.tasks.sequence_to_sequence.model import SequenceToSequenceModelInterface
from rau.tasks.sequence_to_sequence.batching import group_sources_into_batches

def main():

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
    args = parser.parse_args()

    device = model_interface.get_device(args)
    sources = load_prepared_data_file(args.input)
    vocabs = load_vocabularies(args, parser)
    saver = model_interface.construct_saver(args)
    model_interface.on_before_decode(saver, [sources], args.max_target_length)
    batches = list(group_sources_into_batches(
        sources,
        lambda b, s: b * s <= args.batching_max_tokens
    ))
    ordered_outputs = [None] * len(sources)
    for batch in batches:
        source = model_interface.prepare_source([s for i, s in batch], device, vocabs)
        output = model_interface.decode(
            model=saver.model,
            model_source=source,
            bos_symbol=vocabs.target_input_vocab.bos_index,
            beam_size=args.beam_size,
            eos_symbol=vocabs.target_output_vocab.eos_index,
            max_length=args.max_target_length
        )
        for (i, s), output_sequence in zip(batch, output, strict=True):
            ordered_outputs[i] = output_sequence
    for output_sequence in ordered_outputs:
        print(' '.join(vocabs.target_output_vocab.to_string(w) for w in output_sequence))

if __name__ == '__main__':
    main()
