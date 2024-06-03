import dataclasses

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.models.transformer.encoder_decoder import get_transformer_encoder_decoder
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.generation.beam_search import beam_search
from rau.tasks.common.model import pad_sequences

from .vocabulary import get_vocabularies

class SequenceToSequenceModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--num-encoder-layers', type=int,
            help='Number of layers in the transformer encoder.')
        group.add_argument('--num-decoder-layers', type=int,
            help='Number of layers in the transformer decoder.')
        group.add_argument('--d-model', type=int,
            help='The size of the vector representations used in the '
                 'transformer.')
        group.add_argument('--num-heads', type=int,
            help='The number of attention heads used in each layer.')
        group.add_argument('--feedforward-size', type=int,
            help='The size of the hidden layer of the feedforward network in '
                 'each feedforward sublayer.')
        group.add_argument('--dropout', type=float,
            help='The dropout rate used throughout the transformer on input '
                 'embeddings, sublayer function outputs, feedforward hidden '
                 'layers, and attention weights.')
        group.add_argument('--init-scale', type=float,
            help='The scale used for the uniform distribution from which '
                 'certain parameters are initialized.')

    def get_kwargs(self, args, vocabulary_data):
        (
            source_vocab,
            target_input_vocab,
            target_output_vocab
        ) = get_vocabularies(vocabulary_data)
        return dict(
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            source_vocabulary_size=len(source_vocab),
            target_input_vocabulary_size=len(target_input_vocab),
            target_output_vocabulary_size=len(target_output_vocab),
            tie_embeddings=True,
            source_eos_index=source_vocab.eos_index,
            target_input_bos_index=target_input_vocab.bos_index,
            target_output_eos_index=target_output_vocab.eos_index
        )

    def construct_model(self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        source_vocabulary_size,
        target_input_vocabulary_size,
        target_output_vocabulary_size,
        tie_embeddings,
        source_eos_index,
        target_input_bos_index,
        target_output_eos_index
    ):
        if num_encoder_layers is None:
            raise ValueError
        if num_decoder_layers is None:
            raise ValueError
        if d_model is None:
            raise ValueError
        if num_heads is None:
            raise ValueError
        if feedforward_size is None:
            raise ValueError
        if dropout is None:
            raise ValueError
        return get_transformer_encoder_decoder(
            source_vocabulary_size=source_vocabulary_size,
            target_input_vocabulary_size=target_input_vocabulary_size,
            target_output_vocabulary_size=target_output_vocabulary_size,
            tie_embeddings=tie_embeddings,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_source_padding=False,
            use_target_padding=False
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def on_saver_constructed(self, args, saver):
        # See note about padding index in prepare_batch().
        self.source_eos_index = saver.kwargs['source_eos_index']
        self.target_input_bos_index = saver.kwargs['target_input_bos_index']
        self.target_output_eos_index = saver.kwargs['target_output_eos_index']
        self.output_padding_index = saver.kwargs['target_output_vocabulary_size']

    def adjust_source_length(self, source_length):
        # Add 1 for EOS.
        return source_length + 1

    def adjust_target_length(self, target_length):
        # Add 1 for BOS.
        return target_length + 1

    def get_vocabularies(self, vocabulary_data, builder=None):
        return get_vocabularies(vocabulary_data, builder)

    def prepare_batch(self, batch, device):
        model_source = self.prepare_source([s for s, t in batch], device)
        # See commments in rau/tasks/language_modeling/model.py for
        # prepare_batch().
        output_padding_index = self.output_padding_index
        whole_tensor = pad_sequences(
            [t for s, t in batch],
            device,
            bos=self.target_input_bos_index,
            eos=self.target_output_eos_index,
            pad=output_padding_index
        )
        target_input_tensor = whole_tensor[:, :-1]
        target_output_tensor = whole_tensor[:, 1:]
        model_input = ModelSourceAndTarget(
            source=model_source.source,
            source_is_padding_mask=model_source.source_is_padding_mask,
            target=target_input_tensor
        )
        return model_input, target_output_tensor

    def prepare_source(self, sources, device):
        source = pad_sequences(
            sources,
            device,
            eos=self.source_eos_index,
            pad=-1
        )
        # Using a reserved -1 padding index lets us compute the mask on GPU.
        mask = (source == -1)
        # We need to make sure the padding index is a valid embedding index, so
        # we arbitrarily set it to 0. It doesn't matter what value is used; it
        # will always be ignored because of the mask.
        source[mask] = 0
        return ModelSource(
            source=source,
            source_is_padding_mask=mask
        )

    def on_before_process_pairs(self, saver, datasets):
        max_length = max(
            max(
                self.adjust_source_length(len(s)),
                self.adjust_target_length(len(t))
            )
            for dataset in datasets
            for s, t in dataset
        )
        self._preallocate_positional_encodings(saver, max_length)

    def on_before_decode(self, saver, datasets, max_target_length):
        data_max_length = self.adjust_source_length(max(
            len(s)
            for dataset in datasets
            for s in dataset
        ))
        # Subtract 1 because beam search doesn't need the last input.
        max_target_length = self.adjust_target_length(max_target_length) - 1
        max_length = max(max_target_length, data_max_length)
        self._preallocate_positional_encodings(saver, max_length)

    def _preallocate_positional_encodings(self, saver, max_length):
        # Precompute all of the sinusoidal positional encodings up-front based
        # on the maximum length that will be required. This should help with
        # GPU memory fragmentation.
        d_model = saver.kwargs['d_model']
        for module in saver.model.modules():
            if isinstance(module, SinusoidalPositionalEncodingCacher):
                module.get_encodings(max_length, d_model)
                module.set_allow_reallocation(False)

    def get_logits(self, model, model_input):
        # Note that it is unnecessary to pass a padding mask for the target
        # side, because padding only occurs at the end of a sequence, and the
        # decoder is already causally masked.
        return model(
            source_sequence=model_input.source,
            target_sequence=model_input.target,
            source_is_padding_mask=model_input.source_is_padding_mask
        )

    def decode(self, model, model_source, beam_size, max_length):
        model.eval()
        with torch.inference_mode():
            decoder_state = model.initial_decoder_state(
                source_sequence=model_source.source,
                source_is_padding_mask=model_source.source_is_padding_mask
            )
            device = model_source.source.device
            # Feed BOS into the model at the first timestep.
            decoder_state = decoder_state.next(torch.full(
                (decoder_state.batch_size(),),
                self.target_input_bos_index,
                dtype=torch.long,
                device=device
            ))
            return beam_search(
                decoder_state,
                beam_size,
                self.target_output_eos_index,
                max_length,
                device
            )

@dataclasses.dataclass
class ModelSource:
    source: torch.Tensor
    source_is_padding_mask: torch.Tensor

@dataclasses.dataclass
class ModelSourceAndTarget(ModelSource):
    target: torch.Tensor
