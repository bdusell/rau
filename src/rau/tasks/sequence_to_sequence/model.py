import dataclasses
import re

import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.tools.torch.compose import Composable
from rau.unidirectional import SimpleLayerUnidirectional, OutputUnidirectional
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.input_layer import get_transformer_input_unidirectional
from rau.models.transformer.encoder_decoder import get_shared_embeddings
from rau.models.transformer.encoder import get_transformer_encoder
from rau.models.transformer.decoder import get_transformer_decoder
from rau.generation.beam_search import beam_search
from rau.tasks.common.model import pad_sequences

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

    def get_kwargs(self,
        args,
        source_vocab_size,
        target_input_vocab_size,
        target_output_vocab_size,
        tie_embeddings
    ):
        return dict(
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            source_vocab_size=source_vocab_size,
            target_input_vocab_size=target_input_vocab_size,
            target_output_vocab_size=target_output_vocab_size,
            tie_embeddings=tie_embeddings
        )

    def construct_model(self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        source_vocab_size,
        target_input_vocab_size,
        target_output_vocab_size,
        tie_embeddings
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
        # TODO Use function from rau.models.transformer.encoder_decoder
        return get_encoder_decoder(
            source_vocabulary_size=source_vocab_size,
            target_input_vocabulary_size=target_input_vocab_size,
            target_output_vocabulary_size=target_output_vocab_size,
            tie_embeddings=tie_embeddings,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_source_padding=True,
            use_target_padding=True
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def adjust_source_length(self, source_length):
        # Add 1 for EOS.
        return source_length + 1

    def adjust_target_length(self, target_length):
        # Add 1 for BOS.
        return target_length + 1

    def prepare_batch(self, batch, device, data):
        model_source = self.prepare_source([s for s, t in batch], device, data)
        target_input_pad = len(data.target_input_vocab)
        target_input = pad_sequences(
            [t for s, t in batch],
            device,
            bos=data.target_input_vocab.bos_index,
            pad=target_input_pad
        )
        target_output_pad = len(data.target_output_vocab)
        target_output = pad_sequences(
            [t for s, t in batch],
            device,
            eos=data.target_output_vocab.eos_index,
            pad=target_output_pad
        )
        model_input = ModelSourceAndTarget(
            source=model_source.source,
            source_is_padding_mask=model_source.source_is_padding_mask,
            target=target_input
        )
        return model_input, target_output

    def prepare_source(self, sources, device, data):
        source_pad = len(data.source_vocab)
        source = pad_sequences(
            sources,
            device,
            eos=data.source_vocab.eos_index,
            pad=source_pad
        )
        return ModelSource(
            source=source,
            source_is_padding_mask=(source == source_pad)
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
        data_max_length = max(
            self.adjust_source_length(len(s))
            for dataset in datasets
            for s in dataset
        )
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
            encoder_kwargs=self.get_encoder_kwargs(model_input),
            decoder_kwargs=self.get_decoder_kwargs(model_input)
        )

    def decode(self, model, model_source, bos_symbol, beam_size, eos_symbol, max_length):
        model.eval()
        with torch.no_grad():
            decoder_state = model.initial_decoder_state(
                source_sequence=model_source.source,
                encoder_kwargs=self.get_encoder_kwargs(model_source),
                decoder_kwargs=self.get_decoder_kwargs(model_source)
            )
            device = model_source.source.device
            # Feed BOS into the model at the first timestep.
            decoder_state = decoder_state.next(torch.full(
                (decoder_state.batch_size(),),
                bos_symbol,
                dtype=torch.long,
                device=device
            ))
            return beam_search(decoder_state, beam_size, eos_symbol, max_length, device)

    def get_encoder_kwargs(self, model_source):
        return dict(tag_kwargs=dict(
            transformer=dict(
                is_padding_mask=model_source.source_is_padding_mask
            )
        ))

    def get_decoder_kwargs(self, model_source):
        return dict(tag_kwargs=dict(
            transformer=dict(
                encoder_is_padding_mask=model_source.source_is_padding_mask
            )
        ))

@dataclasses.dataclass
class ModelSource:
    source: torch.Tensor
    source_is_padding_mask: torch.Tensor

@dataclasses.dataclass
class ModelSourceAndTarget(ModelSource):
    target: torch.Tensor

def get_encoder_decoder(
    source_vocabulary_size,
    target_input_vocabulary_size,
    target_output_vocabulary_size,
    tie_embeddings,
    num_encoder_layers,
    num_decoder_layers,
    d_model,
    num_heads,
    feedforward_size,
    dropout,
    use_source_padding=True,
    use_target_padding=True
):
    shared_embeddings = get_shared_embeddings(
        tie_embeddings,
        source_vocabulary_size,
        target_input_vocabulary_size,
        target_output_vocabulary_size,
        d_model,
        use_source_padding,
        use_target_padding
    )
    positional_encoding_cacher = SinusoidalPositionalEncodingCacher()
    return EncoderDecoder(
        get_transformer_encoder(
            vocabulary_size=source_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_source_padding,
            tag='transformer'
        ),
        get_transformer_decoder(
            input_vocabulary_size=target_input_vocabulary_size,
            output_vocabulary_size=target_output_vocabulary_size,
            shared_embeddings=shared_embeddings,
            positional_encoding_cacher=positional_encoding_cacher,
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=use_target_padding,
            tag='transformer'
        )
    )

class EncoderDecoder(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
        source_sequence,
        target_sequence,
        encoder_kwargs,
        decoder_kwargs
    ):
        encoder_outputs = self.encoder(
            source_sequence,
            **encoder_kwargs
        )
        decoder_kwargs['tag_kwargs']['transformer']['encoder_sequence'] = encoder_outputs
        return self.decoder(
            target_sequence,
            **decoder_kwargs,
            include_first=False
        )

    def initial_decoder_state(self, source_sequence, encoder_kwargs, decoder_kwargs):
        encoder_outputs = self.encoder(
            source_sequence,
            **encoder_kwargs
        )
        decoder_kwargs['tag_kwargs']['transformer']['encoder_sequence'] = encoder_outputs
        return self.decoder.initial_state(
            batch_size=encoder_outputs.size(0),
            **decoder_kwargs
        )
