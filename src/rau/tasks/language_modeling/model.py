import functools

import humanfriendly
import torch

from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.unidirectional import SimpleLayerUnidirectional, OutputUnidirectional
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.unidirectional_encoder import (
    get_shared_embeddings,
    get_unidirectional_transformer_encoder
)
from rau.tasks.common.model import pad_sequences

class LanguageModelingModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--num-layers', type=int,
            help='Number of layers in the transformer.')
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

    def get_kwargs(self, args, input_vocab_size, output_vocab_size):
        return dict(
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size
        )

    def construct_model(self,
        num_layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        input_vocab_size,
        output_vocab_size
    ):
        if num_layers is None:
            raise ValueError
        if d_model is None:
            raise ValueError
        if num_heads is None:
            raise ValueError
        if feedforward_size is None:
            raise ValueError
        if dropout is None:
            raise ValueError
        return get_unidirectional_transformer_encoder(
            input_vocabulary_size=input_vocab_size,
            output_vocabulary_size=output_vocab_size,
            tie_embeddings=True,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            feedforward_size=feedforward_size,
            dropout=dropout,
            use_padding=True
        )

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def adjust_length(self, length):
        # Add 1 for BOS.
        return length + 1

    def prepare_batch(self, batch, device, data):
        # Use the same index for padding symbols in both the input and output
        # tensor. The input vocab should be bigger than the output vocab, so
        # using the length of the input vocab should work fine. Using the same
        # padding symbol for both allows us to allocate one tensor and simply
        # slice it to get the input and output tensors.
        pad = len(data.input_vocab)
        whole_tensor = pad_sequences(
            batch,
            device,
            bos=data.input_vocab.bos_index,
            eos=data.output_vocab.eos_index,
            pad=pad
        )
        input_tensor = whole_tensor[:, :-1]
        output_tensor = whole_tensor[:, 1:]
        return input_tensor, output_tensor

    def get_logits(self, model, model_input):
        # Note that it is unnecessary to pass a padding mask, because padding
        # only occurs at the end of a sequence, and the model is already
        # causally masked.
        return model(model_input, include_first=False)
