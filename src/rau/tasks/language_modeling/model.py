from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.unidirectional_encoder import (
    get_unidirectional_transformer_encoder
)
from rau.models.rnn import SimpleRNN, LSTM
from rau.models.common.shared_embeddings import get_shared_embeddings
from rau.unidirectional import (
    EmbeddingUnidirectional,
    DropoutUnidirectional,
    OutputUnidirectional
)
from rau.tasks.common.model import pad_sequences
from .vocabulary import get_vocabularies

class LanguageModelingModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--architecture', choices=['transformer', 'rnn', 'lstm'],
            help='The type of neural network architecture to use.')
        group.add_argument('--num-layers', type=int,
            help='(transformer, rnn, lstm) Number of layers.')
        group.add_argument('--d-model', type=int,
            help='(transformer) The size of the vector representations used '
                 'in the transformer.')
        group.add_argument('--num-heads', type=int,
            help='(transformer) The number of attention heads used in each '
                 'layer.')
        group.add_argument('--feedforward-size', type=int,
            help='(transformer) The size of the hidden layer of the '
                 'feedforward network in each feedforward sublayer.')
        group.add_argument('--dropout', type=float,
            help='(transformer) The dropout rate used throughout the '
                 'transformer on input embeddings, sublayer function outputs, '
                 'feedforward hidden layers, and attention weights. '
                 '(rnn, lstm) The dropout rate used between all layers, '
                 'including between the input embedding layer and the first '
                 'layer, and between the last layer and the output layer.')
        group.add_argument('--hidden-units', type=int,
            help='(rnn, lstm) Number of hidden units to use in the hidden '
                 'state.')
        group.add_argument('--init-scale', type=float,
            help='The scale used for the uniform distribution from which '
                 'certain parameters are initialized.')

    def get_kwargs(self, args, vocabulary_data):
        uses_bos = args.architecture == 'transformer'
        input_vocab, output_vocab = get_vocabularies(vocabulary_data, uses_bos)
        return dict(
            architecture=args.architecture,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            hidden_units=args.hidden_units,
            input_vocabulary_size=len(input_vocab),
            output_vocabulary_size=len(output_vocab),
            bos_index=input_vocab.bos_index if uses_bos else None,
            eos_index=output_vocab.eos_index
        )

    def construct_model(self,
        architecture,
        num_layers,
        d_model,
        num_heads,
        feedforward_size,
        dropout,
        hidden_units,
        input_vocabulary_size,
        output_vocabulary_size,
        bos_index,
        eos_index
    ):
        if architecture is None:
            raise ValueError
        if architecture == 'transformer':
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
                input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size,
                tie_embeddings=True,
                num_layers=num_layers,
                d_model=d_model,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout=dropout,
                use_padding=False
            )
        elif architecture in ('rnn', 'lstm'):
            if hidden_units is None:
                raise ValueError
            if num_layers is None:
                raise ValueError
            if dropout is None:
                raise ValueError
            # First, construct the recurrent hidden state module.
            if architecture == 'rnn':
                core = SimpleRNN(
                    input_size=hidden_units,
                    hidden_units=hidden_units,
                    layers=num_layers,
                    dropout=dropout,
                    learned_hidden_state=True
                )
            else:
                core = LSTM(
                    input_size=hidden_units,
                    hidden_units=hidden_units,
                    layers=num_layers,
                    dropout=dropout,
                    learned_hidden_state=True
                )
            # Now, sandwich the recurrent hidden state between an input
            # embedding layer and an output layer.
            shared_embeddings = get_shared_embeddings(
                tie_embeddings=True,
                input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size,
                embedding_size=hidden_units,
                use_padding=False
            )
            return (
                EmbeddingUnidirectional(
                    vocabulary_size=input_vocabulary_size,
                    output_size=hidden_units,
                    use_padding=False,
                    shared_embeddings=shared_embeddings
                ) |
                DropoutUnidirectional(dropout) |
                core.main() |
                DropoutUnidirectional(dropout) |
                OutputUnidirectional(
                    input_size=hidden_units,
                    vocabulary_size=output_vocabulary_size,
                    shared_embeddings=shared_embeddings,
                    bias=False
                )
            )
        else:
            raise ValueError

    def initialize(self, args, model, generator):
        if args.init_scale is None:
            raise ValueError
        smart_init(model, generator, fallback=uniform_fallback(args.init_scale))

    def on_saver_constructed(self, args, saver):
        # See comments in prepare_batch().
        self.bos_index = saver.kwargs['bos_index']
        self.uses_bos = self.bos_index is not None
        self.eos_index = saver.kwargs['eos_index']
        self.output_padding_index = saver.kwargs['output_vocabulary_size']

    def adjust_length(self, length):
        # Add 1 for BOS.
        return length + int(self.uses_bos)

    def get_vocabularies(self, vocabulary_data, builder=None):
        return get_vocabularies(vocabulary_data, self.uses_bos, builder)

    def prepare_batch(self, batch, device):
        # For transformers, use the same index for padding symbols in both the
        # input and output tensor. The padding index needs to be (1) a value
        # unique from all other indexes used in the output, and (2) a valid
        # index for the input embedding matrix.
        # For transformers, because BOS is always in the input vocabulary and
        # never in the output vocabulary, using the size of the output
        # vocabulary satisfies both of these constraints.
        # Using the same padding symbol in the input and output tensors allows
        # us to allocate one tensor and simply slice it, saving memory. The EOS
        # symbol will appear as an input symbol, but its embedding will never
        # receive gradient, because it will only appear in positions where the
        # output is padding, so it is the same as if padding were given as
        # input.
        output_padding_index = self.output_padding_index
        whole_tensor = pad_sequences(
            batch,
            device,
            bos=self.bos_index,
            eos=self.eos_index,
            pad=output_padding_index
        )
        input_tensor = whole_tensor[:, :-1]
        # Remove BOS from the expected output tensor.
        if self.uses_bos:
            output_tensor = whole_tensor[:, 1:]
        else:
            output_tensor = whole_tensor
        # For RNNs, the input vocabulary does not contain any symbols that are
        # not in the output, so the size of the vocabulary is not a valid
        # embedding index. So, for the input tensor, we create a copy and
        # change the padding index to 0.
        # TODO Use packed sequences for RNNs?
        if not self.uses_bos:
            input_tensor = input_tensor.clone()
            input_tensor[input_tensor == output_padding_index] = 0
        return input_tensor, output_tensor

    def on_before_process_pairs(self, saver, datasets):
        if saver.kwargs['architecture'] == 'transformer':
            max_length = max(
                len(x)
                for dataset in datasets
                for x in dataset
            )
            self._preallocate_positional_encodings(saver, self.adjust_length(max_length))

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
        # Note that for the transformer, it is unnecessary to pass a padding
        # mask, because padding only occurs at the end of a sequence, and the
        # model is already causally masked.
        return model(model_input, include_first=not self.uses_bos)
