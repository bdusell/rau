from rau.tools.torch.model_interface import ModelInterface
from rau.tools.torch.init import smart_init, uniform_fallback
from rau.models.transformer.positional_encodings import SinusoidalPositionalEncodingCacher
from rau.models.transformer.unidirectional_encoder import (
    get_unidirectional_transformer_encoder
)
from rau.models.rnn.language_model import (
    get_simple_rnn_language_model,
    get_lstm_language_model
)
from rau.models.stack_nn.transformer.parse import (
    parse_stack_transformer_layers,
    STACK_TRANSFORMER_LAYERS_HELP_MESSAGE
)
from rau.models.stack_nn.transformer.unidirectional_encoder import (
    get_unidirectional_stack_transformer_encoder
)
from rau.models.stack_nn.rnn.parse import (
    parse_stack_rnn_stack
)
from rau.models.stack_nn.rnn.language_model import (
    get_stack_rnn_language_model
)
from rau.tasks.common.model import pad_sequences
from rau.tasks.common.einsum import (
    add_einsum_forward_arguments,
    get_einsum_block_size
)

from .vocabulary import get_vocabularies

class LanguageModelingModelInterface(ModelInterface):

    def add_more_init_arguments(self, group):
        group.add_argument('--architecture',
            choices=['transformer', 'rnn', 'lstm', 'stack-transformer', 'stack-rnn'],
            help='The type of neural network architecture to use. '
                 'transformer: Standard transformer architecture with '
                 'sinusoidal positional encodings, based on the PyTorch '
                 'implementation. '
                 'rnn: Simple RNN (a.k.a. Elman RNN) with tanh activations. '
                 'lstm: LSTM. '
                 'stack-transformer: A transformer that can contain stack '
                 'attention layers. '
                 'stack-rnn: An RNN controller augmented with a '
                 'differentiable stack.')
        group.add_argument('--num-layers', type=int,
            help='(transformer, rnn, lstm) Number of layers. '
                 '(stack-rnn) Number of layers in the recurrent controller.')
        group.add_argument('--d-model', type=int,
            help='(transformer, stack-transformer) The size of the vector '
                 'representations used in the transformer.')
        group.add_argument('--num-heads', type=int,
            help='(transformer, stack-transformer) The number of attention '
                 'heads used in each scaled dot-product attention layer.')
        group.add_argument('--feedforward-size', type=int,
            help='(transformer, stack-transformer) The size of the hidden '
                 'layer of the feedforward network in each feedforward '
                 'sublayer.')
        group.add_argument('--dropout', type=float,
            help='(transformer, stack-transformer) The dropout rate used '
                 'throughout the transformer on input embeddings, sublayer '
                 'function outputs, feedforward hidden layers, and attention '
                 'weights. '
                 '(rnn, lstm) The dropout rate used between all layers, '
                 'including between the input embedding layer and the first '
                 'layer, and between the last layer and the output layer. '
                 '(stack-rnn) Same as rnn and lstm. Stack actions are '
                 'computed from the dropped-out hidden state.')
        group.add_argument('--hidden-units', type=int,
            help='(rnn, lstm, stack-rnn) Number of hidden units to use in '
                 'the hidden state.')
        group.add_argument('--stack-transformer-layers', type=parse_stack_transformer_layers,
            help='(stack-transformer) A string describing which layers to '
                 'use. ' +
                 STACK_TRANSFORMER_LAYERS_HELP_MESSAGE)
        group.add_argument('--stack-rnn-controller', choices=['rnn', 'lstm'],
            help='(stack-rnn) The type of RNN to use as the controller.')
        group.add_argument('--stack-rnn-stack', type=parse_stack_rnn_stack,
            help='(stack-rnn) The type of differentiable stack to connect to '
                 'the RNN controller. Options are: '
                 '(1) stratification-<m>, where <m> is an integer, indicating '
                 'a stratification stack with stack embedding size <m> '
                 '(2) superposition-<m>, where <m> is an integer, indicating '
                 'a superposition stack with stack embedding size <m> '
                 '(3) nondeterministic-<q>-<s>, where <q>, <s> are integers, '
                 'indicating a nondeterministic stack with <q> states and <s> '
                 'stack symbol types '
                 '(4) vector-nondeterministic-<q>-<s>-<m>, where <q>, <s>, '
                 '<m> are integers, indicating a vector nondeterministic '
                 'stack with <q> states, <s> stack symbol types, and stack '
                 'embedding type <m>.')
        group.add_argument('--init-scale', type=float,
            help='The scale used for the uniform distribution from which '
                 'certain parameters are initialized.')

    def add_forward_arguments(self, parser):
        group = parser.add_argument_group('Model Execution')
        add_einsum_forward_arguments(group)

    def get_kwargs(self, args, vocabulary_data):
        uses_bos = args.architecture in ('transformer', 'stack-transformer')
        input_vocab, output_vocab = get_vocabularies(vocabulary_data, uses_bos)
        return dict(
            architecture=args.architecture,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            feedforward_size=args.feedforward_size,
            dropout=args.dropout,
            hidden_units=args.hidden_units,
            stack_transformer_layers=args.stack_transformer_layers,
            stack_rnn_controller=args.stack_rnn_controller,
            stack_rnn_stack=args.stack_rnn_stack,
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
        stack_transformer_layers,
        stack_rnn_controller,
        stack_rnn_stack,
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
            if architecture == 'rnn':
                return get_simple_rnn_language_model(
                    input_vocabulary_size=input_vocabulary_size,
                    output_vocabulary_size=output_vocabulary_size,
                    hidden_units=hidden_units,
                    layers=num_layers,
                    dropout=dropout,
                    learned_hidden_state=True,
                    use_padding=False
                )
            else:
                return get_lstm_language_model(
                    input_vocabulary_size=input_vocabulary_size,
                    output_vocabulary_size=output_vocabulary_size,
                    hidden_units=hidden_units,
                    layers=num_layers,
                    dropout=dropout,
                    learned_hidden_state=True,
                    use_padding=False
                )
        elif architecture == 'stack-transformer':
            if stack_transformer_layers is None:
                raise ValueError
            if d_model is None:
                raise ValueError
            if num_heads is None:
                raise ValueError
            if feedforward_size is None:
                raise ValueError
            if dropout is None:
                raise ValueError
            return get_unidirectional_stack_transformer_encoder(
                input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size,
                tie_embeddings=True,
                layers=stack_transformer_layers,
                d_model=d_model,
                num_heads=num_heads,
                feedforward_size=feedforward_size,
                dropout=dropout,
                use_padding=False
            )
        elif architecture == 'stack-rnn':
            if hidden_units is None:
                raise ValueError
            if num_layers is None:
                raise ValueError
            if stack_rnn_controller is None:
                raise ValueError
            if stack_rnn_stack is None:
                raise ValueError
            if dropout is None:
                raise ValueError
            return get_stack_rnn_language_model(
                input_vocabulary_size=input_vocabulary_size,
                output_vocabulary_size=output_vocabulary_size,
                hidden_units=hidden_units,
                layers=num_layers,
                controller=stack_rnn_controller,
                stack=stack_rnn_stack,
                dropout=dropout,
                learned_hidden_state=True,
                use_padding=False,
                tag=(
                    'nondeterministic'
                    if stack_rnn_stack[0] in ('nondeterministic', 'vector-nondeterministic')
                    else None
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
        self.tag_kwargs = dict(
            nondeterministic=dict(
                block_size=get_einsum_block_size(args)
            )
        )

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
        if saver.kwargs['architecture'] in ('transformer', 'stack-transformer'):
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
        return model(
            model_input,
            include_first=not self.uses_bos,
            tag_kwargs=self.tag_kwargs
        )
