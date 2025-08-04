import torch

from rau.models.stack_nn.differentiable_stacks.semiring import (
    log,
    log_viterbi
)
from rau.models.stack_nn.differentiable_stacks.nondeterministic import (
    NondeterministicStack,
    logits_to_actions,
    ViterbiDecoder
)

from .stack_rnn import StackRNN, StackRNNController, ReadingLayerSizes

class NondeterministicStackRNN(StackRNN):

    def __init__(self,
        input_size: int,
        num_states: int,
        stack_alphabet_size: int,
        controller: StackRNNController,
        controller_output_size: int,
        include_reading_in_output: bool,
        normalize_transition_weights: bool = False,
        include_states_in_reading: bool = True,
        normalize_reading: bool = True,
        reading_layer_sizes: ReadingLayerSizes = None,
        stack_reading_size: int | None = None
    ) -> None:
        Q = num_states
        S = stack_alphabet_size
        if stack_reading_size is None:
            if include_states_in_reading:
                stack_reading_size = Q * S
            else:
                stack_reading_size = S
        super().__init__(
            input_size=input_size,
            stack_reading_size=stack_reading_size,
            controller=controller,
            controller_output_size=controller_output_size,
            include_reading_in_output=include_reading_in_output,
            reading_layer_sizes=reading_layer_sizes
        )
        self.num_states = num_states
        self.stack_alphabet_size = stack_alphabet_size
        self.normalize_transition_weights = normalize_transition_weights
        self.include_states_in_reading = include_states_in_reading
        self.normalize_reading = normalize_reading
        self.num_op_rows = Q * S
        self.num_op_cols = Q * S + Q * S + Q
        self.operation_layer = torch.nn.Linear(
            controller_output_size,
            self.num_op_rows * self.num_op_cols
        )

    def operation_log_scores(self, hidden_state):
        # flat_logits : B x ((Q * S) * (Q * S + Q * S + Q))
        flat_logits = self.operation_layer(hidden_state)
        return logits_to_actions(
            flat_logits,
            self.num_states,
            self.stack_alphabet_size,
            self.normalize_transition_weights
        )

    def initial_stack(self, batch_size, sequence_length, block_size, semiring=log):
        return self.get_new_stack(
            batch_size=batch_size,
            sequence_length=sequence_length,
            semiring=semiring,
            block_size=block_size
        )

    def get_new_stack(self, **kwargs):
        """Construct a new instance of the stack data structure."""
        return self.get_new_viterbi_stack(**kwargs)

    def get_new_viterbi_stack(self, batch_size, sequence_length, semiring, block_size):
        """Construct a new instance of the stack data structure, but ensure
        that it is a version that is compatible with Viterbi decoding."""
        t = next(self.parameters())
        # If the stack reading is not included in the output, then the last
        # timestep is not needed.
        if not self.include_reading_in_output and sequence_length is not None:
            sequence_length -= 1
        return NondeterministicStack.new_empty(
            batch_size=batch_size,
            num_states=self.num_states,
            stack_alphabet_size=self.stack_alphabet_size,
            sequence_length=sequence_length,
            include_states_in_reading=self.include_states_in_reading,
            normalize_reading=self.normalize_reading,
            block_size=block_size,
            dtype=t.dtype,
            device=t.device,
            semiring=semiring
        )

    class State(StackRNN.State):

        def compute_stack(self, hidden_state, stack):
            push, repl, pop = actions = self.rnn.operation_log_scores(hidden_state)
            stack.update(push, repl, pop)
            return stack, actions

    def viterbi_decoder(self, input_sequence, block_size, wrapper=None):
        """Return an object that can be used to run the Viterbi algorithm on
        the stack WFA and get the best run leading up to any timestep.

        If timesteps past a certain timestep will not be used, simply slice
        the input accordingly to save computation."""
        # This allows the model to work when wrapped by RNN wrappers.
        if wrapper is not None:
            input_sequence = wrapper.wrap_input(input_sequence)
        # TODO For the limited nondeterministic stack RNN, it may be useful to
        # implement a version of this that splits the input into chunks to use
        # less memory and work on longer sequences.
        with torch.inference_mode():
            result = self(
                input_sequence,
                block_size=block_size,
                return_state=False,
                include_first=False,
                return_actions=True
            )
            operation_weights, = result.extra_outputs
        # Since include_first is False, operation weights starts at timestep 1.
        # Remove any operation weights that are set to None at the end because
        # they were not needed.
        if operation_weights:
            if operation_weights[-1] is None:
                operation_weights.pop()
        return self.viterbi_decoder_from_operation_weights(operation_weights, block_size)

    def viterbi_decoder_from_operation_weights(self, operation_weights, block_size):
        # operation_weights[0] corresponds to the action weights computed just
        # after timestep j = 1 and before j = 2.
        if not self.include_states_in_reading:
            raise NotImplementedError
        with torch.no_grad():
            batch_size = operation_weights[0][0].size(0)
            sequence_length = len(operation_weights) + 1
            # Compute the gamma and alpha tensor for every timestep in the
            # Viterbi semiring.
            stack = self.get_new_viterbi_stack(
                batch_size=batch_size,
                sequence_length=sequence_length,
                semiring=log_viterbi,
                block_size=block_size
            )
            # The first `result` returned from `update` corresponds to timestep
            # j = 1, so these lists include results starting just before
            # timestep j = 2.
            alpha_columns = []
            gamma_j_nodes = []
            alpha_j_nodes = []
            for push, repl, pop in operation_weights:
                result = stack.update(
                    log_viterbi.primitive(push),
                    log_viterbi.primitive(repl),
                    log_viterbi.primitive(pop)
                )
                # Save the nodes for the columns of gamma and alpha in lists.
                # This makes decoding simpler.
                alpha_columns.append(result.alpha_j)
                gamma_j_nodes.append(result.gamma_j[1])
                alpha_j_nodes.append(result.alpha_j[1])
        return self.get_viterbi_decoder(alpha_columns, gamma_j_nodes, alpha_j_nodes)

    def get_viterbi_decoder(self, alpha_columns, gamma_j_nodes, alpha_j_nodes):
        return ViterbiDecoder(
            alpha_columns,
            gamma_j_nodes,
            alpha_j_nodes
        )
