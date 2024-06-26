import humanfriendly
import torch_semiring_einsum

HELP_MESSAGE = (
    'This tunes the time-memory tradeoff for the nondeterministic stack. '
    'Higher values generally take less time but more memory. '
    'See https://bdusell.github.io/semiring-einsum/'
)

def add_einsum_forward_arguments(group):
    group.add_argument('--einsum-block-size', type=int,
        help='(stack-transformer) Block size used for einsum operations. ' +
             HELP_MESSAGE)
    group.add_argument('--einsum-max-memory', type=humanfriendly.parse_size,
        help='(stack-transformer) Maximum CUDA memory used for einsum '
             'operations. Units such as MB, GB, etc. can be used. ' +
             HELP_MESSAGE)

def get_einsum_block_size(args):
    if hasattr(args, 'einsum_block_size') and args.einsum_block_size is not None:
        return args.einsum_block_size
    else:
        return torch_semiring_einsum.AutomaticBlockSize(
            max_cuda_bytes=getattr(args, 'einsum_max_memory', None)
        )
