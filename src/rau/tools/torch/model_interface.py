import pathlib
import random

import torch

from .saver import ModelSaver
from .init import smart_init

class ModelInterface:

    def __init__(self,
        use_load: bool = False,
        use_init: bool = False,
        use_continue: bool = False
    ) -> None:
        super().__init__()
        self.use_load = use_load
        self.use_init = use_init
        self.use_continue = use_continue
        self.parser = None
        self.device = None
        self.parameter_seed = None

    def add_arguments(self, parser):
        self.add_device_arguments(parser)
        if self.use_init or self.use_continue:
            parser.add_argument('--output', type=pathlib.Path, required=True,
                help='Output directory where logs and model parameters will '
                     'be saved.')
        if self.use_load:
            group = parser.add_argument_group('Load an existing model')
            self.add_load_arguments(group)
        if self.use_init:
            group = parser.add_argument_group('Initialize a new model')
            self.add_init_arguments(group)
        self.parser = parser

    def add_device_arguments(self, group):
        group.add_argument('--device',
            help='PyTorch device where the model will reside. Default is to '
                 'use cuda if available, otherwise cpu.')

    def add_load_arguments(self, group):
        group.add_argument('--load-model', type=pathlib.Path,
            help='Load a pre-existing model. The argument should be a '
                 'directory containing a model.')

    def add_init_arguments(self, group):
        group.add_argument('--parameter-seed',
            type=int,
            help='Random seed used to initialize the parameters of the model.')
        self.add_more_init_arguments(group)

    def add_more_init_arguments(self, group):
        pass

    def add_forward_arguments(self, parser):
        pass

    def get_device(self, args):
        if self.device is None:
            self.device = parse_device(args.device)
        return self.device

    def construct_model(self, **kwargs):
        raise NotImplementedError

    def get_kwargs(self, args, *_args, **kwargs):
        raise NotImplementedError

    def fail_argument_check(self, msg):
        self.parser.error(msg)

    def construct_saver(self, args, *_args, **_kwargs):
        device = self.get_device(args)
        if self.use_continue and args.continue_:
            if args.output is None:
                self.fail_argument_check('When --continue is used, --output is required.')
            saver = ModelSaver.read(
                self.construct_model,
                args.output,
                device=device,
                continue_=True
            )
        elif self.use_init and (not self.use_load or args.load_model is None):
            # Initialize a new model.
            if args.output is None:
                self.fail_argument_check('When initializing a new model, --output is required.')
            try:
                kwargs = self.get_kwargs(args, *_args, **_kwargs)
            except ValueError as e:
                self.fail_argument_check(e)
            # TODO Skip default parameter initialization.
            # See https://pytorch.org/tutorials/prototype/skip_param_init.html
            # TODO Allocate parameters directly on the device using a context manager.
            saver = ModelSaver.construct(self.construct_model, args.output, **kwargs)
            saver.check_output()
            saver.save_kwargs()
            saver.model.to(device)
            self.parameter_seed = args.parameter_seed
            if self.parameter_seed is None:
                self.parameter_seed = random.getrandbits(32)
            if device.type == 'cuda':
                torch.manual_seed(self.parameter_seed)
                param_generator = None
            else:
                param_generator = torch.manual_seed(self.parameter_seed)
            self.initialize(args, saver.model, param_generator)
        else:
            # Load an existing model.
            if args.load_model is None:
                self.fail_argument_check('Argument --load-model is missing.')
            saver = ModelSaver.read(
                self.construct_model,
                args.load_model,
                device=device,
                continue_=False
            )
        self.on_saver_constructed(args, saver)
        return saver

    def initialize(self, args, model, generator):
        smart_init(model, generator)

    def on_saver_constructed(self, args, saver):
        pass

def parse_device(s):
    return torch.device(_get_device_str(s))

def _get_device_str(s):
    if s is None:
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    else:
        return s
