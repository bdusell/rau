import argparse
import pathlib
import random
from typing import Any

import torch

from .saver import ModelSaver
from .init import smart_init

class ModelInterface:

    _use_load: bool
    _use_init: bool
    _use_continue: bool
    parser: argparse.ArgumentParser | None
    device: torch.device | None
    parameter_seed: int | None

    def __init__(self,
        use_load: bool = False,
        use_init: bool = False,
        use_continue: bool = False
    ) -> None:
        super().__init__()
        self._use_load = use_load
        self._use_init = use_init
        self._use_continue = use_continue
        self.parser = None
        self.device = None
        self.parameter_seed = None

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self.add_device_arguments(parser)
        if self._use_init or self._use_continue:
            parser.add_argument('--output', type=pathlib.Path, required=True,
                help='Output directory where logs and model parameters will '
                     'be saved.')
        if self._use_load:
            group = parser.add_argument_group('Load an existing model')
            self.add_load_arguments(group)
        if self._use_init:
            group = parser.add_argument_group('Initialize a new model')
            self.add_init_arguments(group)
        self.parser = parser

    def add_device_arguments(self, group: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
        group.add_argument('--device',
            help='PyTorch device where the model will reside. Default is to '
                 'use cuda if available, otherwise cpu.')

    def add_load_arguments(self, group: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
        group.add_argument('--load-model', type=pathlib.Path,
            help='Load a pre-existing model. The argument should be a '
                 'directory containing a model.')

    def add_init_arguments(self, group: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
        group.add_argument('--parameter-seed',
            type=int,
            help='Random seed used to initialize the parameters of the model.')
        self.add_more_init_arguments(group)

    def add_more_init_arguments(self, group: argparse.ArgumentParser | argparse._ArgumentGroup) -> None:
        pass

    def add_forward_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def get_device(self, args: argparse.Namespace) -> torch.device:
        if self.device is None:
            self.device = parse_device(args.device)
        return self.device

    def construct_model(self, **kwargs: Any) -> torch.nn.Module:
        raise NotImplementedError

    def get_kwargs(self, args: argparse.Namespace, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def fail_argument_check(self, msg: str) -> None:
        if self.parser is None:
            raise ValueError('add_arguments() must be called first')
        self.parser.error(msg)

    def construct_saver(self, args: argparse.Namespace, *_args: Any, **_kwargs: Any) -> ModelSaver:
        device = self.get_device(args)
        if self._use_continue and args.continue_:
            if args.output is None:
                self.fail_argument_check('When --continue is used, --output is required.')
            saver = ModelSaver.read(
                self.construct_model,
                args.output,
                device=device,
                continue_=True
            )
        elif self._use_init and (not self._use_load or args.load_model is None):
            # Initialize a new model.
            if args.output is None:
                self.fail_argument_check('When initializing a new model, --output is required.')
            try:
                kwargs = self.get_kwargs(args, *_args, **_kwargs)
            except ValueError as e:
                self.fail_argument_check(str(e))
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

    def initialize(self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        generator: torch.Generator | None
    ) -> None:
        smart_init(model, generator)

    def on_saver_constructed(self,
        args: argparse.Namespace,
        saver: ModelSaver
    ) -> None:
        pass

def parse_device(s: str | None) -> torch.device:
    return torch.device(_get_device_str(s))

def _get_device_str(s: str | None) -> str:
    if s is None:
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    else:
        return s
