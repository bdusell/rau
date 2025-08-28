import contextlib
import dataclasses
import json
import os
import pathlib
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager
from typing import Any, IO

import torch

from rau.tools.logging import FileLogger, LogEvent, read_log_file

@dataclasses.dataclass
class ModelSaver:

    model: torch.nn.Module
    kwargs: dict[str, Any]
    saved_kwargs: bool
    directory: pathlib.Path
    created_directory: bool
    is_read_only: bool
    append_to_logs: bool

    def _ensure_directory_created(self) -> None:
        if not self.created_directory:
            self._ensure_writable(lambda: f'create model directory {self.directory}')
            try:
                self.directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                raise DirectoryExists(
                    f'the model directory {self.directory} already exists'
                )
            self.created_directory = True

    def _ensure_writable(self, msg: Callable[[], str]) -> None:
        if self.is_read_only:
            raise ValueError(
                f'cannot perform the following action because this model saver '
                f'is read-only: {msg()}'
            )

    def check_output(self) -> None:
        self._ensure_directory_created()

    @property
    def kwargs_file(self) -> pathlib.Path:
        return get_kwargs_file(self.directory)

    def save_kwargs(self) -> None:
        self._ensure_writable(lambda: f'save kwargs to {self.kwargs_file}')
        self._ensure_directory_created()
        with self.kwargs_file.open('x') as fout:
            write_json(fout, self.kwargs)

    @property
    def parameters_file(self) -> pathlib.Path:
        return get_parameters_file(self.directory)

    def save_parameters(self) -> None:
        self._ensure_writable(lambda: f'save model parameters to {self.parameters_file}')
        self._ensure_directory_created()
        torch.save(self.model.state_dict(), self.parameters_file)

    @property
    def log_file(self) -> pathlib.Path:
        return get_log_file(self.directory)

    @contextlib.contextmanager
    def logger(self,
        flush: bool = False,
        reopen: bool = False
    ) -> Generator[FileLogger, None, None]:
        self._ensure_writable(lambda: f'start writing logs to {self.log_file}')
        self._ensure_directory_created()
        # Unless we are supposed to append to the log file, attempt to open the
        # log file in *exclusive* mode so the operation will fail early if the
        # log file already exists.
        mode = 'a' if self.append_to_logs else 'x'
        with self.log_file.open(mode) as fout:
            logger = FileLogger(fout, flush=flush, reopen=reopen)
            try:
                yield logger
            except KeyboardInterrupt:
                # Log an event if the program was interrupted.
                logger.log('keyboard_interrupt')
                raise
            except Exception as e:
                # Log other kinds of exceptions.
                try:
                    e_str = str(e)
                except Exception:
                    e_str = str(type(e))
                logger.log('exception', { 'exception' : e_str })
                raise

    def logs(self) -> AbstractContextManager[Iterable[LogEvent]]:
        return read_logs(self.directory)

    @property
    def temporary_checkpoint_file(self) -> pathlib.Path:
        return self.directory / 'temp-checkpoint.pt'

    @property
    def checkpoint_file(self) -> pathlib.Path:
        return self.directory / 'checkpoint.pt'

    @property
    def checkpoint_lock_file(self) -> pathlib.Path:
        return self.directory / 'checkpoint.lock'

    def save_checkpoint(self, metadata: Any) -> None:
        self._ensure_writable(lambda: f'save checkpoint to {self.temporary_checkpoint_file}')
        # Make sure that the saved checkpoint, if there is one, is in a
        # predictable state.
        self.heal_checkpoint()
        # Write the checkpoint to a temporary file.
        with self.temporary_checkpoint_file.open('xb') as fout:
            torch.save(dict(
                parameters=self.model.state_dict(),
                metadata=metadata
            ), fout)
            fout.flush()
            os.fsync(fout.fileno())
        # Create a lock file while the temporary file is moved.
        with self.checkpoint_lock_file.open('w') as fout:
            os.fsync(fout.fileno())
        # Overwrite the previous checkpoint with the temporary checkpoint.
        self.temporary_checkpoint_file.replace(self.checkpoint_file)
        # Remove the lock file.
        self.checkpoint_lock_file.unlink()

    def load_checkpoint(self, device: torch.device) -> Any:
        self.heal_checkpoint()
        data = torch.load(
            self.checkpoint_file,
            map_location=device,
            weights_only=False
        )
        self.model.load_state_dict(data['parameters'])
        self.model.to(device)
        return data['metadata']

    def heal_checkpoint(self) -> None:
        self._ensure_writable(lambda: f'heal checkpoint')
        if self.checkpoint_lock_file.exists():
            # If the lock file exists, it means that the move did not complete.
            # Finish it now.
            if self.temporary_checkpoint_file.exists():
                self.temporary_checkpoint_file.replace(self.checkpoint_file)
            self.checkpoint_lock_file.unlink()
        else:
            # If a temporary file exists, assume the write was incomplete and
            # delete it.
            self.temporary_checkpoint_file.unlink(missing_ok=True)

    def delete_checkpoint(self) -> None:
        self._ensure_writable(lambda: f'delete checkpoint')
        self.checkpoint_file.unlink(missing_ok=True)

    @staticmethod
    def construct(
        model_constructor: Callable[..., torch.nn.Module],
        directory: pathlib.Path,
        **kwargs: Any
    ) -> 'ModelSaver':
        model = model_constructor(**kwargs)
        return ModelSaver.from_model(model, directory, **kwargs)

    def from_model(
        model: torch.nn.Module,
        directory: pathlib.Path,
        **kwargs: Any
    ) -> 'ModelSaver':
        return ModelSaver(
            model=model,
            kwargs=kwargs,
            saved_kwargs=False,
            directory=directory,
            created_directory=False,
            is_read_only=False,
            append_to_logs=False
        )

    @staticmethod
    def read(
        model_constructor: Callable[..., torch.nn.Module],
        directory: pathlib.Path,
        device: torch.device | None = None,
        continue_: bool = False
    ) -> 'ModelSaver':
        if not directory.exists():
            raise ValueError(f'model directory does not exist: {directory}')
        kwargs = read_kwargs(directory)
        # TODO Skip default parameter initialization.
        # TODO Initialize tensors on the final device.
        model = model_constructor(**kwargs)
        if not continue_:
            load_kwargs = {}
            if device is not None:
                load_kwargs['map_location'] = device
            model.load_state_dict(torch.load(get_parameters_file(directory), **load_kwargs))
            if device is not None:
                model.to(device)
        return ModelSaver(
            model=model,
            kwargs=kwargs,
            saved_kwargs=True,
            directory=directory,
            created_directory=True,
            is_read_only=not continue_,
            append_to_logs=continue_
        )

def get_kwargs_file(directory: pathlib.Path) -> pathlib.Path:
    return directory / 'kwargs.json'

def read_kwargs(directory: pathlib.Path) -> dict[str, Any]:
    with get_kwargs_file(directory).open() as fin:
        return json.load(fin)

def get_parameters_file(directory: pathlib.Path) -> pathlib.Path:
    return directory / 'parameters.pt'

def get_log_file(directory: pathlib.Path) -> pathlib.Path:
    return directory / 'logs.log'

@contextlib.contextmanager
def read_logs(directory) -> Generator[Iterable[LogEvent], None, None]:
    with get_log_file(directory).open() as fin:
        yield read_log_file(fin)

def write_json(fout: IO, data: dict[str, Any]) -> None:
    json.dump(data, fout, indent=2, sort_keys=True)

class DirectoryExists(RuntimeError):
    pass
