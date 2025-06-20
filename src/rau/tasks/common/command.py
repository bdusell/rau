import argparse
import logging
import sys

class Command:
    r"""Base class for commands in the `rau` CLI. Commands should inherit from
    this class, define `DESCRIPTION`, and implement the `add_arguments` and
    `main` methods.
    """

    DESCRIPTION: str
    r"""A description of the command. This should be set by subclasses to
    provide a brief overview of what the command does.
    """

    def description(self) -> str:
        r"""Return the description of the command. This is used to provide help
        information in the CLI.
        """
        return self.DESCRIPTION

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        r"""Add command-specific arguments to the parser."""
        raise NotImplementedError

    def run(self, parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
        r"""Main entry point for the command."""
        raise NotImplementedError

    def main(self):
        r"""Run the command as an independent program."""
        parser = argparse.ArgumentParser(description=self.DESCRIPTION)
        self.add_arguments(parser)
        args = parser.parse_args()
        self.run(parser, args)

def get_logger() -> logging.Logger:
    console_logger = logging.getLogger('main')
    console_logger.addHandler(logging.StreamHandler(sys.stdout))
    console_logger.setLevel(logging.INFO)
    return console_logger
