import pathlib
import sys

from rau.tasks.common.command import Command
from rau.tools.torch.saver import is_finished

class IsFinishedCommand(Command):

    DESCRIPTION = (
        'Tell whether training for a saved model is finished (exit code 0) or '
        'is incomplete and can be resumed (exit code 1).'
    )

    def add_arguments(self, parser):
        parser.add_argument('input', type=pathlib.Path,
            help='Directory containing a saved model.')

    def run(self, parser, args):
        sys.exit(int(not is_finished(args.input)))

if __name__ == '__main__':
    IsFinishedCommand(get_logger()).main()
