# pythia8
import pythia8

# os/sys
import os; import sys
# contextlib
from contextlib import contextmanager


def _locate_cmnd_file():
    pkg_dir, _  = os.path.split(__file__)
    cmnd_file   = os.path.join(pkg_dir, "data", "pythia8.cmnd")

    return cmnd_file


@contextmanager
def suppressed_output(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()
        os.dup2(to.fileno(), fd)
        sys.stdout = os.fdopen(fd, 'w')

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


Particle = pythia8.Particle

Event    = pythia8.Event


class PythiaRunner(object):

    def __init__(self):
        # Create a new Pythia instance
        with suppressed_output():
            self._pythia = pythia8.Pythia()

        # Load the relevant settings
        self._pythia.readFile  ( _locate_cmnd_file() )

        with suppressed_output():
            self._pythia.init()


    def perform_shower():
        pass


PythiaRunner()
