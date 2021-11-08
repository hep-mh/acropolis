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


class Particle(object):

    def __init__(self):
        pass


class Event(object):

    def __init__(self):
        pass


class PythiaRunner(object):

    def __init__(self, lhe_file, num):
        # Create a new Pythia instance
        with suppressed_output():
            self._sPythiaInstance = pythia8.Pythia()

        # Load the relevant settings
        self._sPythiaInstance.readFile  ( _locate_cmnd_file()        )
        self._sPythiaInstance.readString( "Beams:LHEF = " + lhe_file )

        with suppressed_output():
            self._sPythiaInstance.init()

        # Store the number of events in the LHE file
        self._sN = num


    def perform_shower():
        # Loop over each event
        for i in range(self._sN):
            with suppressed_output():
                pythia_next = self._sPythiaInstance.next()
            if not pythia_next:
                pass # TODO

            # TODO Print percentage

            # Loop over each particle in the current event
            for particle in self._sPythiaInstance.event:
                if not particle.isFinal():
                    continue


PythiaRunner("foobar", 10)
