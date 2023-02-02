# tempfile
import tempfile

# All .lhe files are written to tempdir
# unless explicitly stated otherwise
_tempdir = tempfile.gettempdir()


class LheEvent(object):
    pass


def write_lhe_file(filename, events):
    pass