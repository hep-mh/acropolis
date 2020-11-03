# sys
from sys import stdout, stderr

# params
from .params import verbose, debug


def print_error(error, loc="", eol="\n"):
    locf = ""
    if debug == True and loc != "":
        locf = " \x1B[1;35m(" + loc + ")\x1B[0m"

    stderr.write("\x1B[1;31mERROR  \x1B[0m: " + error + " Stop!" + locf + eol)
    exit(1)


def print_warning(warning, loc="", eol="\n"):
    locf = ""
    if debug == True and loc != "":
        locf = " \x1B[1;35m(" + loc + ")\x1B[0m"

    stdout.write("\x1B[1;33mWARNING\x1B[0m: " + warning + locf + eol)

def print_info(info, loc="", eol="\n"):
    locf = ""
    if debug == True and loc != "":
        locf = " \x1B[1;35m(" + loc + ")\x1B[0m"

    if verbose:
        stdout.write("\x1B[1;32mINFO   \x1B[0m: " + info + locf + eol)
