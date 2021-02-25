# sys
from sys import stdout, stderr

# params
from acropolis.params import verbose, debug


def print_Yf(Yf, header=["mean", "high", "low"]):
    # If not verbose, simply print one line
    # including all abundances
    if not verbose:
        print(*Yf.transpose().reshape(1, Yf.size)[0,:])
        return

    # Fill potentially missing header entries
    NYf = Yf.shape[1]
    header.extend( [""] * ( NYf - len(header) ) )

    # Set small values to zero to guarantee the same
    # width for all abundances in the output
    Yf[Yf <= 1e-99] = 0

    # Define the set of all possible labels
    labels = ['n', 'p', 'H2', 'H3', 'He3', 'He4', 'Li6', 'Li7', 'Be7']

    # Print the header
    header_str = "\n{:^4}"
    for i in range(NYf):
        header_str  += " | \x1B[35m{:^11}\x1B[0m"

    print( header_str.format("", *header) )
    print("----------------------------------------------")

    # Print the different abundances
    for j, l in enumerate(labels):
        line = "\x1B[34m{:>4}\x1B[0m"
        for i in range(NYf):
            line += " | {:11.5e}"

        if l in ['n', 'H3', 'Be7']:
            line += "  [\x1B[36m{:7}\x1B[0m]"

        print( line.format(l, *Yf[j], 'decayed') )


def print_error(error, loc="", eol="\n"):
    locf = ""
    if debug == True and loc != "":
        locf = " \x1B[1;35m(" + loc + ")\x1B[0m"

    stderr.write("\x1B[1;31mERROR  \x1B[0m: " + error + locf + eol)
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
