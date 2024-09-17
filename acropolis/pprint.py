# sys
from sys import stdout, stderr, exit

# flags
import acropolis.flags as flags
# info
from acropolis.info import version, dev_version, url


_max_verbose_level = 1

_use_color = True


def print_version():
    if flags.verbose:
        # Differentiate between stable and dev version
        version_str = ""
        # Stable version
        if version == dev_version:
            version_str = "v{}".format(version)
        # Development version
        else:
            version_str = "v{} [dev]".format(dev_version)

        if _use_color:
            ctxt = "\x1B[38;5;209m"
            cend = "\x1B[0m"
        else:
            ctxt = cend = ""

        stdout.write( f"{ctxt}ACROPOLIS {version_str} ({url}){cend}\n\n" )


def print_Yf(Yf, header=["mean", "high", "low"]):
    # If not verbose, simply print one line
    # including all abundances
    if not flags.verbose:
        print(*Yf.transpose().reshape(1, Yf.size)[0,:])
        return

    # Define the colors
    if _use_color:
        chdr = "\x1B[35m"
        celm = "\x1B[34m"
        cdcy = "\x1B[36m"
        cend = "\x1B[0m"
    else:
        chdr = celm = cdcy = cend = ""

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
        header_str  += f" | {chdr}    {{:8}}{cend}"

    print( header_str.format("", *header) )
    print("-------------------------------------------------")

    # Print the different abundances
    for j, label in enumerate(labels):
        line = f"{celm}{{:>4}}{cend}"
        for i in range(NYf):
            line += " | {:11.6e}"

        if label in ['n', 'H3', 'Be7']:
            line += f"  [{cdcy}{{:7}}{cend}]"

        print( line.format(label, *Yf[j], 'decayed') )


def print_error(error, loc="", eol="\n", flush=False):
    # Define the colors
    if _use_color:
        cloc  = "\x1B[1;35m"
        ctyp  = "\x1B[1;31m"
        cend  = "\x1B[0m"
    else:
        cloc = ctyp = cend = ""

    locf = ""
    if flags.debug and loc != "":
        locf = f" {cloc}({loc}){cend}"

    stderr.write(f"{ctyp}ERROR  {cend}: {error}{locf}{eol}")
    
    if flush:
        stderr.flush()
    
    exit(1)


def print_warning(warning, loc="", eol="\n", flush=False):
    # Define the colors
    if _use_color:
        cloc  = "\x1B[1;35m"
        ctyp = "\x1B[1;33m"
        cend  = "\x1B[0m"
    else:
        cloc = ctyp = cend = ""
    
    locf = ""
    if flags.debug and loc != "":
        locf = f" {cloc}({loc}){cend}"

    stdout.write(f"{ctyp}WARNING{cend}: {warning}{locf}{eol}")
    
    if flush:
        stdout.flush()


def print_info(info, loc="", eol="\n", flush=False, verbose_level=None):
    global _max_verbose_level

    if verbose_level is None:
        verbose_level = _max_verbose_level

    _max_verbose_level = max( _max_verbose_level, verbose_level )

    # Define the colors
    if _use_color:
        cloc  = "\x1B[1;35m"
        ctyp = "\x1B[1;32m"
        cend  = "\x1B[0m"
    else:
        cloc = ctyp = cend = ""

    locf = ""
    if flags.debug and loc != "":
        locf = f" {cloc}({loc}){cend}"

    if flags.verbose and verbose_level >= _max_verbose_level:
        stdout.write(f"{ctyp}INFO   {cend}: {info}{locf}{eol}")
        
        if flush:
            stdout.flush()


def set_max_verbose_level(max_verbose_level=None):
    global _max_verbose_level

    if max_verbose_level is None:
        max_verbose_level = 1

    _max_verbose_level = max_verbose_level


def disable_color():
    global _use_color

    _use_color = False