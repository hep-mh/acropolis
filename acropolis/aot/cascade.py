# importlib
from importlib import import_module

# Load the compiled module dynamically
_aot = import_module(".compiled", package="acropolis.aot")

# Expose the compiled functions
ph_rate_pair_creation_ae   = _aot.ph_rate_pair_creation_ae
ph_kernel_inverse_compton  = _aot.ph_kernel_inverse_compton
el_kernel_pair_creation_ae = _aot.el_kernel_pair_creation_ae
el_rate_inverse_compton    = _aot.el_rate_inverse_compton
el_kernel_inverse_compton  = _aot.el_kernel_inverse_compton

dsdE_Z2                    = _aot.dsdE_Z2

solve_cascade_equation     = _aot.solve_cascade_equation

# Specify which functions to export
__all__ = [
    "ph_rate_pair_creation_ae",
    "ph_kernel_inverse_compton",
    "el_kernel_pair_creation_ae",
    "el_rate_inverse_compton",
    "el_kernel_inverse_compton",
    "dsdE_Z2",
    "solve_cascade_equation"
]
