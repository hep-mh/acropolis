# importlib
from importlib import import_module

# Load the compiled module dynamically
aot = import_module(".compiled", package="acropolis.aot")

# Expose the compiled functions
ph_rate_pair_creation_ae   = aot.ph_rate_pair_creation_ae
ph_kernel_inverse_compton  = aot.ph_kernel_inverse_compton
el_kernel_pair_creation_ae = aot.el_kernel_pair_creation_ae
el_rate_inverse_compton    = aot.el_rate_inverse_compton
el_kernel_inverse_compton  = aot.el_kernel_inverse_compton

dsdE_Z2                    = aot.dsdE_Z2

solve_cascade_equation     = aot.solve_cascade_equation

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
