class AbundanceObservation(object):

    def __init__(self, mean, err):
        self.mean = mean
        self.err = err


# 2020 ####################################################
pdg2020 = {
    "Yp" : AbundanceObservation( 2.45e-1,  0.03e-1),
    "DH" : AbundanceObservation(2.547e-5, 0.035e-5),
    "HeD": AbundanceObservation(  8.3e-1,   1.5e-1),
    "LiH": AbundanceObservation( 1.6e-10,  0.3e-10)
}


# 2021 ####################################################
pdg2021       = pdg2020.copy()
pdg2021["DH"] = AbundanceObservation(2.547e-5, 0.025e-5)
# Smaller error on D/H compared to 2020


# 2022 ####################################################
pdg2022 = pdg2021.copy()
# No change compared to 2021


# 2023 ####################################################
pdg2023       = pdg2022.copy()
pdg2023["DH"] = AbundanceObservation(2.547e-5, 0.029e-5)
# Larger error on D/H compared to 2022


# 2024 ####################################################
pdg2024 = pdg2023
# No change compared to 2023


# MOST RECENT #############################################
pdg = pdg2024
