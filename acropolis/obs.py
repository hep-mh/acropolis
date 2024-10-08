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
pdg2021 = {
    "Yp" : AbundanceObservation( 2.45e-1,  0.03e-1),
    "DH" : AbundanceObservation(2.547e-5, 0.025e-5),
    "HeD": AbundanceObservation(  8.3e-1,   1.5e-1),
    "LiH": AbundanceObservation( 1.6e-10,  0.3e-10)
}
# Smaller error on DH compared to 2020

# 2022 ####################################################
pdg2022 = pdg2021
# No change compared to 2021