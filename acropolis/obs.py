class AbundanceObservation(object):

    def __init__(self, mean, err):
        self.mean = mean
        self.err = err


pdg2020 = {
    "Yp" : AbundanceObservation( 2.45e-1,  0.03e-1),
    "DH" : AbundanceObservation(2.547e-5, 0.035e-5),
    "HeD": AbundanceObservation(  8.3e-1,   1.5e-1),
    "LiH": AbundanceObservation( 1.6e-10,  0.3e-10)
}
