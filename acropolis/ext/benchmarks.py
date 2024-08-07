# functools
from functools import partial

# params
from acropolis.params import me2
from acropolis.params import pi

# ext.models
from acropolis.ext.models import ResonanceModel, estimate_tempkd_ee


# Benchmark (1)
def sigma_ee_b1(s, mchi, delta, gammad, gammav):
    mx2 = mchi**2.
    mx4 = mchi**4.

    pref = 16. * pi * gammad * gammav / mx4 / (2. + delta)**4. / s**2.
    # -->
    return pref * mx2 * ( s*s - 2.*s*(mx2 - 3.*me2) + (me2 - mx2)**2. )


# Benchmark (2)
def sigma_ee_b2(s, mchi, delta, gammad, gammav):
    mx2 = mchi**2.
    mx4 = mchi**4.

    me4 = me2*me2

    pref = 16. * pi * gammad * gammav / mx4 / (2. + delta)**4. / s**2.
    # -->
    #return 3. * pref * ( 2.*(mx2 - me2)**4. - 8.*(mx4 - me4)*(mx2 - me2)*s \
    #       + (15.*mx4 + 26.*mx2*me2 + 15.*me4)*s*s - 14.*(mx2 + me2)*(s**3.) + 5.*(s**4.) ) / s
    return .25* pref * ( 4.*(s**3.) - 10.*(s**2.)*(mx2 + me2) + s*(9.*mx4 + 22.*mx2*me2 + 9.*me4) - 4.*(mx4 - me4)*(mx2 - me2) + (mx2 - me2)**4./s )


# Benchmark (3)
def sigma_ee_b3(s, mchi, delta, gammad, gammav):
    mx2 = pow(mchi, 2.)
    mx4 = pow(mchi, 4.)

    me4 = me2*me2

    pref = 16. * pi * gammad * gammav / mx4 / (2. + delta)**4. / s**2.
    # -->
    #return 48. * pref * ( mx4 * (me2 + 3.*s) + (me2 - s)**2. * (me2 + 3.*s) - 2.*mx2 * (me2**2. - 4.*me2*s + 3.*s*s) )
    return 4.5 * pref * ( mx4*(me2 + s) - 2.*mx2*(me2 - s)**2. + (me4 - s**2.)*(me2 - s) )


# -->
BenchmarkModel1 = partial(ResonanceModel, nd = 0., S = 1./2., tempkd = partial(estimate_tempkd_ee, sigma_ee=sigma_ee_b1)) # (1)
BenchmarkModel2 = partial(ResonanceModel, nd = 0., S = 3./4., tempkd = partial(estimate_tempkd_ee, sigma_ee=sigma_ee_b2)) # (2)
BenchmarkModel3 = partial(ResonanceModel, nd = 1., S = 3./2., tempkd = partial(estimate_tempkd_ee, sigma_ee=sigma_ee_b3)) # (3)