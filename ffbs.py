from scipy.stats import norm
from Hamilton import *


def ffbs(y, y_est, sigma, K, P, rng, multi=False, nreg=0):
    if multi:
        clkl = np.array([[norm.logpdf(y[l], y_est[k, l], sigma[k, l] ** .5) for l in range(nreg)] for k in range(K)])
        clkl = clkl.sum(axis=1)
    else:
        clkl = np.array([norm.logpdf(y, y_est[_], sigma[_] ** .5) for _ in range(K)])

    filtp, smooths = Hamilton(P, clkl, rng)    # To do: add smoothed probability

    return filtp, smooths
