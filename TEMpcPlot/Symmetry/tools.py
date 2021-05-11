"""just few tools for crysttalgrapy
"""
import numpy as np


def gen_hkl(h0, h1, k0, k1, l0, l1):
    hkl = np.mgrid[h0: h1 + 1, k0: k1 + 1, l0: l1 + 1]
    return np.vstack(map(np.ravel, hkl)).T


def gen_hklrange(h0, k0, l0):
    return gen_hkl(h0, h0, k0, k0, l0, l0)
