# -*- coding: utf-8 -*-

#import numpy as np


# def uniform(n, m0, xmin, xmax):
#     k = np.arange(n)
#     moments = m0 * (xmax**(k+1) - xmin**(k+1)) / ((k+1) * (xmax - xmin))
#     return np.array(moments)


# def lognormal(n, m0, mu, sig):
#     k = np.arange(n)
#     moments = m0 * np.exp(k * mu + 0.5 * (sig * k)**2)
#     return np.array(moments)


# def delta(n, m0, x):
#     moments = [m0 * x**k for k in np.arange(n)]
#     return np.array(moments)