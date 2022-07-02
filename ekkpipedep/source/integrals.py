# -*- coding: utf-8 -*-
import numpy as np


def integral1_0l(a,b,l):
    #integral from 0 to l of (1/(a+bx^3))
    #1/(a+bx^3) = 1/a*1/(1+(b/a)*x^3)
    #int(1/(a+bx^3)) = 1/a*int(1/(1+(b/a)*x^3))
    k = b/a
    k13 = k**(1./3)
    k23 = k**(2./3)
    term1 = -np.log(np.abs(k23*l**2-k13*l+1))/(6*k13)
    term2 = np.log((k*l+k23)/k)/(3*k13)
    term3 = np.arctan((2*np.sqrt(3)*k13*l-np.sqrt(3))/3)/(np.sqrt(3)*k13)
    term4 = (2*np.log(k)+np.sqrt(3)*np.pi)/(18*k13)
    res = term1 + term2 + term3 + term4
    res = res/a
    return res


def integral1_0inf(a,b):
    #integral from 0 to inf of (1/(a+bx^3))
    return 2*np.sqrt(3)*np.pi/(9*(a**2*b)**(1/3))


def integral1_linf(a,b,l):
    #integral from l to inf of (1/(a+bx^3))
    return integral1_0inf(a,b) - integral1_0l(a,b,l)


def integral1_lm(a,b,l,m):
    #integral from l to m of (1/(a+bx^3))
    return integral1_0l(a,b,m) - integral1_0l(a,b,l)


def integral2_0l(a,b,r,l):
    #integral from 0 to l of (1/(a+b(x+r)^3))
    #l,l+r
    return integral1_lm(a, b, r, r+l)


def integral2_0inf(a,b,r):
    #integral from 0 to inf of (1/(a+b(x+r)^3))
    return integral1_linf(a, b, r)


def integral2_linf(a, b, r, l):
    #integral from l to inf of (1/(a+b(x+r)^3))
    return integral1_linf(a, b, r+l)


def integral2_lm(a, b, r, l, m):
    #integral from l to m of (1/(a+b(x+r)^3))
    return integral1_lm(a, b, l+r, m+r)


def deriv_11xr3(y,k,r):
    #Derivative (in relation to y) of 
    #1/(1+k*(y+r)^3)
    return -3*k*(y+r)**2/((k*(y+r)**3+1)**2)