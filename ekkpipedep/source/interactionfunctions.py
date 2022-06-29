#!/usr/bin/env python3
import functools
import warnings

import numpy as np
from scipy import constants
from scipy import optimize
from scipy import integrate


KBOLTZ = constants.Boltzmann #J K^-1
AVOG = constants.Avogadro #mol^-1
ECHARGE = constants.elementary_charge #C
INTEGRATION_ZERO = 1e-10

# =============================================================================
# MAIN INTERACTION FUNCTIONS
# =============================================================================
def wdeposition(r,dynamic_viscosity,kinematic_viscosity,shear_velocity,
            temp,permittivity,phi_dl,dl_thickness,hamacker,
            y_v,b=9.5*1e-4,sct=1.0):
    dbr = (KBOLTZ*temp)/(6*np.pi*dynamic_viscosity*r)
    kappa = b*shear_velocity**3/(kinematic_viscosity**2*sct)
    
    k = kappa/dbr
    potential_function = lambda y : _potential_wall(y,r,
                                                    hamacker,
                                                    permittivity,phi_dl,
                                                    dl_thickness)
    def inner_integrand(y):
        return potential_function(y)*_deriv_11xr3(y,k,r)\
                /(KBOLTZ*temp)
    def outer_integrand(y):
        inner_integral = 0 #Use approximation trick (works empirically)
        exp_term = potential_function(y)/(KBOLTZ*temp)/(1+k*(y+r)**3) - \
                    inner_integral
        integrand = _ghydro_wall(y,r)/dbr*(1/(1+k*(y+r)**3))*np.exp(exp_term)
        return integrand
    term1 = _warned_integration(outer_integrand,
                                        INTEGRATION_ZERO,
                                        y_v,
                                        1e60)
    term2 = -_integral2_0l(dbr,kappa,r,y_v)
#    term2 = -integral1_0l(dbr,kappa,y_v)
    wd = term1 + term2
    return wd

def wbrownian(r,hamacker,permittivity,phi_dl,dl_thickness,temp,
               low=1e-6,high=1e6):
    f = lambda s : _expify(_vectorize(_integrand_brownian))(s,r,hamacker,permittivity,phi_dl,dl_thickness,temp)
#    return integrate.quad(f,np.log(low),np.log(high))[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error",integrate.IntegrationWarning)
        try:
            integral = integrate.quad(f,np.log(low),np.log(high))[0]
        except:
            integral = np.inf
    return integral


def wturbulent(r,
                hamacker,permittivity,phi_dl,dl_thickness,temp,
                dynamic_viscosity,turb_dissipation,
                kinematic_viscosity,const_turb,
                low=1e-6,high=1e6):
    f = lambda s : _expify(_vectorize(_integrand_turbulent))(s,r,high,
                    hamacker,permittivity,phi_dl,dl_thickness,temp,
                    dynamic_viscosity,turb_dissipation,
                    kinematic_viscosity,const_turb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error",integrate.IntegrationWarning)
        try:
            integral = integrate.quad(f,np.log(low),np.log(high))[0]
        except:
            integral = np.inf
    return integral

# =============================================================================
# AUXILIARY DEPOSITION FUNCTIONS
# =============================================================================
def _integrand_wall(s,r,hamacker_d,permittivity,phi_dl,dl_thickness,temp,k):
    inner_integrand = 0.0
    potential = _potential_wall(s,r,hamacker_d,permittivity,phi_dl,dl_thickness)
    exp_factor = 1/(KBOLTZ*temp)*(potential/(1+k*s**3))
#     exp_factor = 0
    res = _ghydro_wall(s,r)*np.exp(exp_factor)/(1+k*s**3)
    return res


def _vdw_potential_wall(s,r,hamacker_d):
    sr = s/r
    res = -1/6*hamacker_d*(2*(sr+1)/(sr*(sr+2)) - np.log((sr+2)/sr))
#     res = res*(res < 0) #Avoid numerical errors
    return res

def _dl_potential_wall(s,r,permittivity,phi_dl,dl_thickness):
#    print(s)
#    print(phi_dl)
#    print(permittivity)
#    print(dl_thickness)
    if r/dl_thickness > 1:
        res = permittivity*r*phi_dl**2*np.log(1+np.exp(-s/dl_thickness))
    else:
        res = permittivity*r*phi_dl**2*np.log(1+np.exp(-s/dl_thickness))
#    print(res)
#    print('ok')
#     res = res*(res > 0) #Avoid numerical errors
    return res


def _potential_wall(s,r,hamacker_d,permittivity,phi_dl,dl_thickness):
    return _vdw_potential_wall(s,r,hamacker_d) + \
            _dl_potential_wall(s,r,permittivity,phi_dl,dl_thickness)


def _ghydro_wall(y,r):
#    return 1.0
    return 1 + 1/(y/r) + 0.128/np.sqrt(y/r)


def _integral_dtdb(k,l):
    #integral from 0 to l of (1/(1+kx^3))
    k13 = k**(1./3)
    k23 = k**(2./3)
    term1 = -np.log(np.abs(k23*l**2-k13*l+1))/(6*k13)
    term2 = np.log((k*l+k23)/k)/(3*k13)
    term3 = np.arctan((2*np.sqrt(3)*k13*l-np.sqrt(3))/3)/(np.sqrt(3)*k13)
    term4 = (2*np.log(k)+np.sqrt(3)*np.pi)/(18*k13)
    res = term1 + term2 + term3 + term4
    return res


# =============================================================================
# AUXILIARY COAGULATION FUNCTIONS
# =============================================================================
def _vdw_potential_tilde(s,hamacker): #Potential of s/r
    res = -1/6*hamacker*(2/(s**2+2*s) + 2/((s+2)**2) + np.log((s**2+2*s)/((s+2)**2)))
#    res = res*(res < 0) #Avoid numerical errors
    return res


def _dl_potential_tilde(s,r,permittivity,phi_dl,dl_thickness):
    if r/dl_thickness > 1:
        res = 1/2*permittivity*r*phi_dl**2*np.log(1+np.exp(-r/dl_thickness*s))
    else:
        res = 1/2*permittivity*r*phi_dl**2*np.log(1+np.exp(-r/dl_thickness*s))
#    res = res*(res > 0) #Avoid numerical errors
    return res


def _potential_tilde(s,r,hamacker,permittivity,phi_dl,dl_thickness): #Potential of s*(r+2)
    return _vdw_potential_tilde(s,hamacker) + \
           _dl_potential_tilde(s,r,permittivity,phi_dl,dl_thickness)


def _ghydro(s):
    return (6*s**2+4*s)/(6*s**2+13*s+2)


def _integrand_brownian(s,r,hamacker,permittivity,phi_dl,dl_thickness,temp):
    potential = _potential_tilde(s,r,hamacker,permittivity,phi_dl,dl_thickness)
    return 2/((s+2)**2*_ghydro(s))*np.exp(potential/(KBOLTZ*temp))


def _integrand_turbulent(s,r,maxval,hamacker,permittivity,phi_dl,dl_thickness,temp,
                        dynamic_viscosity,turb_dissipation,
                        kinematic_viscosity,const_turb):
    #
    #Pa s, J/m^3
    A = 0.5*const_turb*r*dynamic_viscosity*np.sqrt(turb_dissipation/kinematic_viscosity) #J/m^2
#     inner_integrand = lambda xi : expify(inner_integrand_turbulent)(xi,r)
#     second_potential_term = integrate.quad(inner_integrand,np.log(s),np.log(maxval))[0] #J/m^3
    
    second_potential_term = 0.0 #J #Same trick as in deposition
    first_potential_term = _potential_tilde(s,r,hamacker,permittivity,phi_dl,dl_thickness)/((s+2)**2) #J
    inner_term = 1/(A*r**2)*(first_potential_term - second_potential_term) #unitless
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = 24/((s+2)**4*_ghydro(s))*np.exp(inner_term) #unitless
        except:
            res = np.inf
    return res


# =============================================================================
# AUXILIARY INTEGRATION FUNCTIONS
# =============================================================================
def _vectorize(f):
    def g(s,*args):
        if type(s) == type(np.array(1.0)):
            return np.array([f(ss,*args) for ss in s.flatten()]).reshape(*s.shape)
        else:
            return f(s,*args)
    return g


def _expify(f):
    def g(s,*args,**kwargs):
        return np.exp(s)*f(np.exp(s),*args,**kwargs)
    return g

def _integral1_0l(a,b,l):
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


def _integral1_0inf(a,b):
    #integral from 0 to inf of (1/(a+bx^3))
    return 2*np.sqrt(3)*np.pi/(9*(a**2*b)**(1/3))


def _integral1_linf(a,b,l):
    #integral from l to inf of (1/(a+bx^3))
    return _integral1_0inf(a,b) - _integral1_0l(a,b,l)


def _integral1_lm(a,b,l,m):
    #integral from l to m of (1/(a+bx^3))
    return _integral1_0l(a,b,m) - _integral1_0l(a,b,l)


def _integral2_0l(a,b,r,l):
    #integral from 0 to l of (1/(a+b(x+r)^3))
    #l,l+r
    return _integral1_lm(a,b,l,l+r)


def _integral2_0inf(a,b,r):
    #integral from 0 to inf of (1/(a+b(x+r)^3))
    return _integral1_linf(a,b,r)


def _integral2_linf(a,b,r,l):
    #integral from l to inf of (1/(a+b(x+r)^3))
    return _integral1_linf(a,b,r+l)
    return _integral1_0inf(a,b) - _integral1_0l(a,b,l)


def _integral2_lm(a,b,r,l,m):
    #integral from l to m of (1/(a+b(x+r)^3))
    return _integral1_lm(a,b,l+r,m+r)


def _deriv_11xr3(y,k,r):
    #Derivative (in relation to y) of 
    #1/(1+k*(y+r)^3)
    return -3*k*(y+r)**2/((k*(y+r)**3+1)**2)


def _nonwarned_integration(integrand,a,b):
    return integrate.quad(integrand,
                          a,
                          b)[0]

    
def _warned_integration(integrand,a,b,error_integral):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error",integrate.IntegrationWarning)
        try:
            integral = integrate.quad(integrand,
                                      a,
                                      b)[0]
        except integrate.IntegrationWarning:
            integral = error_integral
    return integral
