# -*- coding: utf-8 -*-
import numpy as np
from scipy import constants

from . import interactionfunctions


KBOLTZ = constants.Boltzmann #J K^-1


def particle_deposition_rate(x,temp,dynamic_viscosity,
                     kinematic_viscosity,shear_velocity,
                     bturb,sct=1.0,
                     hamacker=None,
                     permittivity=None,
                     dl_thickness=None,
                     phi_dl=None,
                     vrepr=None,
                     adjustment_factor=1.0,
                     interactions=False):
    """
        Calculate deposition rate according to X theory

        :x: Crystal size (m^3)
        :temp: Temperature (K)
        :dynamic_viscosity: Dynamic viscosity (Pa s)
        :kinematic_viscosity: Kinematic viscosity (m^2 s^-1)
        :shear_velocity: Shear velocity (m/s)
        :hamacker: Hamacker constant (J)
        :permittivity: Permittivity of water (C V^-1 m^-1)
        :dl_thickness: Debye length (m)
        :phi_dl: Electric potential at surface of particle and wall (V)
        :vrepr: Representative volume for interactions (m^3)
        :b: Turbulent diffusivity constant (dimensionless)
        :sct: Turbulent Schmidt number (dimensionless)
        :interactions: Whether to consider interactions (default: True)
        returns deposition (m/s)
    """
    #Calculate without interaction
    if interactions:
        assert hamacker is not None
        assert permittivity is not None
        assert dl_thickness is not None
        assert phi_dl is not None
        assert vrepr is not None
    rrepr = (3/(4*np.pi)*vrepr)**(1./3) if vrepr is not None else None
    
    b = bturb
    r = (3/(4*np.pi)*x)**(1.0/3)
    dbr = (KBOLTZ*temp)/(6*np.pi*dynamic_viscosity*r)
    kappa = b*shear_velocity**3/(kinematic_viscosity**2*sct)
    d0 = 1/_integral2_0inf(dbr,kappa,r)

    #Calculate interactions
    wall_length = kinematic_viscosity/shear_velocity
    y_v = 5*wall_length
    if interactions:
        wd = interactionfunctions.wdeposition(
                rrepr,dynamic_viscosity,kinematic_viscosity,shear_velocity,
                temp,permittivity,phi_dl,dl_thickness,hamacker,
                y_v,b,sct=sct)
        deposition_rate = d0/(1+wd*d0)
    else:
        wd = 1.0
        deposition_rate = d0
    
    deposition_rate *= adjustment_factor
    return deposition_rate


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


