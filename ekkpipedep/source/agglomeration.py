# -*- coding: utf-8 -*-
import warnings

import numpy as np
from scipy import constants

from . import interactionfunctions


KBOLTZ = constants.Boltzmann #J K^-1


def agglomeration_rate(x,y,
                      temp,dynamic_viscosity,
                      kinematic_viscosity,
                      turbulent_dissipation,
                      const_turb,
                      komolgorov_length,
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
        :y: Crystal size (m^3)
        :temp: Temperature (K)
        :dynamic_viscosity: Dynamic viscosity (Pa s)
        :kinematic_viscosity: Kinematic viscosity (m^2 s^-1)
        :turbulent_dissipation: Turbulent dissipation (J/kg/s)
        :komolgorov_length: Komolgorov length (m)
        :hamacker: Hamacker constant (J)
        :permittivity: Permittivity of water (C V^-1 m^-1)
        :dl_thickness: Debye length (m)
        :phi_dl: Electric potential at surface of particle and wall (V)
        :vrepr: Representative volume for interactions (m^3)
        :sct: Turbulent Schmidt number (dimensionless)
        :interactions: Whether to consider interactions (default: False)
        returns coagulation kernel (m^3/s)
    """
    if interactions:
        assert hamacker is not None
        assert permittivity is not None
        assert dl_thickness is not None
        assert phi_dl is not None
        assert vrepr is not None
    rrepr = (3/(4*np.pi)*vrepr)**(1./3) if vrepr is not None else None
    c_brownian = 2.*KBOLTZ*temp/(3.*dynamic_viscosity)
    rate_brownian = c_brownian*(2 + (x/y)**(1./3) + (y/x)**(1./3))
    #turbulent coagulation
    c_turbulent = 3*const_turb*\
        np.sqrt(turbulent_dissipation/kinematic_viscosity)/(4*np.pi)
    rate_turbulent = c_turbulent*(x**(1./3) + y**(1./3))**3.0
    dx,dy = 2*(3/(4*np.pi)*x)**(1.0/3),2*(3/(4*np.pi)*y**(1.0/3))
    soft_constraint = sigmoid(-2*dx/komolgorov_length)*\
                      sigmoid(-2*dy/komolgorov_length)
    rate_turbulent *= soft_constraint
    iwbr = 1.0
    iwt = 1.0
    if interactions:
        wbr = interactionfunctions.wbrownian(rrepr,hamacker,permittivity,phi_dl,dl_thickness,temp)
        wt = interactionfunctions.wturbulent(rrepr,hamacker,permittivity,phi_dl,dl_thickness,temp,
                    dynamic_viscosity,turbulent_dissipation,
                    kinematic_viscosity,const_turb)
        if wbr <= 1e-4 or wt <= 1e-4:
            warnings.warn("Some error in integral calculations. Considering no interaction")
            iwbr = 1.0
            iwt = 1.0
        else:
            iwbr = 1/wbr
            iwt = 1/wt
    else:
        iwbr = 1
        iwt = 1
    rate = iwbr*rate_brownian + iwt*rate_turbulent
    rate = adjustment_factor*rate
    return rate


def sigmoid(x):
    return 1/(1+np.exp(-x))