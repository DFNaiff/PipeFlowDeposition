# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
from scipy import constants

from . import interactionfunctions
from . import integrals


KBOLTZ = constants.Boltzmann #J K^-1


def particle_deposition_rate(x : np.ndarray,
                             temp : float,
                             dynamic_viscosity : float,
                             kinematic_viscosity : float,
                             shear_velocity : float,
                             bturb : float,
                             vrepr : float,
                             sct : float = 1.0,
                             hamaker : Optional[float] = None,
                             permittivity : Optional[float] = None,
                             dl_thickness : Optional[float] = None,
                             phi_dl : Optional[float] = None,
                             adjustment_factor : Optional[float] = 1.0,
                             interactions : Optional[float] = False) -> np.ndarray:
    """
    Calculate deposition rate
    
    Parameters
    ----------
    x : float
        Crystal size (m^3)
    temp : float
        Temperature (K)
    dynamic_viscosity : float
        Dynamic viscosity (Pa s)
    kinematic_viscosity : float
        Kinematic viscosity (m^2 s^-1)
    shear_velocity : float
        Shear velocity (m/s)
    bturb : float
        Wall turbulent viscosity constant (dimensionless).
    vrepr : float
        Representative volume for interactions (m^3)
    sct : float, optional
        Schmidt number (dimensionless). The default is 1.0.
    hamaker : Optional[float], optional
        Hamaker constant (J). The default is None.
    permittivity : Optional[float], optional
        Permittivity of water (C V^-1 m^-1). The default is None.
    dl_thickness : Optional[float], optional
        Debye length (m). The default is None.
    phi_dl : Optional[float], optional
        Electric potential at surface of particle and wall (V). The default is None
    interactions : Optional[float], optional
        Whether to consider interactions (default: True)
    
    Returns
    -------
    deposition_rate : np.ndarray
        The deposition rate constant
    """
    #Calculate without interaction
    rrepr = (3/(4*np.pi)*vrepr)**(1./3)
    
    b = bturb
    r = (3/(4*np.pi)*x)**(1.0/3)
    dbr = (KBOLTZ*temp)/(6*np.pi*dynamic_viscosity*r)
    kappa = b*shear_velocity**3/(kinematic_viscosity**2*sct)
    d0 = 1/integrals.integral2_0inf(dbr,kappa,r)

    #Calculate interactions
    wall_length = kinematic_viscosity/shear_velocity
    y_v = 5*wall_length
    if interactions:
        wd = interactionfunctions.particle_deposition_efficiency(
                rrepr,dynamic_viscosity,kinematic_viscosity,shear_velocity,
                temp,permittivity,phi_dl,dl_thickness,hamaker,
                y_v,b,sct=sct)
        deposition_rate = d0/(1+wd*d0)
    else:
        wd = 1.0
        deposition_rate = d0
    deposition_rate *= adjustment_factor
    return deposition_rate

