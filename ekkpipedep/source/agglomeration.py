# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Callable
import warnings

import numpy as np
import numpy.typing
from scipy import constants
from scipy import interpolate


from . import interactionfunctions
from . import auxfunctions


KBOLTZ = constants.Boltzmann  # J K^-1


def agglomeration_rate(x: np.ndarray, y: np.ndarray,
                       temp: float,
                       dynamic_viscosity: float,
                       kinematic_viscosity: float,
                       turbulent_dissipation: float,
                       const_turb: float,
                       komolgorov_length: float,
                       vrepr: float,
                       hamaker: Optional[float] = None,
                       permittivity: Optional[float] = None,
                       dl_thickness: Optional[float] = None,
                       phi_dl: Optional[float] = None,
                       adjustment_factor: float = 1.0,
                       interactions: bool = False):
    """

    Parameters
    ----------
    x : np.ndarray
        Crystal size (m^3).
    y : np.ndarray
        Crystal size (m^3).
    temp : float
        Temperature (K).
    dynamic_viscosity : float
        Dynamic viscosity (Pa s).
    kinematic_viscosity : float
        Kinematic viscosity (m2/s).
    turbulent_dissipation : float
        Turbulent dissipation (J/kg/s).
    const_turb : float
        Turbulent agglomeration constant (dimensionless).
    komolgorov_length : float
        Komolgorov length (m).
    vrepr : float
        Representative volume of particles (m3).
    hamaker : Optional[float], optional
        Hamaker constant (J). The default is None.
    permittivity of fluid: Optional[float], optional
        Permittivity (C V^-1 m^-1). The default is None.
    dl_thickness : Optional[float], optional
        Debye length (m). The default is None.
    phi_dl : Optional[float], optional
        Double layer potential (m). The default is None.
    adjustment_factor : float, optional
        Adjustment factor (dimensionless). The default is 1.0.
    interactions : bool, optional
        Whether to consider particle-particle interactions. The default is False.

    Returns
    -------
    rate : np.ndarray
        The agglomeration kernel.

    """
    rrepr = (3/(4*np.pi)*vrepr)**(1./3)
    c_brownian = 2.*KBOLTZ*temp/(3.*dynamic_viscosity)
    rate_brownian = c_brownian*(2 + (x/y)**(1./3) + (y/x)**(1./3))
    # turbulent coagulation
    c_turbulent = 3*const_turb *\
        np.sqrt(turbulent_dissipation/kinematic_viscosity)/(4*np.pi)
    rate_turbulent = c_turbulent*(x**(1./3) + y**(1./3))**3.0
    dx, dy = 2*(3/(4*np.pi)*x)**(1.0/3), 2*(3/(4*np.pi)*y**(1.0/3))
#    soft_constraint = 4*sigmoid(-2*dx/komolgorov_length) * \
#        sigmoid(-2*dy/komolgorov_length)
    soft_constraint = auxfunctions.smooth_transition(dx, komolgorov_length, komolgorov_length/4)*\
                      auxfunctions.smooth_transition(dy, komolgorov_length, komolgorov_length/4)
    #rate_turbulent *= soft_constraint
    iwbr = 1.0
    iwt = 1.0
    if interactions:
        wbr = interactionfunctions.brownian_agglomeration_efficiency(
            rrepr, hamaker, permittivity, phi_dl, dl_thickness, temp)
        wt = interactionfunctions.turbulent_agglomeration_efficiency(
            rrepr, hamaker, permittivity, phi_dl, dl_thickness, temp,
            dynamic_viscosity, turbulent_dissipation, kinematic_viscosity, const_turb)
        if wbr <= 1e-10 or wt <= 1e-10:
            warnings.warn(
                "Some error in integral calculations. Considering no interaction")
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


def make_interaction_memoizers(rlow : float, rhigh : float, nsteps : int,
                               hamaker : Optional[float], permittivity : Optional[float],
                               phi_dl : Optional[float], dl_thickness : Optional[float],
                               temp : float,
                               dynamic_viscosity : float,
                               turbulent_dissipation : float,
                               kinematic_viscosity : float,
                               const_turb : float
                               ) -> Tuple[Callable[float, float], Callable[float, float]]:
    rsteps = np.logspace(np.log10(rlow), np.log10(rhigh), nsteps)
    iwbrs = []
    iwts = []
    for r in rsteps:
        wbr = interactionfunctions.brownian_agglomeration_efficiency(
            r, hamaker, permittivity, phi_dl, dl_thickness, temp)
        wt = interactionfunctions.turbulent_agglomeration_efficiency(
            r, hamaker, permittivity, phi_dl, dl_thickness, temp,
            dynamic_viscosity, turbulent_dissipation, kinematic_viscosity, const_turb)
        if wbr <= 1e-10 or wt <= 1e-10:
            warnings.warn(
                "Some error in integral calculations. Considering no interaction")
            iwbr = 1.0
            iwt = 1.0
        else:
            iwbr = 1/wbr
            iwt = 1/wt
        iwbrs.append(iwbr)
        iwt.append(iwt)
    iwbrs, iwts = map(np.array, [iwbrs, iwts])
    iwbrfunc = interpolate.interp1d(rsteps, iwbrs, axis=0, fill_value="extrapolate")
    iwtfunc = interpolate.interp1d(rsteps, iwts, axis=0, fill_value="extrapolate")
    return iwbrfunc, iwtfunc