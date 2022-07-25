# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np
from scipy import constants

from . import interactionfunctions
from . import auxfunctions


KBOLTZ = constants.Boltzmann #J K^-1


def particle_deposition_rate(x : np.ndarray,
                             temp : float,
                             dynamic_viscosity : float,
                             kinematic_viscosity : float,
                             komolgorov_length : float,
                             shear_velocity : float,
                             fluid_density : float,
                             particle_density : float,
                             bturb : float,
                             vrepr : float,
                             sct : float = 1.0,
                             hamaker : Optional[float] = None,
                             permittivity : Optional[float] = None,
                             dl_thickness : Optional[float] = None,
                             phi_dl : Optional[float] = None,
                             rcorrection : float = 0.0,
                             turbophoretic_constant : float = 0.0,
                             transition_width_factor : float = 0.25,
                             adjustment_factor : Optional[float] = 1.0,
                             diffusional_particle_point_approximation = False,
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
    komolgorov_length : float
        Komolgorov length (m)
    shear_velocity : float
        Shear velocity (m/s)
    fluid_density : float
        Fluid density (kg/m3)
    particle_density : float
        Particle density (kg/m3)
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
    rcorrection : float, optional.
        Hydrodynamic correction for rough surfaces. The default is 0.0.
    turbophoretic_constant : float, optional.
        Turbophoretic constant. The default is 0.0
    transition_width_factor : float = 0.25,
        Turbophoretic adjustment. The default is 0.0.
    diffusional_particle_point_approximation : bool, optional
        DESCRIBE
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
    if not diffusional_particle_point_approximation:
        d0 = 1/auxfunctions.integral2_0inf(dbr,kappa,r)
    else:
        d0 = 1/auxfunctions.integral1_0inf(dbr, kappa)

    #Calculate interactions
    wall_length = kinematic_viscosity/shear_velocity
    y_v = 5*wall_length
    if interactions:
#        wd = interactionfunctions.particle_deposition_efficiency(
#                rrepr,dynamic_viscosity,kinematic_viscosity,shear_velocity,
#                temp,permittivity,phi_dl,dl_thickness,hamaker,
#                y_v,b,sct=sct,rcorrection=rcorrection)
        wdf = lambda rrepr : interactionfunctions.particle_deposition_efficiency(
                 rrepr,dynamic_viscosity,kinematic_viscosity,shear_velocity,
                 temp,permittivity,phi_dl,dl_thickness,hamaker,
                 y_v,b,sct=sct,rcorrection=rcorrection)
        wd = np.array([wdf(rr) for rr in r])
        diffusional_deposition_rate = d0/(1+wd*d0)
    else:
        wd = 1.0
        diffusional_deposition_rate = d0
    turbophoretic_deposition_rate = turbophoretic_constant*shear_velocity

    wall_stokes_number = 1.0/18*particle_density/fluid_density*\
                            (shear_velocity*2*r/kinematic_viscosity)**2
    lb_transition = 3*1e-1 #Young
    ub_transition = 5.0 #Young
    transition_factor = auxfunctions.smooth_transition_log(wall_stokes_number,
                                                           lb_transition,
                                                           ub_transition)
    # wall_stokes_limit = 5.0
    # transition_width = wall_stokes_limit*transition_width_factor
    # transition_factor = auxfunctions.centered_smooth_transition(wall_stokes_number,
    #                                                             wall_stokes_limit,
    #                                                             transition_width)
    # print(wall_stokes_number)
    # print(transition_factor)
    # transition_factor = auxfunctions.centered_smooth_transition(2*r,
    #                                                             komolgorov_length,
    #                                                             komolgorov_length*transition_width_factor)
    # print(2*r)
    # print(komolgorov_length)
    # print(transition_factor)
    # print('--')
    deposition_rate = transition_factor*diffusional_deposition_rate + \
                      (1 - transition_factor)*turbophoretic_deposition_rate
    deposition_rate *= adjustment_factor
    return deposition_rate
