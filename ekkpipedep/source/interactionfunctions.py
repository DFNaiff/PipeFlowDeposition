#!/usr/bin/env python3
from typing import Optional, Callable, List, Any, Union
import warnings

import numpy as np
import numpy.typing
from scipy import constants
from scipy import integrate

from . import integrals


SingleCallable = Callable[[float, List[Any]], float]
VectorizedCallable = Callable[[Union[float, numpy.typing.ArrayLike], List[Any]],
                              Union[float, numpy.typing.ArrayLike]]


KBOLTZ = constants.Boltzmann #J K^-1
AVOG = constants.Avogadro #mol^-1
ECHARGE = constants.elementary_charge #C
INTEGRATION_ZERO = 2.75*1e-10 #Diameter of water molecule, in m
INTEGRATION_INFINITY = 1e10

# =============================================================================
# MAIN INTERACTION FUNCTIONS
# =============================================================================
def particle_deposition_efficiency(radius : float,
                                   dynamic_viscosity : float,
                                   kinematic_viscosity : float,
                                   shear_velocity : float,
                                   temp : float,
                                   permittivity : Optional[float],
                                   phi_dl : Optional[float],
                                   dl_thickness : Optional[float],
                                   hamaker : Optional[float],
                                   max_interaction_distance : float,
                                   bturb : float = 9.5*1e-4,
                                   sct : float = 1.0) -> float:
    """

    Parameters
    ----------
    radius : float
        Radius of particle (m).
    dynamic_viscosity : float
        Dynamiac viscosity (Pa s).
    kinematic_viscosity : float
        Kinematic viscosity (m2/s).
    shear_velocity : float
        Shear velocity (m/s).
    temp : float
        Temperature (K).
    permittivity of fluid: Optional[float], optional
        Permittivity (C V^-1 m^-1).
    dl_thickness : Optional[float], optional
        Debye length (m).
    phi_dl : Optional[float]
        Double layer potential (m).
    hamaker : Optional[float], optional
        Hamaker constant (J).
    max_interaction_distance : float
        Maximum distance where interactions are considered.
    bturb : float, optional
        Wall turbulent viscosity constant (dimensionless). The default is 9.5*1e-4.
    sct : float, optional
        Schmidt number (dimensionless). The default is 1.0.

    Returns
    -------
    float
        The particle-wall efficiency

    """
    brownian_diffusion_term = (KBOLTZ*temp)/(6*np.pi*dynamic_viscosity*radius)
    turbulent_diffusion_term = bturb*shear_velocity**3/(kinematic_viscosity**2*sct)
    diffusion_ratio = turbulent_diffusion_term/brownian_diffusion_term
    
    potential_function = lambda y : potential_wall(y,radius,
                                                    hamaker,
                                                    permittivity,phi_dl,
                                                    dl_thickness)
    def inner_integrand(y):
        potential_term = potential_function(y)/(KBOLTZ*temp)
        radius_term = integrals.deriv_11xr3(y,diffusion_ratio,radius)
        return potential_term*radius_term

    def outer_integrand(y):
        inner_integral = 0 #Use approximation trick (works empirically)
        potential_term = potential_function(y)/(KBOLTZ*temp)
        radius_term = (1+diffusion_ratio*(y+radius)**3)
        exp_term = potential_term/radius_term - inner_integral
        hydro_term = gfunction_particle_wall(y/radius)
        outer_radius_term = (1/(1+diffusion_ratio*(y+radius)**3))
        integrand = hydro_term/brownian_diffusion_term*outer_radius_term*np.exp(exp_term)
        return integrand
    term1 = warned_integration(outer_integrand,
                               INTEGRATION_ZERO,
                               max_interaction_distance,
                               1e60)
    term2 = -integrals.integral2_0l(brownian_diffusion_term,
                                    turbulent_diffusion_term,
                                    radius,
                                    max_interaction_distance)
    wd = term1 + term2
    return wd


def brownian_agglomeration_efficiency(radius : float,
                                      hamaker : Optional[float],
                                      permittivity : Optional[float],
                                      phi_dl : Optional[float],
                                      dl_thickness : Optional[float],
                                      temp : float) -> float:
    """
    Calculates brownian agglomeration efficiency, considering both 
    hydrodynamic interactions and potential interactions

    Parameters
    ----------
    radius : float
        Radius of particle.
    hamaker : Optional[float], optional
        Hamaker constant (J).
    permittivity: Optional[float], optional
        Permittivity of fluid (C V^-1 m^-1).
    dl_thickness : Optional[float], optional
        Debye length (m).
    phi_dl : Optional[float]
        Double layer potential (m).
    temp : float
        Temperature in Kelvin.

    Returns
    -------
    Value of efficiency integral.

    """
    def integrand(dist):
        return expify(vectorize(integrand_brownian_agglomeration))(dist,
                                                                   radius,
                                                                   hamaker,
                                                                   permittivity,
                                                                   phi_dl,
                                                                   dl_thickness,
                                                                   temp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error",integrate.IntegrationWarning)
        try:
            integral = integrate.quad(integrand,
                                      np.log(INTEGRATION_ZERO/radius),
                                      np.log(INTEGRATION_INFINITY/radius))[0]
        except:
            integral = np.inf
    return integral


def turbulent_agglomeration_efficiency(radius : float,
                                       hamaker : Optional[float],
                                       permittivity : Optional[float],
                                       phi_dl : Optional[float],
                                       dl_thickness : Optional[float],
                                       temp : float,
                                       dynamic_viscosity : float,
                                       turb_dissipation : float,
                                       kinematic_viscosity : float,
                                       const_turb : float) -> float:
    """

    Parameters
    ----------
    radius : float
        Radius of particle.
    hamaker : Optional[float], optional
        hamaker constant (J).
    permittivity: Optional[float], optional
        Permittivity of fluid (C V^-1 m^-1).
    dl_thickness : Optional[float], optional
        Debye length (m).
    phi_dl : Optional[float]
        Double layer potential (m).
    temp : float
        Temperature in Kelvin.
    dynamic_viscosity : float
        Dynamic viscosity (Pa s).
    turbulent_dissipation : float
        Turbulent dissipation (J/kg/s).
    kinematic_viscosity : float
        Kinematic viscosity (m2/s).
    const_turb : float
        Turbulent agglomeration constant (dimensionless).

    Returns
    -------
    float
        Value of efficiency integral.

    """
    def integrand(dist):
        return expify(vectorize(integrand_turbulent_agglomeration))(
                    dist,radius,
                    hamaker,permittivity,phi_dl,dl_thickness,temp,
                    dynamic_viscosity,turb_dissipation,
                    kinematic_viscosity,const_turb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error",integrate.IntegrationWarning)
        try:
            integral = integrate.quad(integrand,
                                      np.log(INTEGRATION_ZERO/radius),
                                      np.log(INTEGRATION_INFINITY/radius))[0]
        except:
            integral = np.inf
    return integral

# Integrands
def integrand_brownian_agglomeration(addist : float,
                                     radius : float,
                                     hamaker : Optional[float],
                                     permittivity : Optional[float],
                                     phi_dl : Optional[float],
                                     dl_thickness : Optional[float],
                                     temp : float) -> float:
    """
    Integrand function for brownian agglomeration
    
    Parameters
    ----------
    addist : float
        Distance between the particles divided by their radius
    radius : float
        Radius of particle.
    hamaker : Optional[float], optional
        hamaker constant (J).
    permittivity of fluid: Optional[float], optional
        Permittivity (C V^-1 m^-1).
    dl_thickness : Optional[float], optional
        Debye length (m).
    phi_dl : Optional[float]
        Double layer potential (m).
    temp : float
        Temperature in Kelvin.

    Returns
    -------
    integrand : float
        Value of integrand.

    """
    s = addist
    potential = potential_tilde_particle_particle(s, radius, hamaker,
                                                  permittivity, phi_dl, dl_thickness)
    first_term = 2/((s+2)**2*gfunction_particle_particle(s))
    second_term = np.exp(potential/(KBOLTZ*temp))
    integrand = first_term*second_term
    return integrand


def integrand_turbulent_agglomeration(addist : float,
                                      radius : float,
                                      hamaker : Optional[float],
                                      permittivity : Optional[float],
                                      phi_dl : Optional[float],
                                      dl_thickness : Optional[float],
                                      temp : float,
                                      dynamic_viscosity : float,
                                      turb_dissipation : float,
                                      kinematic_viscosity : float,
                                      const_turb : float):
    """

    Parameters
    ----------
    addist : float
        Distance between the particles divided by their radius
    radius : float
        Radius of particle.
    hamaker : Optional[float], optional
        hamaker constant (J).
    permittivity: Optional[float], optional
        Permittivity of fluid (C V^-1 m^-1).
    dl_thickness : Optional[float], optional
        Debye length (m).
    phi_dl : Optional[float]
        Double layer potential (m).
    temp : float
        Temperature in Kelvin.
    dynamic_viscosity : float
        Dynamic viscosity (Pa s).
    turbulent_dissipation : float
        Turbulent dissipation (J/kg/s).
    kinematic_viscosity : float
        Kinematic viscosity (m2/s).
    const_turb : float
        Turbulent agglomeration constant (dimensionless).

    Returns
    -------
    float
        Value of efficiency integral.

    """
    
    s = addist
    A = 0.5*const_turb*radius**3*dynamic_viscosity*np.sqrt(turb_dissipation/kinematic_viscosity) #J/m^2
    second_potential_term = 0.0 #J #Same trick as in deposition
    first_potential_term = potential_tilde_particle_particle(s,radius,hamaker,permittivity,phi_dl,dl_thickness)/((s+2)**2) #J
    inner_term = 1/A*(first_potential_term - second_potential_term) #unitless
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = 24/((s+2)**4*gfunction_particle_particle(s))*np.exp(inner_term) #unitless
        except:
            res = np.inf
    return res


def potential_wall(dist : float,
                   radius : float,
                   hamaker : Optional[float],
                   permittivity : Optional[float],
                   phi_dl : Optional[float],
                   dl_thickness : Optional[float]) -> float:
    """
    Calculate the potential between particle and wall

    Parameters
    ----------
    dist : float
        Distance from particle and wall, from the particle surface (m).
    radius : float
        Radius of particle (m).
    hamaker : Optional[float]
        Hamaker constant (J). If None, then VdW potential is not considered.
    permittivity: Optional[float]
        Permittivity of fluid (C V^-1 m^-1). If None, then DL potential is not considered.
    dl_thickness : Optional[float]
        Debye length (m). If None, then DL potential is not considered.
    phi_dl : Optional[float]
        Double layer potential (m). If None, then DL potential is not considered.


    Returns
    -------
    Potential function (J)

    """
    has_vdw = bool(hamaker)
    has_dl = bool(permittivity and phi_dl and dl_thickness)
    vdw_term = 0.0 if not has_vdw \
                else van_der_waals_potential_wall(dist, radius, hamaker)
    dl_term = 0.0 if not has_dl \
                else double_layer_potential_tilde(dist, radius, 
                                                  permittivity,
                                                  phi_dl,
                                                  dl_thickness)
    return vdw_term + dl_term
            

def van_der_waals_potential_wall(dist : float,
                                 radius : float,
                                 hamaker : float) -> float:
    """
        Van der Waals potential for particle-wall separated by dist
    """
    sr = dist/radius
    res = -1/6*hamaker*(2*(sr+1)/(sr*(sr+2)) - np.log((sr+2)/sr))
    return res


def double_layer_potential_wall(dist : float,
                                radius : float,
                                permittivity : float,
                                phi_dl : float,
                                dl_thickness : float) -> float:
    """
        Double layer potential for particle-wall separated by dist,
        using Dejarguin approximation.
    """
    res = permittivity*radius*phi_dl**2*np.log(1+np.exp(-dist/dl_thickness))
    return res



def gfunction_particle_wall(s : float) -> float:
    """
    Calculate the G(s) function for particle-wall hydrodynamic interaction
    
    Parameters
    ----------
    s : float
        Distance between the particles divided by their radius.

    Returns
    -------
    float
        G(s) value.

    """
    return 1 + 1/(s) + 0.128/np.sqrt(s)


def potential_tilde_particle_particle(s : float,
                                      r : float,
                                      hamaker : Optional[float],
                                      permittivity : Optional[float],
                                      phi_dl : Optional[float],
                                      dl_thickness : Optional[float]) -> float:
    """
    Potential at the surface of s*r
    
    Parameters
    ----------
    s : float
        Adimensional distance.
    r : float
        Particle radius (m).
    hamaker : Optional[float]
        Hamaker constant (J). If None, then VdW potential is not considered.
    permittivity of fluid: Optional[float]
        Permittivity (C V^-1 m^-1). If None, then DL potential is not considered.
    dl_thickness : Optional[float]
        Debye length (m). If None, then DL potential is not considered.
    phi_dl : Optional[float]
        Double layer potential (m). If None, then DL potential is not considered.
    
    Returns
    -------
    Value of potential.
    """
    has_vdw = bool(hamaker)
    has_dl = bool(permittivity and phi_dl and dl_thickness)
    vdw_term = 0.0 if not has_vdw else van_der_waals_potential_tilde(s, hamaker)
    dl_term = 0.0 if not has_dl else double_layer_potential_tilde(s,r,permittivity,phi_dl,dl_thickness)
    return vdw_term + dl_term


def van_der_waals_potential_tilde(s : float, hamaker : float) -> float: #Potential of s/r
    """
        Van der waals potential of spherical particles separated by s*r
    """
    innerterm1 = 2/(s**2 + 4*s)
    innerterm2 = 2/((s+2)**2)
    innerterm3 = np.log((s**2+4*s)/((s+2)**2))
    res = -1/6*hamaker*(innerterm1 + innerterm2 + innerterm3)
    return res


def double_layer_potential_tilde(s,r,permittivity,phi_dl,dl_thickness):
    """
        Double layer potential for spherical particles separated by s*r,
        under the Dejarguin approximation
    """
    res = 1/2*permittivity*r*phi_dl**2*np.log(1+np.exp(-r/dl_thickness*s))
    return res


def gfunction_particle_particle(s : float) -> float:
    """
    Calculate the G(s) function for symmetric particle particle hydrodynamic interaction
    
    Parameters
    ----------
    s : float
        Distance between the particles divided by their radius.

    Returns
    -------
    float
        G(s) value.

    """
    return (6*s**2+4*s)/(6*s**2+13*s+2)


# =============================================================================
# AUXILIARY INTEGRATION FUNCTIONS
# =============================================================================
def vectorize(f : SingleCallable) -> VectorizedCallable:
    def g(s,*args):
        if type(s) == type(np.array(1.0)):
            return np.array([f(ss,*args) for ss in s.flatten()]).reshape(*s.shape)
        else:
            return f(s,*args)
    return g


def expify(f : VectorizedCallable) -> VectorizedCallable:
    def g(s,*args,**kwargs):
        return np.exp(s)*f(np.exp(s),*args,**kwargs)
    return g


def nonwarned_integration(integrand,a,b):
    return integrate.quad(integrand,
                          a,
                          b)[0]

    
def warned_integration(integrand,a,b,error_integral):
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
