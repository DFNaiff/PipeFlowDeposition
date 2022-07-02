# -*- coding: utf-8 -*-

"""

A collection of equations for calculating properties of pipe flows

"""

from typing import Optional

import numpy as np
import pyequion2.water_properties


def reynolds_number(flow_velocity : float,
                    pipe_diameter : float,
                    TK : float = 298.15,
                    kinematic_viscosity : Optional[float] = None): #Dimensionless
    """
        Calculates Reynolds number of water from velocity and diameter
    """
    kinematic_viscosity = kinematic_viscosity or water_kinematic_viscosity(TK)
    return flow_velocity*pipe_diameter/kinematic_viscosity


def darcy_friction_factor(flow_velocity : float,
                          pipe_diameter : float,
                          TK : float = 298.15,
                          kinematic_viscosity : Optional[float] = None):
    reynolds = reynolds_number(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    if reynolds < 2300:
        return 64/reynolds
    else: #Blasius
        return 0.316*reynolds**(-1./4)
    

def shear_velocity(flow_velocity : float,
                   pipe_diameter : float,
                   TK : float = 298.15,
                   kinematic_viscosity : Optional[float] = None):
    f = darcy_friction_factor(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    return np.sqrt(f/8.0)*flow_velocity


def shear_length(flow_velocity : float,
                 pipe_diameter : float,
                 TK : float = 298.15,
                 kinematic_viscosity : Optional[float] = None): #m
    shearv = shear_velocity(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    kv = kinematic_viscosity or water_kinematic_viscosity(TK)
    return kv/shearv


def pipe_flux_scale(pipe_diameter : float): #m: cross_area/cross_perimieter
    return pipe_diameter/4 #pipe_cross_area(pipe_diameter)/pipe_cross_perimeter(pipe_diameter)


def pipe_cross_area(pipe_diameter : float):
    return np.pi*pipe_diameter**2/4


def pipe_cross_perimeter(pipe_diameter : float):
    return np.pi*pipe_diameter


def water_density(TK : float = 298.15):
    return pyequion2.water_properties.water_density(TK)


def water_dynamic_viscosity(TK : float = 298.15):
    return pyequion2.water_properties.water_dynamic_viscosity(TK)


def water_kinematic_viscosity(TK : float = 298.15):
    return pyequion2.water_properties.water_kinematic_viscosity(TK)


def water_thermal_conductivity(TK : float = 298.15):
    return pyequion2.water_properties.water_thermal_conductivity(TK)


def water_specific_heat_capacity(TK : float = 298.15):
    return pyequion2.water_properties.water_specific_heat_capacity(TK)


def water_thermal_diffusivity(TK : float = 298.15):
    return water_thermal_conductivity(TK)/(water_density(TK)*water_specific_heat_capacity(TK))


def water_prandtl_number(TK : float = 298.15):
    return water_kinematic_viscosity(TK)/water_thermal_diffusivity(TK)


def water_nusselt_number(flow_velocity : float,
                         pipe_diameter : float,
                         TKb : float = 298.15,
                         TKw : float = 298.15,
                         kinematic_viscosity : Optional[float] = None): #Dittus-Boelter
    n = 0.4 if TKw > TKb else 0.3
    return 0.023*reynolds_number(flow_velocity, pipe_diameter, TKb, kinematic_viscosity)**(0.8)*\
           water_prandtl_number(TKb)**n


def turbulent_dissipation(flow_velocity : float,
                          pipe_diameter : float,
                          TK : float = 298.15,
                          kinematic_viscosity : Optional[float] = None): #m^2 s^-3
    f = darcy_friction_factor(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    return f/2*flow_velocity**3/pipe_diameter


def vol_flow_rate(flow_velocity : float,
                  pipe_diameter : float): #m^3 s^-1
    return np.pi*pipe_diameter**2*flow_velocity/4.0


def komolgorov_time(flow_velocity : float,
                    pipe_diameter : float,
                    TK : float = 298.15,
                    kinematic_viscosity : Optional[float] = None): #s
    nu = kinematic_viscosity or water_kinematic_viscosity(TK)
    tbd = turbulent_dissipation(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    return np.sqrt(nu/tbd)


def komolgorov_length(flow_velocity : float,
                      pipe_diameter : float,
                      TK : float = 298.15,
                      kinematic_viscosity : Optional[float] = None): #m
    nu = kinematic_viscosity or water_kinematic_viscosity(TK)
    tbd = turbulent_dissipation(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    return (nu**3/tbd)**(1.0/4)


def komolgorov_velocity(flow_velocity : float,
                        pipe_diameter : float,
                        TK : float = 298.15,
                        kinematic_viscosity : Optional[float] = None): #m/s
    nu = kinematic_viscosity or water_kinematic_viscosity(TK)
    tbd = turbulent_dissipation(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    return (nu*tbd)**(1.0/4)


def darcy_weisbach(flow_velocity : float,
                   pipe_diameter : float,
                   TK : float = 298.15,
                   kinematic_viscosity : Optional[float] = None,
                   flow_density : Optional[float] = None):
    f = darcy_friction_factor(flow_velocity, pipe_diameter, TK, kinematic_viscosity)
    flow_density = flow_density or water_density(TK)
    dpdx = f*flow_density/2.0*flow_velocity**2/pipe_diameter
    return dpdx