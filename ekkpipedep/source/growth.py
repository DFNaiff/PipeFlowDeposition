# -*- coding: utf-8 -*-
import numpy as np
import pyequion2


def growth_rate(x,
                repr_solution,
                bulk_solution,
                molec_vol,
                reactive_growth_constant_map,
                rrepr,
                solid_density,
                adjustment_factor=1.0):
    """
        Calculates growth rate

        :x: particle size (m^3)
        :chemical_solver: chemical solver class
        :bulk_concentrations: concentrations in bulk (mol/m^3)
        :temp: temperature (K)
        :satur: supersaturation ratio (dimensionless)
        :molec_vol: volume of elementary crystal (m^3)
        :k_2g: reactive growth constant (m s^-1)
        returns growth_rate (m^3/s)
    """
    r = (3/(4*np.pi)*x)**(1.0/3)
    area = 4*np.pi*r**2
    
    # sum(pyequion2.converters.ELEMENTS_MOLAR_WEIGHTS[k]*1e-3*v 
    #     for i, (k, v) in 
    #     enumerate(repr_solution.elements_reaction_fluxes.items()))
    #self.molar_weight/(solid_density*AVOG)
    #mol/m^2*s
    #m*mol/
    # k_r = reactive_growth_constant/molec_vol
#    rate,ionic_strength,charge_density, \
#    _,_,_, \
#    _ = \
#        chemical_solver.solve_full_diffusion_reaction(bulk_concentrations,temp,
#                                    k_ma,k_r,return_surface_concentrations=True,
#                                    return_diffusions=True,return_saturation=True,
#                                    precipitating_phase=init_growth_phase,
#                                    type_reaction_wall="spinoidal")
#    rate = k_r*(np.sqrt(bulk_satur)-1.0)**2*(bulk_satur > 1)
    g = rate*molec_vol*area
    return g
