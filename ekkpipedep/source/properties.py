# -*- coding: utf-8 -*-
import warnings

import numpy as np
from pyequion2 import water_properties

from .. import constants


def _calcite_growth_function(satur, temp):
    return 4.6*1e-11*np.clip(np.sqrt(satur) - 1, 0.0, None)**2


_CALCITE_PROPERTIES = \
{
    'surface_energy': 0.085, #J/m^2
    'molar_mass': (60.01 + 40.078)*1e-3, #kg/mol
    'density': 2.711*1e3, #kg/mol
    'growth_function': _calcite_growth_function, #[unitless, K] -> #m/s
    'relative_permittivity': 2.0,
    'refractivity': 1.59
}

_VATERITE_PROPERTIES = \
{
    'surface_energy': 0.04, #J/m^2
    'molar_mass': (60.01 + 40.078)*1e-3, #kg/mol
    'density': 2.54*1e3, #kg/mol
    'growth_function': _calcite_growth_function,
    'relative_permittivity': 2.0,
    'refractivity': 1.59 #[unitless, K] -> #m/s
}


DUMMY_GROWTH_FUNCTION = lambda satur, temp : 0.0
DEFAULT_SURFACE_ENERGY = 0.1 #J/m2
DEFAULT_DENSITY = 1000.0
DEFAULT_MOLAR_MASS = 1.0
DEFAULT_RELATIVE_PERMITTIVITY = 1.0
DEFAULT_REFRACTIVITY = 1.0
DEFAULT_DOUBLE_LAYER_POTENTIAL = -40*1e-3 #J/C


def get_solid_phase_properties(phase):
    if phase == 'Calcite':
        return SolidPhaseProperties(_CALCITE_PROPERTIES)
    elif phase == 'Vaterite':
        return SolidPhaseProperties(_VATERITE_PROPERTIES)
    else:
        warnings.warn("Phase not in database. Returning default")
        return SolidPhaseProperties(dict())


class SolidPhaseProperties(object):
    def __init__(self, kwargs):
        self.density = kwargs.get('density',
                                  DEFAULT_DENSITY)
        self.molar_mass = kwargs.get('molar_mass',
                                     DEFAULT_MOLAR_MASS)
        self.surface_energy = kwargs.get('surface_energy',
                                         DEFAULT_SURFACE_ENERGY)
        self.num_elem = kwargs.get('num_el', 1.0)
        self.growth_function = kwargs.get("growth_function",
                                          DUMMY_GROWTH_FUNCTION)
        self.relative_permittivity = kwargs.get("relative_permittivity",
                                                DEFAULT_RELATIVE_PERMITTIVITY)
        self.refractivity = kwargs.get("refractivity", DEFAULT_REFRACTIVITY)
        self.double_layer_potential = kwargs.get("double_layer_potential",
                                                 DEFAULT_DOUBLE_LAYER_POTENTIAL)
    
    @property
    def vol(self): #m^3
        #Volume of elementary crystal
        return self.molar_mass/(self.density*constants.AVOG)
    
    @property
    def molar_vol(self): #m^3/mol
        #Volume of a mol of crystal
        return self.molar_mass/self.density
    
    @property
    def molecular_mass(self):
        return self.molar_mass/constants.AVOG
    
    @property
    def elementary_diameter(self): #m
        return _volume_to_diameter(self.num_elem*self.vol)

    @property
    def permittivity(self):
        return self.relative_permittivity*constants.VACUUM_PERMITTIVITY

    def get_debye_length(self, ionic_strength: float, temp: float) -> float:
        """
            Calculates the Debye length
        """
        water_density = water_properties.water_density(temp)
        numerator = self.permittivity*constants.KBOLTZ*temp
        denominator = (2*water_density*constants.ECHARGE**2*constants.AVOG*ionic_strength)
        return np.sqrt(numerator/denominator)

    def get_hamaker_constant(self, temp : float) -> float:
        """
            Calculate Hamaker constant according to Lifshitz theory
        """
        term1a = 3/4*constants.KBOLTZ*temp
        term1b = ((self.relative_permittivity - constants.WATER_RELATIVE_PERMITTIVITY)/\
                  (self.relative_permittivity + constants.WATER_RELATIVE_PERMITTIVITY))**2
        term1 = term1a*term1b
        term2a = 3*constants.HPLANCK*constants.UV_ABSORPTION_FREQUENCY/(16*np.sqrt(2))
        term2b1 = (self.refractivity**2 - constants.REFRACTIVITY_WATER**2)**2
        term2b2 = (self.refractivity**2 + constants.REFRACTIVITY_WATER**2)**1.5
        term2 = term2a*term2b1/term2b2
        hamaker = term1 + term2
        return hamaker

        
def _volume_to_diameter(v):
    return (6. * v / np.pi)**(1./3.)


