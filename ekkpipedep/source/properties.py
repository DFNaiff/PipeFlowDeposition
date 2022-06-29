# -*- coding: utf-8 -*-
import numpy as np

from .. import constants


_CALCITE_PROPERTIES = \
{
    'surface_energy': 0.085, #J/m^2
    'molar_mass': (60.01 + 40.078)*1e-3, #kg/mol
    'density': 2.711*1e3 #kg/mol
}



def get_solid_phase_properties(phase):
    if phase == 'Calcite':
        return SolidPhaseProperties(_CALCITE_PROPERTIES)
    else:
        raise ValueError("Phase not in database")

class SolidPhaseProperties(object):
    def __init__(self, kwargs):
        self.density = kwargs.get('density')
        self.molar_mass = kwargs.get('molar_mass')
        self.surface_energy = kwargs.get('surface_energy', 0.1)
        self.num_elem = kwargs.get('num_el', 1.0)
        
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


def _volume_to_diameter(v):
    return (6. * v / np.pi)**(1./3.)


