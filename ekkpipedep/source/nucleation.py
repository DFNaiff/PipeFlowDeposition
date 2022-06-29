# -*- coding: utf-8 -*-
import numpy as np


from .. import constants


"""
A collection of nucleation helpers
"""


def primary_nucleus_size_and_rate(satur, temp, dynamic_viscosity, 
                                  sigma_app, molec_vol, elem_diam):
    if satur <= 1.0: #No nucleation
        return np.pi*(elem_diam**3)/6,0.0
    else:
        #Nucleus diamater and volume, avoiding non-physical volume
        nucleus_diameter = 4*molec_vol*sigma_app/(constants.KBOLTZ*temp*np.log(satur))
        nucleus_diameter = np.max([nucleus_diameter,elem_diam])
        nucleus_volume = np.pi*(nucleus_diameter**3)/6.
        exp_numerator = 16*np.pi/3*sigma_app**3*molec_vol**2/((constants.KBOLTZ*temp)**3)
        exp_factor = exp_numerator/((np.log(satur))**2)
        nucleus_diff = 1/6.0*(constants.KBOLTZ*temp)/(np.pi*dynamic_viscosity*nucleus_diameter/2)
        geometric_factor = 0.6802904372442957 #2*(np.pi/6)**(5.0/3)
        preexp_factor = geometric_factor*nucleus_diff*(molec_vol)**(-5.0/3)
        log_nucleation_rate = np.log(preexp_factor) - exp_factor
        nucleation_rate = np.exp(log_nucleation_rate)
        return nucleus_volume,nucleation_rate