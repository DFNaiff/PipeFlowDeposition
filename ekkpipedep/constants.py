# -*- coding: utf-8 -*-
import numpy as np
from scipy import constants


ATM = 101325.0
KBOLTZ = constants.Boltzmann #J K^-1
AVOG = constants.Avogadro #mol^-1
TURB_AGGLOMERATION_CONSTANT = 1.3
TURB_VISCOSITY_CONSTANT = 9.5*1e-4
HBAR = constants.hbar #J s
UV_ABSORPTION_FREQUENCY = 3.2*1e15 #1/s
REFRACTIVITY_WATER = 1.33
VACUUM_PERMITTIVITY = constants.epsilon_0 #C^2 J^{-1} m^{-1}
ECHARGE = constants.elementary_charge #C
HPLANCK = 2*np.pi*HBAR #J s
WATER_RELATIVE_PERMITTIVITY = 80.2
WATER_PERMITTIVITY = WATER_RELATIVE_PERMITTIVITY*VACUUM_PERMITTIVITY