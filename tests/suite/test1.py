# -*- coding: utf-8 -*-

import sys
sys.path.append("../..")

from ekkpipedep.source import flowproperties

TK = 300
Tb = 350
nu = 1e-6
u = 1.0
d = 1e-2

print(flowproperties.reynolds_number(u, d, kinematic_viscosity=nu))
print(flowproperties.reynolds_number(u, d, TK=TK))
print(flowproperties.reynolds_number(u, d, TK=TK, kinematic_viscosity=nu))
print(flowproperties.darcy_friction_factor(u, d, TK=TK, kinematic_viscosity=nu))
print(flowproperties.shear_velocity(u, d, TK=TK))
print(flowproperties.shear_length(u, d, TK=TK))
print(flowproperties.water_density(TK))
print(flowproperties.water_dynamic_viscosity(TK))
print(flowproperties.water_kinematic_viscosity(TK))
print(flowproperties.water_thermal_conductivity(TK))
print(flowproperties.water_specific_heat_capacity(TK))
print(flowproperties.water_thermal_diffusivity(TK))
print(flowproperties.water_nusselt_number(u, d))