# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')
import numpy as np
import scipy.integrate
from pyequion2 import InterfaceSystem
import matplotlib.pyplot as plt

from ekkpipedep import StationaryPipeSourceFunction

interface_system = InterfaceSystem(['C', 'Ca', 'Na', 'Cl', 'Mg'], from_elements=True, activity_model="IDEAL")
pipe_source = StationaryPipeSourceFunction(interface_system, 1.0, 1e-2, 298.15, wall_phases=['Calcite'])
initial_concentrations = {'C':0.065, 'Ca':0.028, 'Na':0.075, 'Cl':0.056, 'Mg':0.020}
initial_nparticles = 1e3
initial_dparticles = 1e-6
initial_vector = pipe_source.get_concentration_moments_vector(initial_concentrations,
                                                              initial_nparticles,
                                                              initial_dparticles)
out = pipe_source.f(0.0, initial_vector)

# ode_params = dict()
# solver = scipy.integrate.ode(pipe_source.f)
# solver.set_integrator('vode',
#                       method=ode_params.get("method",'bdf'),
#                       order=ode_params.get("order",5),
#                       nsteps=ode_params.get("nsteps",3000),
#                       rtol=ode_params.get("rtol",1e-6),
#                       atol=ode_params.get("atol",1e-16))
# solver.set_initial_value(pipe_source.get_concentration_vector(initial_concentrations))

# tarray = [0.0]
# yarray = [pipe_source.get_concentration_vector(initial_concentrations)]
# tmax = 80.0

# while solver.successful():
#     solver.integrate(tmax, step=True)
#     tarray.append(solver.t)
#     yarray.append(solver.y)
#     pipe_source.turn_temporary_record_into_permanent()
#     print(solver.t, solver.y)
#     if solver.t > tmax:
#         break
    
# tarray = np.array(tarray)
# yarray = np.array(yarray)

# for i, el in enumerate(pipe_source.solute_elements):
#     plt.plot(tarray, yarray[:, i], label=el)
# plt.legend()
# plt.figure()

# tt, kk = map(np.array, zip(*[(key, value['Jr_Calcite']) for key, value in pipe_source.permanent_recorder.items()]))
# tti = np.argsort(tt)
# tt = tt[tti]
# kk = kk[tti]

# for i, el in enumerate(pipe_source.solute_elements):
# #    plt.plot(tt, kk)
#     plt.semilogy(tt, kk)