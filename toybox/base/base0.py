# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')
import numpy as np
import scipy.integrate
from pyequion import InterfaceSystem
import matplotlib.pyplot as plt

import ekkpipedep.source.flowproperties

from ekkpipedep import StationaryPipeSourceFunction


interface_system = InterfaceSystem(['C', 'Ca', 'Na', 'Cl'], from_elements=True, activity_model="IDEAL")
flow_velocity = 1.0
TK = 298.15
pipe_diameter = 1e-2
pipe_length = 60.0
pressure = {'flow_velocity', flow_velocity}

print(ekkpipedep.source.flowproperties.darcy_weisbach(flow_velocity, pipe_diameter, TK))
pipe_source = StationaryPipeSourceFunction(interface_system, flow_velocity, pipe_diameter, TK,
                                           wall_phases=['Calcite'], pressure_model='dw_outlet',
                                           pipe_length=pipe_length, pressure=1e5)

initial_concentrations = {'C':0.065, 'Ca':0.028, 'Na':0.075, 'Cl':0.056}
#pipe_source.f(0.0, pipe_source.get_concentration_vector(initial_concentrations))

ode_params = dict()
solver = scipy.integrate.ode(pipe_source.f)
solver.set_integrator('vode',
                      method=ode_params.get("method",'bdf'),
                      order=ode_params.get("order",5),
                      nsteps=ode_params.get("nsteps",3000),
                      rtol=ode_params.get("rtol",1e-6),
                      atol=ode_params.get("atol",1e-16))
solver.set_initial_value(pipe_source.get_concentration_vector(initial_concentrations))

tarray = [0.0]
yarray = [pipe_source.get_concentration_vector(initial_concentrations)]
tmax = pipe_length/flow_velocity

while solver.successful():
    solver.integrate(tmax, step=True)
    tarray.append(solver.t)
    yarray.append(solver.y)
    pipe_source.turn_temporary_record_into_permanent()
    if solver.t > tmax:
        break

tarray = np.array(tarray)
yarray = np.array(yarray)

for i, el in enumerate(pipe_source.solute_elements):
    plt.semilogy(tarray, yarray[:, i], label=el)
plt.legend()
plt.figure()

tt, kk, coo = map(np.array, zip(*[(key, value['Jr_Calcite'], value['CO2(g)']) for key, value in pipe_source.permanent_recorder.items()]))
tti = np.argsort(tt)
tt = tt[tti]
kk = kk[tti]

plt.semilogy(tt, kk)
plt.figure()
plt.plot(tt, coo)