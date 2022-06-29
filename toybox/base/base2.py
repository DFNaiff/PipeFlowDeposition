# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')
import numpy as np
import scipy.integrate
from pyequion import InterfaceSystem
import matplotlib.pyplot as plt

import ekkpipedep.source.flowproperties

from ekkpipedep import StationaryPipeSourceFunction, StationaryPipeFlowSolver


flow_velocity = 1.0
#TKb = lambda x : (273.15 + 80.0) if x > 40.0 and x < 80.0 else 298.15
pipe_diameter = 1e-2
pipe_length = 80.0

def TKb(x):
    return 298.15

def TKw(x):
    between = (x > 60.0) & (x < 62.0)
    return 273.15 + 25*(1-between) + 75*between

solver = StationaryPipeFlowSolver()
solver.set_equilibrium_system(['C', 'Ca', 'Na', 'Cl'], activity_model="PITZER")
solver.set_flow_params(flow_velocity, pipe_diameter, pipe_length, TKb, TKw=TKw)
solver.set_initial_values(concentrations={'C':0.065, 'Ca':0.028, 'Na':0.075, 'Cl':0.056})
solver.set_pipe_source(wall_phases=['Calcite'])

solver.solve(print_frequency=1)

tt, kk, coo = map(np.array, zip(*[(key, value['Jr_Calcite'], value['CO2(g)']) for key, value in solver.recorder.items()]))
tti = np.argsort(tt)
tt = tt[tti]
kk = kk[tti]
#
plt.semilogy(tt, kk*np.pi*pipe_diameter*100.09*3600)
plt.figure()
plt.plot(tt, coo)