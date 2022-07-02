# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')
from pyequion2 import InterfaceSystem

from ekkpipedep import StationaryPipeSourceFunction, StationaryPipeFlowSolver

def single_evaluation():
    interface_system = InterfaceSystem(['C', 'Ca', 'Na', 'Cl', 'Mg'], from_elements=True, activity_model="IDEAL")
    pipe_source = StationaryPipeSourceFunction(interface_system, 1.0, 1e-2, 298.15, wall_phases=['Calcite'])
    initial_concentrations = {'C':65., 'Ca':28., 'Na':75., 'Cl':56., 'Mg':20.}
    initial_nparticles = 1e3
    initial_dparticles = 1e-6
    initial_vector = pipe_source.get_concentration_moments_vector(initial_concentrations,
                                                                  initial_nparticles,
                                                                  initial_dparticles)
    out = pipe_source.f(0.0, initial_vector)
    return out

def model_evaluation():
    solver = StationaryPipeFlowSolver()
    solver.set_equilibrium_system(['Ca', 'C', 'Na', 'Cl'], activity_model="EXTENDED_DEBYE")
    solver.set_flow_params(1.0, 1.0, 1e-3)
    initial_concentrations = {'C':65., 'Ca':28., 'Na':75., 'Cl':56.}
    initial_nparticles = 1e3
    initial_dparticles = 1e-9
    solver.set_initial_values(initial_concentrations,
                              initial_nparticles,
                              initial_dparticles)
    solver.set_pipe_source(['Calcite'], ['Calcite'])
    solver.solve(print_frequency=100)
    return solver

single_evaluation()
model_evaluation()