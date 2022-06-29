# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate
import pyequion2

from .source import StationaryPipeSourceFunction
from . import flowparams
from . import constants


class StationaryPipeFlowSolver():
    def __init__(self):
        self.interface_system = None
        self.flow_params = None
        
    def set_equilibrium_system(self, elements, activity_model="PITZER"):
        self.interface_system = pyequion2.InterfaceSystem(elements,
                                                         from_elements=True,
                                                         activity_model=activity_model)
    
    def set_flow_params(self, flow_velocity, pipe_diameter, pipe_length,
                 TKb=298.15, TKw=None, pressure=1e5, pressure_model='dw_outlet'):
        self.flow_velocity = flow_velocity
        self.pipe_diameter = pipe_diameter
        self.pipe_length = pipe_length
        self.TKb = TKb
        self.TKw = TKb if TKw is None else TKw
        self.pressure = pressure
        self.pressure_model = pressure_model
        
    def set_pipe_source(self, wall_phases):
        pipe_source = StationaryPipeSourceFunction(self.interface_system,
                                                   self.flow_velocity,
                                                   self.pipe_diameter,
                                                   self.TKb,
                                                   wall_temperature=self.TKw,
                                                   pressure_model=self.pressure_model,
                                                   pipe_length=self.pipe_length,
                                                   pressure=self.pressure,
                                                   wall_phases=wall_phases)
        self.pipe_source = pipe_source

    def set_initial_values(self, concentrations, moments=None):
        self.initial_concentrations = concentrations
        
    
    def solve(self, print_frequency=None):
        ode_params = dict()
        solver = scipy.integrate.ode(self.pipe_source.f)
        solver.set_integrator('vode',
                              method=ode_params.get("method",'bdf'),
                              order=ode_params.get("order",5),
                              nsteps=ode_params.get("nsteps",3000),
                              rtol=ode_params.get("rtol",1e-6),
                              atol=ode_params.get("atol",1e-16))
        initial_vector = self.pipe_source.get_concentration_vector(self.initial_concentrations) #TODO: Generalize
        solver.set_initial_value(initial_vector)
        
        tmax = self.pipe_length/self.flow_velocity
        counter = 0
        while solver.successful():
            counter += 1
            solver.integrate(tmax, step=True)
            self.pipe_source.turn_temporary_record_into_permanent()
            if print_frequency:
                if counter%print_frequency == 0:
                    print('{0}: {1} %'.format(counter, solver.t/tmax*100))
            if solver.t > tmax:
                break
        return solver


    @property
    def recorder(self):
        return self.pipe_source.permanent_recorder