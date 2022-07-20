# -*- coding: utf-8 -*-
from typing import List, Union, Callable, Optional, Dict, Any, Tuple
import warnings

import numpy as np
import scipy.integrate
import pyequion2

from .source import StationaryPipeSourceFunction
from . import flowparams
from . import constants
from . import interpolator


TemperatureFunction = Callable[[float], float]
Temperature = Union[TemperatureFunction, float]
PressureFunction = Callable[[float], float]
Pressure = Union[PressureFunction, float]
WallCallable = Union[Callable[..., float]]
WallReaction = Tuple[Union[WallCallable, str], List[float], Optional[WallCallable]]
WallReactions = Optional[Dict[str, WallCallable]]


class StationaryPipeFlowSolver():
    """
    
    Attributes
    ----------
    interface_system : InterfaceSystem
        InterfaceSystem responsible for chemical calculations
    flow_velocity : float
        Flow velocity of system (m/s).
    pipe_diameter : float
        Pipe diameter (m).
    pipe_length : float
        Pipe length (m).
    TKb : Temperature, optional
        Temperature at bulk (K).
    TKw : Temperature, optional
        Temperature at bulk (K).
    pressure : Pressure, optional
        Pressure at pipe (Pa).
    pressure_model : str, optional
        Model for pressure drop.
    initial_concentrations : Dict[str, float]
        Initial concentrations in mol/m3.
    initial_nparticles : float
        Initial number of particles in #/m3.
    initial_dparticles : float
        Initial particle diameter.

    """
    def __init__(self):
        self.interface_system = None
        self.flow_velocity = None
        self.pipe_diameter = None
        self.pipe_length = None
        self.TKb = None
        self.TKw = None
        self.pressure = None
        self.pressure_model = None
        self.pipe_source = None
        self.initial_concentrations = None
        self.initial_nparticles = None
        self.initial_dparticles = None
        
    def set_equilibrium_system(self, elements : List[str],
                               activity_model : str = "EXTENDED_DEBYE",
                               transport_model : str = "A"):
        """

        Parameters
        ----------
        elements : List[str]
            Elements in our system.
        activity_model : str, optional
            Activity model used. The default is "EXTENDED_DEBYE".
        transport_model : str, optional
            Transport model used. The default is "A".

        """
        self.interface_system = pyequion2.InterfaceSystem(elements,
                                                          from_elements=True,
                                                          activity_model=activity_model)
        self.interface_system.set_global_transport_model(transport_model)
    
    def set_flow_params(self, flow_velocity : float,
                        pipe_diameter : float,
                        pipe_length : float,
                        TKb : Temperature = 298.15,
                        TKw : Temperature = None,
                        pressure : Pressure = 1e5,
                        pressure_model : str = 'dw_outlet'):
        """

        Parameters
        ----------
        flow_velocity : float
            Flow velocity of system (m/s).
        pipe_diameter : float
            Pipe diameter (m).
        pipe_length : float
            Pipe length (m).
        TKb : Temperature, optional
            Temperature at bulk (K). The default is 298.15.
        TKw : Temperature, optional
            Temperature at bulk (K). The default is None.
        pressure : Pressure, optional
            Pressure at pipe (Pa). The default is 1e5.
        pressure_model : str, optional
            Model for pressure drop. The default is 'dw_outlet'.

        """
        self.flow_velocity = flow_velocity
        self.pipe_diameter = pipe_diameter
        self.pipe_length = pipe_length
        self.TKb = TKb
        self.TKw = TKb if TKw is None else TKw
        self.pressure = pressure
        self.pressure_model = pressure_model
        
    def set_pipe_source(self,
                        wall_phases : List[str],
                        bulk_phases : Optional[List[str]] = None,
                        wall_reactions : WallReactions = None,
                        nmoments : int = 2,
                        length_based_formulation : bool = False):
        """

        Parameters
        ----------
        wall_phases : List[str]
            Mineral phases at the wall.
        bulk_phases : Optional[List[str]], optional
            Mineral phases at the wall. The default is None.
        wall_reactions : WallReactions, optional
            DESCRIPTION. The default is None.
        nmoments: int, optional
            Number of moments to be considered. The default is 2.
        length_based_formulation : bool, optional
            Whether we are using a length-based formulation for
            the method of moments. False means a volume-based formulation.
            The default is False.
        """
        assert(self.interface_system is not None)
        assert(self.flow_velocity is not None)
        if not bulk_phases:
            bulk_phases = wall_phases
        if nmoments%2 != 0:
            warnings.warn("Number of moments must be even. Setting to nmoments - 1")
            nmoments = nmoments - 1
        pipe_source = StationaryPipeSourceFunction(self.interface_system,
                                                   self.flow_velocity,
                                                   self.pipe_diameter,
                                                   self.TKb,
                                                   wall_temperature=self.TKw,
                                                   pressure_model=self.pressure_model,
                                                   pipe_length=self.pipe_length,
                                                   pressure=self.pressure,
                                                   wall_phases=wall_phases,
                                                   wall_reactions=wall_reactions,
                                                   bulk_phases=bulk_phases,
                                                   nmoments=nmoments,
                                                   length_based_formulation=length_based_formulation)
        self.pipe_source = pipe_source

    def set_initial_values(self,
                           concentrations : Dict[str, float],
                           nparticles : float,
                           dparticles : float):
        """

        Parameters
        ----------
        concentrations : Dict[str, float]
            Initial concentrations in mol/m3.
        nparticles : float
            Initial number of particles in #/m3.
        dparticles : float
            Initial particle diameter.

        """
        self.initial_concentrations = concentrations
        self.initial_nparticles = nparticles
        self.initial_dparticles = dparticles 
    
    def solve(self,
              print_frequency : Optional[int] = None):
        """
        Solves our system

        Parameters
        ----------
        print_frequency : Optional[int], optional
            If not None, frequency on which progress is printed. The default is None.

        Returns
        -------
        solver : scipy.integrate._ode.ode
            Statistics of ODE solver.

        """
        assert(self.pipe_source is not None)
        assert(self.initial_concentrations is not None)
        ode_params = dict()
        solver = scipy.integrate.ode(self.pipe_source.f)
        solver.set_integrator('vode',
                              method=ode_params.get("method",'bdf'),
                              order=ode_params.get("order",5),
                              nsteps=ode_params.get("nsteps",3000),
                              rtol=ode_params.get("rtol",1e-6),
                              atol=ode_params.get("atol",1e-16))
        initial_vector = self.pipe_source.get_concentration_moments_vector(self.initial_concentrations,
                                                                           self.initial_nparticles,
                                                                           self.initial_dparticles) #TODO: Generalize
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
    def recorder(self) -> Dict[str, Any]:
        """
        Get the recorded data for computation
        
        Returns
        -------
        Dict[str, Any]
            Get recorded data on solving.

        """
        return self.pipe_source.permanent_recorder
    
    @property
    def interpolator(self) -> interpolator.ResultInterpolator:
        """
            Get interpolator of results

        Returns
        -------
        interpolator.ResultInterpolator
            The interpolator class.

        """
        return interpolator.ResultInterpolator(self.recorder)