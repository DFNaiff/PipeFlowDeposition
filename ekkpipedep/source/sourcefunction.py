# -*- coding: utf-8 -*-
import warnings

import numpy as np
import ordered_set

from . import flowproperties
from . import properties
from . import nucleation
from . import agglomeration
from . import particledeposition
from . import momentinversion
from .. import constants


class StationaryPipeSourceFunction():
    def __init__(self,
                 interface_system,
                 flow_velocity,
                 pipe_diameter,
                 temperature=298.15,
                 wall_temperature=None,
                 pressure=constants.ATM,
                 pressure_model=None,
                 pipe_length=None,
                 wall_phases=['Calcite'],
                 wall_reactions=None,
                 bulk_phases=['Calcite']):
        """
        Parameters
        ----------
        interface_system: pyequion.InterfaceSystem
            Main controler for ionic chemistry
        """
        #Let's be specific here: the arguments are:
        #concentrations
        #moments for each precipitating phase
        #deposited mass for each phase
        self.interface_system = interface_system #The interface system for calculating chemical balances
        self.moments_included = set() #Included phases for moments
        self.nmoments = 2 #Number of moments to be considered
        self.models = ordered_set.OrderedSet() # (?)
        self.flow_velocity = flow_velocity #Velocity of flow
        self.pipe_diameter = pipe_diameter #Diameter of pipe
        self.pipe_length = pipe_length #Length of pipe
        self.set_wall_phases(wall_phases, fill_defaults=True) #Set the phases precipitation on wall
        self.set_wall_reaction_functions(wall_reactions, fill_defaults=True) #Set the reaction functions
        self.set_temperature_function(temperature, wall_temperature) #Set the temperature functions
        self.set_pressure_function(pressure, pressure_model) #Set the pressure functions
        self.bulk_phases = bulk_phases #Phases at the bulk
        self.wall_phases = wall_phases #Phases at the wall
        
        self.has_ionic_deposition = True
        self.has_nucleation = True
        self.has_agglomeration = True
        self.has_growth = False
        self.has_particle_deposition = True
        self.nzeroagg = None
        self.bulk_initial_guess = 'default'
        self.wall_initial_guess = 'bulk'
        self.sphere_initial_guess = 'bulk'
        
        self.permanent_recorder = dict()
        self.temporary_recorder = dict()
        self.recorder_time_key = 0.0
        
    def f(self, t, y): #Main source function
        self.make_temporary_record('t', t)
        
        y = np.array(y)
        input_dict = self.split_input(y)
        output_dict = self.split_input(np.zeros_like(y))
        elements_balance = self.get_elements_balance_dict(input_dict['elements'])
        moments = input_dict['moments']
        pos = t*self.flow_velocity
        TKb = self.bulk_temperature_function(pos)
        TKw = self.wall_temperature_function(pos)
        
        pressure = self.pressure_function(pos)
        sol_bulk = self._solve_bulk_equilibrium(elements_balance, TKb, pressure)
        
        if self.has_agglomeration or self.has_growth or self.has_particle_deposition:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if np.all(moments ==0.0): #Special case: zero initial particles
                    vols = np.ones(self.nmoments//2)*1e-20
                    weights = (np.arange(self.nmoments//2)+1)*1e-30
                else:
                    vols, weights = momentinversion.ZetaChebyshev(moments)
            assert np.all(vols >= 0.) & np.all(weights >= 0.) #Asserts everything is okay
        
        if self.has_ionic_deposition:
            sol_wall = self._solve_wall_equilibrium(sol_bulk,
                                                    TKw,
                                                    self.flow_velocity,
                                                    self.pipe_diameter)
            ionic_elements_source_vector = self.ionic_deposition_elements_sources(sol_wall,
                                                                                  self.pipe_diameter,
                                                                                  TKb)
            output_dict['elements'] += ionic_elements_source_vector
        if self.has_nucleation:
            nucleation_elements_source_vector, \
            nucleation_moments_source_vector = self.nucleation_sources(sol_bulk, TKb)
            output_dict['elements'] += nucleation_elements_source_vector
            output_dict['moments'] += nucleation_moments_source_vector
            
        if self.has_agglomeration:
            agglomeration_moments_source_vector = self.agglomeration_sources(vols,
                                                                             weights,
                                                                             TKb,
                                                                             self.flow_velocity,
                                                                             self.pipe_diameter)
            output_dict['moments'] += agglomeration_moments_source_vector
            
        if self.has_growth:
            representative_radius = (np.mean(vols)*4/3)**(1.0/3)
            sol_sphere_repr = self._solve_sphere_equilibrium(sol_bulk,
                                                             representative_radius,
                                                             TKb)
            
            
        if self.has_particle_deposition:
            particle_moments_source_vector, source_particle_mass = \
                self.particle_deposition_sources(vols, weights, TKb,
                                                 self.flow_velocity,
                                                 self.pipe_diameter) #self.particle_deposition_sources
        
        output_vector = self.join_input(output_dict)
        
        self.make_temporary_record_from_dict(sol_wall.reaction_fluxes, 'Jr_')
        self.make_temporary_record_from_dict(sol_bulk.gas_molals)
        return output_vector
            
    def ionic_deposition_elements_sources(self, sol_wall, pipe_diameter, TKb):
        wall_scale_params = 1/(flowproperties.pipe_flux_scale(pipe_diameter)*\
                               flowproperties.water_density(TKb))#1/m*(m^3/kg)
        interface_elements_sources = {k: -wall_scale_params*v 
                                      for k, v in sol_wall.elements_reaction_fluxes.items() 
                                      if k not in ['H', 'O']} #mol/m^2 s #FIXME:
        source_vector = _dict_hstack(interface_elements_sources, self.solute_elements)
        return source_vector
    
    def nucleation_sources(self, sol_bulk, TKb):
        solid_indexes = self.interface_system.get_solid_indexes(self.bulk_phases)
        element_balance_vector = \
            self.interface_system.formula_matrix@\
            self.interface_system.solid_stoich_matrix[solid_indexes, :].transpose()
        elements_balance_vector = element_balance_vector[2:-1, :]
        element_source_vector = np.zeros(elements_balance_vector.shape[0])
        moments_source_vector = np.zeros(self.nmoments)
        rates_and_volumes = dict()
        for phase in self.bulk_phases:
            rate, vol = self._get_nucleation_rate_and_volume(sol_bulk, TKb, phase)
            rates_and_volumes[phase] = (rate, vol)
        for i, phase in enumerate(self.bulk_phases):
            phase_properties = properties.get_solid_phase_properties(phase)
            molar_mass = phase_properties.molar_mass
            density = phase_properties.density
            rate, vol = rates_and_volumes[phase]
            element_source_vector += -elements_balance_vector[:, i]/molar_mass*density*rate*vol
            moments_source_vector += rate*vol**self.kmoments
        return element_source_vector, moments_source_vector

    def agglomeration_sources(self, vols, weights, TK, flow_velocity, pipe_diameter):
        dynamic_viscosity = flowproperties.water_dynamic_viscosity(TK)
        kinematic_viscosity = flowproperties.water_kinematic_viscosity(TK)
        turbulent_dissipation = flowproperties.turbulent_dissipation(flow_velocity, pipe_diameter, TK)
        komolgorov_length = flowproperties.komolgorov_length(flow_velocity, pipe_diameter, TK)
        const_turb = constants.TURB_AGGLOMERATION_CONSTANT
        interactions = False #Set to true
        vols_n11,vols_1n1 = vols.reshape(-1,1,1),vols.reshape(1,-1,1)
        agglomeration_kernel = agglomeration.agglomeration_rate(
                                                  vols_n11, vols_1n1,
                                                  TK, dynamic_viscosity,
                                                  kinematic_viscosity,
                                                  turbulent_dissipation,
                                                  const_turb,
                                                  komolgorov_length,
                                                  interactions=interactions)
        weights_n11,weights_1n1 = weights.reshape(-1,1,1),weights.reshape(1,-1,1) #(1,n,1),#(n,1,1)
        weights_matrix = weights_n11*weights_1n1 #(n,n,1)
        vols_matrix = 0.5*((vols_n11 + vols_1n1)**self.kmoments - 2*vols_1n1**self.kmoments) #(n,n,k)
        agg_tensor = weights_matrix*vols_matrix*agglomeration_kernel #(n,n,k)
        source_agg = agg_tensor.sum(axis=-2).sum(axis=-2) #(k,)
        source_agg[1] = 0.0 #First moment variaton due to coag is analytically zero.
        if self.nzeroagg is not None and self.nmoments > self.nzeroagg:
            source_agg = np.hstack([source_agg[:self.nzeroagg],
                                     np.zeros(self.nmoments-self.num_zero_coag),])
        return source_agg

    def particle_deposition_sources(self, vols, weights, TK, flow_velocity, pipe_diameter):
        dynamic_viscosity = flowproperties.water_dynamic_viscosity(TK)
        kinematic_viscosity = flowproperties.water_kinematic_viscosity(TK)
        shear_velocity = flowproperties.shear_velocity(flow_velocity, pipe_diameter, TK)
        bturb = constants.TURB_VISCOSITY_CONSTANT
        particle_deposition_rate_vector = particledeposition.particle_deposition_rate(vols,
                                                                                      TK,
                                                                                      dynamic_viscosity,
                                                                                      kinematic_viscosity,
                                                                                      shear_velocity,
                                                                                      bturb)
        kaux = self.kmoments.reshape(-1, 1) #(k, 1)
        particle_matrix = vols**kaux*particle_deposition_rate_vector*weights
        source_moments_particle_deposition = -4/pipe_diameter*particle_matrix.sum(axis=1)
        source_mass_particle_deposition = np.pi*pipe_diameter*self.solid_density*\
            (particle_deposition_rate_vector*vols*weights).sum()
        return source_moments_particle_deposition, source_mass_particle_deposition

    def _solve_bulk_equilibrium(self, elements_balance, TK, P, update_guess=True):
        #TODO: Variable initial guess
        sol, sol_stats = self.interface_system.\
                                solve_equilibrium_elements_balance_phases(TK,
                                                                   elements_balance,
                                                                   solid_phases=[],
                                                                   initial_guess=self.bulk_initial_guess,
                                                                   has_gas_phases=False,
                                                                   PATM=P/constants.ATM,
                                                                   maxiter=10000)
        if update_guess:
            self.bulk_initial_guess = sol_stats['x']
        return sol

    def _solve_wall_equilibrium(self, sol_bulk, TK, flow_velocity, pipe_diameter, update_guess=True):
        shear_velocity = flowproperties.shear_velocity(flow_velocity, pipe_diameter, TK)
        molals_bulk = sol_bulk.solute_molals
        transport_params = {'type': 'pipe', 'shear_velocity': shear_velocity}
        sol, sol_stats = self.interface_system.solve_interface_equilibrium(TK,
                                                                           molals_bulk,
                                                                           transport_params,
                                                                           initial_guess=self.wall_initial_guess)
        if update_guess:
            self.wall_initial_guess = sol_stats['x']
        return sol
    
    def _solve_sphere_equilibrium(self, sol_bulk, repr_radius, TK, update_guess=True):
        molals_bulk = sol_bulk.solute_molals
        transport_params = {'type': 'sphere',
                            'radius': repr_radius}
        sol, sol_stats = self.interface_system.solve_interface_equilibrium(TK,
                                                                           molals_bulk,
                                                                           transport_params,
                                                                           initial_guess=self.sphere_initial_guess,
                                                                           fully_diffusive=True)
        if update_guess:
            self.sphere_initial_guess = sol_stats['x']
        return sol

    def _get_nucleation_rate_and_volume(self, sol_bulk, TKb, phase_name):
        satur = 10**sol_bulk.saturation_indexes[phase_name]
        dynamic_viscosity = flowproperties.water_dynamic_viscosity(TKb)
        phase_properties = properties.get_solid_phase_properties(phase_name)
        sigma_app = phase_properties.surface_energy
        molec_vol = phase_properties.vol
        elem_diam = phase_properties.elementary_diameter
        vol, rate = nucleation.primary_nucleus_size_and_rate(satur, TKb, dynamic_viscosity, 
                                          sigma_app, molec_vol, elem_diam)
        return rate, vol

    def split_input(self, x):
        """

        Parameters
        ----------
        x : numpy.ndarray
            1-d array to be splitted.

        Returns
        -------
        dict
            The array map.

        """
        index_dict = self.make_index_dictionary()
        return {key: x[value] for key, value in index_dict.items()}
    
    def join_input(self, index_values):
        """

        Parameters
        ----------
        index_values : dict
            The array map.

        Returns
        -------
        numpy.ndarray
            The joined input as vector.

        """
        index_dict = self.make_index_dictionary()
        return np.hstack([index_values[key] for key in index_dict.keys()])
    
    def make_index_dictionary(self):
        """

        Returns
        -------
        index_dict : dict
            Dictionary mapping the property name to its indexes
            {"elements": [0, 1, 2, 3], "moments": [4, 5]}

        """
        #elements concentrations, moments
        nsolutes = len(self.solute_elements) #For instance, in Ca, C, Na, Cl, returns 4
        nmoments = self.nmoments #For instance, in Calcite, returns [2]
        chunks = np.array([nsolutes, nmoments]).astype(int) #[0, 1, ...]
        indexes = _chunk_split(np.arange(sum(chunks)), chunks) #[0, 1, 2, 3], [4, 5, 6]
        index_dict = {'elements': indexes[0], 'moments': indexes[1]}
        return index_dict
        
    def set_temperature_function(self, temperature, wall_temperature):
        if callable(temperature):
            self.bulk_temperature_function = temperature
        else:
            self.bulk_temperature_function = lambda x : temperature
        if wall_temperature is None:
            self.wall_temperature_function = self.bulk_temperature_function
        elif callable(wall_temperature):
            self.wall_temperature_function = wall_temperature
        else:
            self.wall_temperature_function = lambda x : wall_temperature

    def set_pressure_function(self, pressure, pressure_model):
        TK = self.bulk_temperature_function(0.0)
        flow_velocity = self.flow_velocity
        pipe_diameter = self.pipe_diameter
        if pressure_model is None:
            if callable(pressure):
                pfunc = pressure
            else:
                pfunc = lambda x : pressure
        elif pressure_model == 'dw_inlet':
            dw = flowproperties.darcy_weisbach(flow_velocity, pipe_diameter, TK)
            pfunc = lambda x : pressure - dw*x
        elif pressure_model == 'dw_outlet':
            assert self.pipe_length is not None
            dw = flowproperties.darcy_weisbach(flow_velocity, pipe_diameter, TK)
            pfunc = lambda x : pressure + dw*(self.pipe_length - x)
        self.pressure_function = pfunc
                
    def get_elements_balance_dict(self, x):
        return {el: x[i] for i, el in enumerate(self.solute_elements)}
    
    def get_concentration_vector(self, d):
        return _dict_hstack(d, self.solute_elements)
    
    def get_delta_moment_vector(self, nparticles, diamparticles):
        radiusparticles = diamparticles/2
        volparticles = 4.0/3*np.pi*radiusparticles**3
        moments = [nparticles * volparticles**k for k in np.arange(self.nmoments)]
        return np.array(moments)
    
    def get_concentration_moments_vector(self, elements_dict, nparticles, volparticles):
        return np.hstack([self.get_concentration_vector(elements_dict),
                          self.get_delta_moment_vector(nparticles, volparticles)])
    
    @property
    def set_wall_phases(self): #FIXME: Put actual parameters
        """Alias for InterfaceSystem.set_interface_phases"""
        return self.interface_system.set_interface_phases
    
    @property
    def set_wall_reaction_functions(self):
        """Alias for InterfaceSystem.set_reaction_functions"""
        return self.interface_system.set_reaction_functions
    
    @property
    def solute_elements(self):
        return self.interface_system.solute_elements
    
    @property
    def nmoments(self):
        return self._nmoments
    
    @nmoments.setter
    def nmoments(self, n):
        assert n%2 == 0
        self._nmoments = n
        
    @property
    def kmoments(self):
        return np.arange(self.nmoments, dtype=float)
    
    @property
    def solid_density(self):
        return 2.71*1e3 #FIXME
    
    def make_temporary_record(self, key, value):
        if key == 't':
            self.recorder_time_key = value
        else:
            self.temporary_recorder[key] = value
        
    def make_temporary_record_from_dict(self, d, prefix=''):
        for key, value in d.items():
            self.temporary_recorder[prefix + str(key)] = value
    
    def turn_temporary_record_into_permanent(self):
        self.permanent_recorder[self.recorder_time_key] = self.temporary_recorder
        self.temporary_recorder = dict()


#Auxiliary functions
def _chunk_split(ary, chunks, axis=0):
    assert ary.shape[axis] == sum(chunks)
    return np.split(ary, np.cumsum(chunks, dtype=int), axis=axis)[:-1]


def _dict_hstack(d, order):
    return np.hstack([d[o] for o in order])