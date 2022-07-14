# -*- coding: utf-8 -*-
from typing import Union, List, Callable, Optional, Any, Tuple, Dict, Protocol
import warnings

import numpy as np
import numpy.typing
import pyequion2

from . import flowproperties
from . import properties
from . import nucleation
from . import agglomeration
from . import particledeposition
from . import growth
from . import momentinversion
from .. import constants


InterfaceSystem = pyequion2.InterfaceSystem
TemperatureFunction = Callable[[float], float]
Temperature = Union[TemperatureFunction, float]
PressureFunction = Callable[[float], float]
Pressure = Union[PressureFunction, float]
MineralPhase = Union[List[str], str]
InterfaceSolution = pyequion2.interface.interface_solution.InterfaceSolutionResult
BulkSolution = pyequion2.solution.SolutionResult
WallCallable = Union[Callable[..., float]]
WallReaction = Tuple[Union[WallCallable, str], List[float], Optional[WallCallable]]
WallReactions = Optional[Dict[str, WallCallable]]



class StationaryPipeSourceFunction():
    def __init__(self,
                 interface_system: InterfaceSystem,
                 flow_velocity: float,
                 pipe_diameter: float,
                 temperature: Temperature = 298.15,
                 wall_temperature: Optional[Temperature] = None,
                 pressure: Pressure = constants.ATM,
                 pressure_model: Optional[str] = None,
                 pipe_length: Optional[float] = None,
                 wall_phases: MineralPhase = ['Calcite'],
                 wall_reactions: WallReactions = None,
                 bulk_phases: MineralPhase = ['Calcite'],
                 nmoments: int = 2):
        """

        Parameters
        ----------
        interface_system : InterfaceSystem
            InterfaceSystem class for chemical solving.
        flow_velocity : float
            Flow velocity.
        pipe_diameter : float
            Pipe diameter.
        temperature : Temperature, optional
            Temperature of the flow. The default is 298.15.
        wall_temperature : Optional[Temperature], optional
            Temperature of the wall. The default is None.
        pressure : Pressure, optional
            Pressure (Pa). The default is constants.ATM.
        pressure_model : Optional[str], optional
            Model for pressure. The default is None.
        pipe_length : Optional[float], optional
            Length of the pipe. The default is None.
        wall_phases : MineralPhase, optional
            Mineral phases at the wall. The default is ['Calcite'].
        wall_reactions : WallReactions, optional
            DESCRIPTION. The default is None.
        bulk_phases : MineralPhase, optional
            Mineral phases at the bulk. The default is ['Calcite'].
        nmoments: int, optional
            Number of moments to be considered. The default is 2.
        """
        self.interface_system = interface_system  # The interface system for calculating chemical balances
        self.nmoments = nmoments  # Number of moments to be considered
        self.flow_velocity = flow_velocity  # Velocity of flow
        self.pipe_diameter = pipe_diameter  # Diameter of pipe
        self.pipe_length = pipe_length  # Length of pipe
        # Set the phases precipitation on wall
        self.set_wall_phases(wall_phases, fill_defaults=True)
        self.set_wall_reaction_functions(
            wall_reactions, fill_defaults=True)  # Set the reaction functions
        # Set the temperature functions
        self.set_temperature_function(temperature, wall_temperature)
        # Set the pressure functions
        self.set_pressure_function(pressure, pressure_model)
        self.bulk_phases = bulk_phases  # Phases at the bulk
        self.wall_phases = wall_phases  # Phases at the wall

        self.has_ionic_deposition = True
        self.has_nucleation = True
        self.has_agglomeration = True
        self.has_growth = True
        self.has_particle_deposition = True
        self.nzeroagg = None
        self.agglomeration_interactions = True
        self.particle_deposition_interactions = True
        self.van_der_walls_interactions = True
        self.double_layer_interactions = True
        self.bulk_initial_guess = 'default'
        self.wall_initial_guess = 'bulk'
        self.sphere_initial_guess = 'bulk'
        self.roughness_wall_correction = 0.0 #Experimental variable
        self.turbulent_agglomeration_correction = 1.0 #Experimental variable
        
        self.permanent_recorder = dict()
        self.temporary_recorder = dict()
        self.recorder_space_key = 0.0

    def f(self, t: float, y: np.ndarray) -> np.ndarray:  # Main source function
        """
            Source function for calculating ODE. In principle only to be used inside ODE.

            Parameters
            ----------
            t : float
                The time of ODE
            y : float
                The variable of the ODE

            Returns
            -------
            The value of the source function
        """
        pos = t*self.flow_velocity  # Position
        self.make_temporary_space_record(pos)

        y = np.array(y)  # Turn y into an array
        # EXTREMELY IMPORTANT : 'elements' concentration units here is in mol/m3
        # Split input in ['moments', 'elements']
        input_dict = self.split_input(y)
        # Make dictionary to hold outputs
        output_dict = self.split_input(np.zeros_like(y))
        moments = input_dict['moments']  # Moments
        TKb = self.bulk_temperature_function(pos)  # Temperature at bulk
        TKw = self.wall_temperature_function(pos)  # Temperature at wall
        elements_balance = self.get_elements_balance_dict(
            input_dict['elements'], TKb)  # Get elements balance
        vrepr = moments[1]/moments[0]

        pressure = self.pressure_function(pos)  # Pressure at bulk
        sol_bulk = self._solve_bulk_equilibrium(
            elements_balance, TKb, pressure)  # Solve bulk chemical equilibrium
        ionic_strength = sol_bulk.ionic_strength

        if self.has_agglomeration or self.has_growth or self.has_particle_deposition:
            if len(self.bulk_phases) > 1:
                raise NotImplementedError(
                    "We currently cannot model particles with more than one mineral phase")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if np.all(moments == 0.0):  # Special case: zero initial particles
                    vols = np.ones(self.nmoments//2)*1e-20
                    weights = (np.arange(self.nmoments//2)+1)*1e-30
                else:
                    vols, weights = momentinversion.ZetaChebyshev(moments)
            try:
                assert np.all(vols >= 0.) & np.all(
                    weights >= 0.)  # Asserts everything is okay
            except AssertionError:
                raise AssertionError("Some error in moment inversion")
        if self.has_ionic_deposition:
            sol_wall = self._solve_wall_equilibrium(sol_bulk,
                                                    TKw,
                                                    self.flow_velocity,
                                                    self.pipe_diameter)  # Solve equilibrium at wall
            ionic_elements_source_vector, source_ionic_mass = \
                self.ionic_deposition_elements_mass_sources(sol_wall,
                                                            self.pipe_diameter)  # Get source function for elements
            output_dict['elements'] += ionic_elements_source_vector

        if self.has_nucleation:
            nucleation_elements_source_vector, \
                nucleation_moments_source_vector = self.nucleation_sources(
                    sol_bulk, TKb)
            # Get sourcce function for elements
            output_dict['elements'] += nucleation_elements_source_vector
            # Get source function for momentts
            output_dict['moments'] += nucleation_moments_source_vector

        if self.has_agglomeration:
            agglomeration_moments_source_vector = self.agglomeration_sources(vols,
                                                                             weights,
                                                                             TKb,
                                                                             self.flow_velocity,
                                                                             self.pipe_diameter,
                                                                             ionic_strength,
                                                                             vrepr)
            # Agglomeration only influence moments
            output_dict['moments'] += agglomeration_moments_source_vector

        if self.has_growth:
            growth_elements_source_vector, \
                growth_moments_source_vector = self.growth_sources(vols, weights, TKb,
                                                                   sol_bulk)
            output_dict['elements'] += growth_elements_source_vector
            output_dict['moments'] += growth_moments_source_vector

        if self.has_particle_deposition:
            particle_moments_source_vector, source_particle_mass = \
                self.particle_deposition_sources(vols, weights, TKb,
                                                 self.flow_velocity,
                                                 self.pipe_diameter,
                                                 ionic_strength,
                                                 vrepr)  # self.particle_deposition_sources
            output_dict['moments'] += particle_moments_source_vector

        # Now we join everything, record and return
        output_vector = self.join_input(
            output_dict)  # Join everything together
        reactions_fluxes_dict = {k: 0 for k in sol_bulk.solid_phase_names} \
            if not self.has_ionic_deposition \
            else sol_wall.reaction_fluxes
        self.make_temporary_record_from_dict(
            reactions_fluxes_dict, 'Jr_')
        self.make_temporary_record_from_dict(sol_bulk.gas_molals, 'molal_')
        self.make_temporary_record_from_dict(sol_bulk.molals, 'molal_')
        self.make_temporary_record_from_dict(sol_bulk.saturation_indexes, "satur_")
        self.make_temporary_record_from_dict(sol_bulk.activities, 'act_')
        self.make_temporary_record_from_dict(sol_wall.molals, 'wall_molal_')
        self.make_temporary_record_from_dict(sol_wall.saturation_indexes, 'wall_satur_')
        self.make_temporary_record_from_dict(sol_wall.activities, 'wall_act_')
        self.make_temporary_record("ionic_strength", sol_bulk.ionic_strength)
        self.make_temporary_record("source_ionic_mass",
                                   0.0 if not self.has_ionic_deposition else source_ionic_mass)
        self.make_temporary_record("source_particle_mass",
                                   0.0 if not self.has_particle_deposition else source_particle_mass)
        self.make_temporary_record("moments", moments)
        self.make_temporary_record("elements", elements_balance)
        self.make_temporary_record("source_elements", output_dict["elements"])
        self.make_temporary_record("source_moments", output_dict["moments"])
        return output_vector

    def ionic_deposition_elements_mass_sources(self, sol_wall: InterfaceSolution,
                                               pipe_diameter: float) -> np.ndarray:
        """

        Parameters
        ----------
        sol_wall : InterfaceSolution
            Solution at the wall.
        pipe_diameter : float
            Diameter of pipe.

        Returns
        -------
        source_vector : np.ndarray
            Source function for ionic deposition.

        """
        wall_scale_params = 1 / \
            (flowproperties.pipe_flux_scale(pipe_diameter))  # 1/m*(m^3/kg)
        interface_elements_sources = {k: -wall_scale_params*v
                                      for k, v in sol_wall.elements_reaction_fluxes.items()
                                      if k not in ['H', 'O']}  # mol/m^2 s
        source_vector = _dict_hstack(
            interface_elements_sources, self.solute_elements)
        source_mass = 0.0
        for phase in self.wall_phases:
            phase_properties = properties.get_solid_phase_properties(phase)
            molar_mass = phase_properties.molar_mass
            source_mass += np.pi*pipe_diameter * \
                molar_mass*sol_wall.reaction_fluxes[phase]
        return source_vector, source_mass

    def nucleation_sources(self, sol_bulk: BulkSolution, TKb: float) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        sol_bulk : BulkSolution
            Solution at the. bulk
        TKb : float
            Bulk temperature.

        Returns
        -------
        element_source_vector : np.ndarray
            Source vector for elements.
        moments_source_vector : np.ndarray
            Source vector for moments.

        """
        solid_indexes = self.interface_system.get_solid_indexes(
            self.bulk_phases)
        element_balance_vector = \
            self.interface_system.formula_matrix @\
            self.interface_system.solid_stoich_matrix[solid_indexes, :].transpose(
            )
        # J_r to J_e
        elements_balance_vector = element_balance_vector[2:-1, :]
        element_source_vector = np.zeros(elements_balance_vector.shape[0])
        moments_source_vector = np.zeros(self.nmoments)
        rates_and_volumes = dict()
        for phase in self.bulk_phases:
            rate, vol = self._get_nucleation_rate_and_volume(
                sol_bulk, TKb, phase)
            rates_and_volumes[phase] = (rate, vol)
        for i, phase in enumerate(self.bulk_phases):
            phase_properties = properties.get_solid_phase_properties(phase)
            molar_mass = phase_properties.molar_mass
            density = phase_properties.density
            rate, vol = rates_and_volumes[phase]
            element_source_vector += - \
                elements_balance_vector[:, i]/molar_mass*density*rate*vol
            moments_source_vector += rate*vol**self.kmoments
        return element_source_vector, moments_source_vector

    def agglomeration_sources(self, vols: np.ndarray, weights: np.ndarray,
                              TK: float, flow_velocity: float,
                              pipe_diameter: float,
                              ionic_strength: float,
                              vrepr: float) -> np.ndarray:
        """

        Parameters
        ----------
        vols : np.ndarray
            Volumes array of population balance.
        weights : np.ndarray
            Weights array of population balance.
        TK : float
            Temperature in Kelvin.
        flow_velocity : float
            Flow velocity in m/s.
        pipe_diameter : float
            Diameter in m.
        ionic_strength : float
            Ionic strength at the bulk, in molals
        vrepr : float
            Representative volume for calculating interactions
        Returns
        -------
        source_agg : np.ndarray
            Return moments source for agglomeration.

        """
        dynamic_viscosity = flowproperties.water_dynamic_viscosity(TK)
        kinematic_viscosity = flowproperties.water_kinematic_viscosity(TK)
        turbulent_dissipation = flowproperties.turbulent_dissipation(
            flow_velocity, pipe_diameter, TK)
        komolgorov_length = flowproperties.komolgorov_length(
            flow_velocity, pipe_diameter, TK)
        const_turb = constants.TURB_AGGLOMERATION_CONSTANT
        const_turb *= self.turbulent_agglomeration_correction
        interactions = self.agglomeration_interactions  # Set to true
        vdw_interactions = self.van_der_walls_interactions
        dl_interactions = self.double_layer_interactions
        if interactions:
            if len(self.bulk_phases) > 1:
                warnings.warn(
                    "More than one bulk phase. Assuming properties of the first one")
            phase = self.bulk_phases[0]
            phase_properties = properties.get_solid_phase_properties(phase)
        hamaker = None if not (interactions and vdw_interactions) \
                    else phase_properties.get_hamaker_constant(TK)
        debye_length = None if not (interactions and dl_interactions) \
                        else phase_properties.get_debye_length(ionic_strength, TK)
        phi_dl = None if not (interactions and dl_interactions) else \
                    phase_properties.double_layer_potential
        permittivity = None if not (interactions and dl_interactions) else \
                        constants.WATER_PERMITTIVITY

        vols_n11, vols_1n1 = vols.reshape(-1, 1, 1), vols.reshape(1, -1, 1)
        agglomeration_kernel = agglomeration.agglomeration_rate(
            vols_n11, vols_1n1,
            TK, dynamic_viscosity,
            kinematic_viscosity,
            turbulent_dissipation,
            const_turb,
            komolgorov_length,
            vrepr,
            hamaker=hamaker,
            permittivity=permittivity,
            phi_dl=phi_dl,
            dl_thickness=debye_length,
            interactions=interactions)
        # (1,n,1),#(n,1,1)
        weights_n11, weights_1n1 = weights.reshape(
            -1, 1, 1), weights.reshape(1, -1, 1)
        weights_matrix = weights_n11*weights_1n1  # (n,n,1)
        vols_matrix = 0.5*((vols_n11 + vols_1n1)**self.kmoments -
                           2*vols_1n1**self.kmoments)  # (n,n,k)
        agg_tensor = weights_matrix*vols_matrix*agglomeration_kernel  # (n,n,k)
        source_agg = agg_tensor.sum(axis=-2).sum(axis=-2)  # (k,)
        # First moment variaton due to coag is analytically zero.
        source_agg[1] = 0.0
        if self.nzeroagg is not None and self.nmoments > self.nzeroagg:
            source_agg = np.hstack([source_agg[:self.nzeroagg],
                                    np.zeros(self.nmoments-self.num_zero_coag), ])
        return source_agg

    def particle_deposition_sources(self, vols: np.ndarray,
                                    weights: np.ndarray,
                                    TK: float,
                                    flow_velocity: float,
                                    pipe_diameter: float,
                                    ionic_strength: float,
                                    vrepr: float) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        vols : np.ndarray
            Volumes array of population balance (m3).
        weights : np.ndarray
            Weights array of population balance.
        TK : float
            Temperature in Kelvin.
        flow_velocity : float
            Flow velocity in m/s.
        pipe_diameter : float
            Diameter in m.
        vrepr : float
            Representative volume for calculating interactions
        ionic_strength : float
            Ionic strength at the bulk, in molals

        Returns
        -------
        source_moments_particle_deposition : np.ndarray
            Source for moments.
        source_mass_particle_deposition : np.ndarray
            Source of mass.

        """
        dynamic_viscosity = flowproperties.water_dynamic_viscosity(TK)
        kinematic_viscosity = flowproperties.water_kinematic_viscosity(TK)
        shear_velocity = flowproperties.shear_velocity(
            flow_velocity, pipe_diameter, TK)
        bturb = constants.TURB_VISCOSITY_CONSTANT
        interactions = self.particle_deposition_interactions
        vdw_interactions = self.van_der_walls_interactions
        dl_interactions = self.double_layer_interactions
        if interactions:
            if len(self.bulk_phases) > 1:
                warnings.warn(
                    "More than one bulk phase. Assuming properties of the first one")
            phase = self.bulk_phases[0]
            phase_properties = properties.get_solid_phase_properties(phase)
        hamaker = None if not (interactions and vdw_interactions) \
                    else phase_properties.get_hamaker_constant(TK)
        debye_length = None if not (interactions and dl_interactions) \
                        else phase_properties.get_debye_length(ionic_strength, TK)
        phi_dl = None if not (interactions and dl_interactions) else \
                    phase_properties.double_layer_potential
        permittivity = None if not (interactions and dl_interactions) else \
                        constants.WATER_PERMITTIVITY
        rcorrection = self.roughness_wall_correction
        particle_deposition_rate_vector = particledeposition.particle_deposition_rate(
            vols,
            TK,
            dynamic_viscosity,
            kinematic_viscosity,
            shear_velocity,
            bturb,
            vrepr,
            hamaker=hamaker,
            permittivity=permittivity,
            phi_dl=phi_dl,
            dl_thickness=debye_length,
            rcorrection=rcorrection,
            interactions=interactions)
        kaux = self.kmoments.reshape(-1, 1)  # (k, 1)
        particle_matrix = vols**kaux*particle_deposition_rate_vector*weights
        source_moments_particle_deposition = -4 / \
            pipe_diameter*particle_matrix.sum(axis=1)
        _, solid_density, _ = self.single_bulk_phase_properties()
        source_mass_particle_deposition = np.pi*pipe_diameter*solid_density *\
            (particle_deposition_rate_vector*vols*weights).sum()
        return source_moments_particle_deposition, source_mass_particle_deposition

    def growth_sources(self, vols: np.ndarray, weights: np.ndarray,
                       TKb: float, sol_bulk: BulkSolution):
        """

        Parameters
        ----------
        vols : np.ndarray
            Volumes array of population balance.
        weights : np.ndarray
            Weights array of population balance.
        TK : float
            Temperature in Kelvin.
        sol_bulk : BulkSolution
            Solution at the bulk.

        Returns
        -------
        source_elements_growth : np.ndarray
            Source for elements.
        source_moments_growth : np.ndarray
            Source for moments.

        """
        growth_rate_vectors = []
        for phase in self.bulk_phases:
            phase = self.bulk_phases[0]
            phase_properties = properties.get_solid_phase_properties(phase)
            growth_rate_vector = growth.single_phase_growth_rate(vols, sol_bulk, TKb,
                                                                 phase,
                                                                 phase_properties)  # m3/s
            growth_rate_vectors.append(growth_rate_vector)
        growth_rate_vector = sum(growth_rate_vectors)

        solid_indexes = self.interface_system.get_solid_indexes(
            self.bulk_phases)
        element_balance_vector = \
            self.interface_system.formula_matrix @\
            self.interface_system.solid_stoich_matrix[solid_indexes, :].transpose(
            )

        elements_balance_vector = element_balance_vector[2:-1, :]
        # (n-1, 1) #zero-th moment source will be zero by definition
        k_aux = self.kmoments[1:].reshape(-1, 1)
        source_moments_growth_ = (
            k_aux*growth_rate_vector*vols**(k_aux - 1)*weights).sum(axis=-1)  # (n-1,)
        source_moments_growth = np.hstack([0, source_moments_growth_])

        source_elements_growth = np.zeros(elements_balance_vector.shape[0])
        _, _, molar_vol = self.single_bulk_phase_properties()
        for i, phase in enumerate(self.bulk_phases):
            phase_growth_rate_vector = growth_rate_vectors[i]
            source_elements_phase_growth = elements_balance_vector[:, i]*molar_vol*np.sum(
                phase_growth_rate_vector*weights).sum()
            source_elements_growth += source_elements_phase_growth
        return source_elements_growth, source_moments_growth

    def _solve_bulk_equilibrium(self, elements_balance: Dict[str, float],
                                TK: float,
                                P: float,
                                update_guess: bool = True) -> BulkSolution:
        """


        Parameters
        ----------
        elements_balance : Dict[str, float]
            Maps from element to element concentration in mol/kg H2O.
        TK : float
            Temperature in Kelvin.
        P : float
            Pressure in Pa.
        update_guess : bool, optional
            Whether to update guess for sequential solving. The default is True.

        Returns
        -------
        sol : BulkSolution
            Solution at the bulk.

        """
        # TODO: Variable initial guess
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

    def _solve_wall_equilibrium(self, sol_bulk: BulkSolution,
                                TK: float,
                                flow_velocity: float,
                                pipe_diameter: float,
                                update_guess: bool = True) -> InterfaceSolution:
        """

        Parameters
        ----------
        sol_bulk : BulkSolution
            DESCRIPTION.
        TK : float
            DESCRIPTION.
        flow_velocity : float
            DESCRIPTION.
        pipe_diameter : float
            DESCRIPTION.
        update_guess : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        sol : InterfaceSolution
            DESCRIPTION.

        """
        shear_velocity = flowproperties.shear_velocity(
            flow_velocity, pipe_diameter, TK)
        molals_bulk = sol_bulk.solute_molals
        transport_params = {'type': 'pipe', 'shear_velocity': shear_velocity}
        sol, sol_stats = self.interface_system.solve_interface_equilibrium(TK,
                                                                           molals_bulk,
                                                                           transport_params,
                                                                           initial_guess=self.wall_initial_guess)
        if update_guess:
            self.wall_initial_guess = sol_stats['x']
        return sol

    def _solve_sphere_equilibrium(self, sol_bulk: BulkSolution,
                                  repr_radius: float,
                                  TK: float,
                                  update_guess: bool = True) -> InterfaceSolution:
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

    def _get_nucleation_rate_and_volume(self, sol_bulk: BulkSolution,
                                        TKb: float,
                                        phase_name: str) -> Tuple[float, float]:
        satur = 10**sol_bulk.saturation_indexes[phase_name]
        dynamic_viscosity = flowproperties.water_dynamic_viscosity(TKb)
        phase_properties = properties.get_solid_phase_properties(phase_name)
        sigma_app = phase_properties.surface_energy
        molec_vol = phase_properties.vol
        elem_diam = phase_properties.elementary_diameter
        vol, rate = nucleation.primary_nucleus_size_and_rate(satur, TKb, dynamic_viscosity,
                                                             sigma_app, molec_vol, elem_diam)
        return rate, vol

    def split_input(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """

        Parameters
        ----------
        x : np.ndarray
            DESCRIPTION.

        Returns
        -------
        res : Dict[str, np.ndarray]
            DESCRIPTION.

        """
        index_dict = self.make_index_dictionary()
        res = {key: x[value] for key, value in index_dict.items()}
        return res

    def join_input(self, index_values: Dict[str, np.ndarray]) -> np.ndarray:
        """

        Parameters
        ----------
        index_values : Dict[str, np.ndarray]
            The array map.

        Returns
        -------
        res : np.ndarray
            The joined input as vector.

        """
        index_dict = self.make_index_dictionary()
        res = np.hstack([index_values[key] for key in index_dict.keys()])
        return res

    def make_index_dictionary(self) -> Dict[str, np.ndarray]:
        """

        Returns
        -------
        index_dict : Dict
            Dictionary mapping the property name to its indexes
            {"elements": [0, 1, 2, 3], "moments": [4, 5]}

        """
        # elements concentrations, moments
        # For instance, in Ca, C, Na, Cl, returns 4
        nsolutes = len(self.solute_elements)
        nmoments = self.nmoments  # For instance, in Calcite, returns [2]
        chunks = np.array([nsolutes, nmoments]).astype(int)  # [0, 1, ...]
        # [0, 1, 2, 3], [4, 5, 6]
        indexes = _chunk_split(np.arange(sum(chunks)), chunks)
        index_dict = {'elements': indexes[0], 'moments': indexes[1]}
        return index_dict

    def set_temperature_function(self, temperature: Temperature, wall_temperature: Temperature):
        if callable(temperature):
            self.bulk_temperature_function = temperature
        else:
            self.bulk_temperature_function = lambda x: temperature
        if wall_temperature is None:
            self.wall_temperature_function = self.bulk_temperature_function
        elif callable(wall_temperature):
            self.wall_temperature_function = wall_temperature
        else:
            self.wall_temperature_function = lambda x: wall_temperature

    def set_pressure_function(self, pressure: Pressure, pressure_model: Optional[str]):
        TK = self.bulk_temperature_function(0.0)
        flow_velocity = self.flow_velocity
        pipe_diameter = self.pipe_diameter
        if pressure_model is None:
            if callable(pressure):
                pfunc = pressure
            else:
                def pfunc(x): return pressure
        elif pressure_model == 'dw_inlet':
            dw = flowproperties.darcy_weisbach(
                flow_velocity, pipe_diameter, TK)

            def pfunc(x): return pressure - dw*x
        elif pressure_model == 'dw_outlet':
            assert self.pipe_length is not None
            dw = flowproperties.darcy_weisbach(
                flow_velocity, pipe_diameter, TK)

            def pfunc(x): return pressure + dw*(self.pipe_length - x)
        self.pressure_function = pfunc

    def get_elements_balance_dict(self, x: np.ndarray, TKb: float) -> Dict[str, np.ndarray]:
        """
        Get elements balance, from array in mol/m3 to molal

        Parameters
        ----------
        x : np.ndarray
            Array of balances in mol/m3.
        TKb : float
            DESCRIPTION.

        Returns
        -------
        dict
            Array mapping element to concentration in mol/kg H2O.

        """
        return {el: x[i]/flowproperties.water_density(TKb) for i, el in enumerate(self.solute_elements)}

    def get_concentration_vector(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        return _dict_hstack(d, self.solute_elements)

    def get_delta_moment_vector(self, nparticles: float, diamparticles: float) -> np.ndarray:
        radiusparticles = diamparticles/2
        volparticles = 4.0/3*np.pi*radiusparticles**3
        moments = [nparticles * volparticles **
                   k for k in np.arange(self.nmoments)]
        return np.array(moments)

    def get_concentration_moments_vector(self, elements_dict: Dict[str, np.ndarray],
                                         nparticles: float,
                                         volparticles: float) -> np.ndarray:
        return np.hstack([self.get_concentration_vector(elements_dict),
                          self.get_delta_moment_vector(nparticles, volparticles)])

    def single_bulk_phase_properties(self) -> Tuple[float, float]:
        if len(self.bulk_phases) > 1:
            warnings.warn(
                "More than one bulk phase. Using the first one as representative")
        phase = self.bulk_phases[0]
        phase_properties = properties.get_solid_phase_properties(phase)
        molar_mass = phase_properties.molar_mass
        density = phase_properties.density
        molar_vol = phase_properties.molar_vol
        return molar_mass, density, molar_vol

    @property
    def set_wall_phases(self):  # FIXME: Put actual parameters
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
        assert n % 2 == 0
        self._nmoments = n

    @property
    def kmoments(self):
        return np.arange(self.nmoments, dtype=float)

    def make_temporary_space_record(self, value: float):
        self.recorder_space_key = value

    def make_temporary_record(self, key: str, value: numpy.typing.ArrayLike):
        self.temporary_recorder[key] = value

    def make_temporary_record_from_dict(self, d: Dict[str, numpy.typing.ArrayLike], prefix: str = ''):
        for key, value in d.items():
            self.temporary_recorder[prefix + str(key)] = value

    def turn_temporary_record_into_permanent(self):
        self.permanent_recorder[self.recorder_space_key] = self.temporary_recorder
        self.temporary_recorder = dict()


# Auxiliary functions
def _chunk_split(ary: np.ndarray, chunks: List[int], axis: Optional[int] = 0):
    assert ary.shape[axis] == sum(chunks)
    return np.split(ary, np.cumsum(chunks, dtype=int), axis=axis)[:-1]


def _dict_hstack(d: Dict[Any, numpy.typing.ArrayLike], order: List[Any]):
    return np.hstack([d[o] for o in order])
