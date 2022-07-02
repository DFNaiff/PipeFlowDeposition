# -*- coding: utf-8 -*-
import numpy as np
import pyequion2

from . import properties


BulkSolution = pyequion2.solution.SolutionResult
PhaseProperties = properties.SolidPhaseProperties


def single_phase_growth_rate(x : np.ndarray,
                             bulk_solution : BulkSolution,
                             temp : float,
                             phase : str,
                             phase_properties : PhaseProperties,
                             adjustment_factor : float=1.0) -> np.ndarray:
    """

    Parameters
    ----------
    x : np.ndarray
        Volumes [m3].
    bulk_solution : BulkSolution
        Bulk chemical solution of aqueous system.
    temp : float
        Temperature [K].
    phase : str
        Phase considered.
    phase_properties : PhaseProperties
        Properties of phase considered.
    adjustment_factor : float, optional
        Adjustment factor. The default is 1.0.

    Returns
    -------
    g : np.ndarray
        Volumetric growth rate [m3/s].

    """
    r = (3/(4*np.pi)*x)**(1.0/3)
    area = 4*np.pi*r**2
    satur = 10**bulk_solution.saturation_indexes[phase]
    vertical_growth = phase_properties.growth_function(satur, temp)
    g = adjustment_factor*vertical_growth*area
    return g
