# -*- coding: utf-8 -*-
from typing import Dict, Any, List

import numpy as np

from scipy import interpolate


class ResultInterpolator():
    def __init__(self, recorder : Dict[float, Dict[str, np.typing.ArrayLike]]):
        self.recorder = recorder
        
    def result_keys(self) -> List[str]:
        return self.base_keys() + self.derived_keys()
    
    def value_at(self, item : str, xval : np.ndarray) -> np.ndarray:
        try:
            assert item in self.result_keys()
        except:
            raise KeyError("Cannot find item")
        x, y = map(np.array,
                   zip(*[(xx, self.get_value(d, item)) for xx, d in self.recorder.items()]))
        return interpolate.interp1d(x, y, axis=0, fill_value="extrapolate")(xval)
    
    def get_value(self, d : Dict[str, np.typing.ArrayLike], k : str):
        """
        An extended get function where if 

        Parameters
        ----------
        d : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if k in self.base_keys():
            return d[k]
        elif k in self.derived_keys():
            if k == 'source_mass':
                return d['source_ionic_mass'] + d['source_particle_mass']
            elif k == 'particle_deposition_proportion':
                return d['source_particle_mass']/(d['source_ionic_mass'] + d['source_particle_mass'])
            elif k == 'ionic_deposition_proportion':
                return d['source_ionic_mass']/(d['source_ionic_mass'] + d['source_particle_mass'])
            elif k == 'mean_volume':
                return d['moments'][1]/d['moments'][0]
            elif k == 'number_particles':
                return d['moments'][0]
            elif k == 'mean_diameter_estimate':
                v = d['moments'][1]/d['moments'][0]
                return (3/(4*np.pi)*v)**(1.0/3)
            else:
                raise KeyError
        else:
            raise KeyError
        
    def base_keys(self) -> List[str]:
        return list(next(iter(self.recorder.values())).keys())
    
    def derived_keys(self) -> List[str]:
        return ['source_mass',
                'particle_deposition_proportion',
                'ionic_deposition_proportion',
                'mean_volume',
                'number_particles',
                'mean_diameter_estimate']