# -*- coding: utf-8 -*-

from . import constants


class PipeFlowParams(object):
    def __init__(self, flow_velocity, pipe_diameter, pipe_length,
                 TKb, TKw=None, pressure=constants.ATM, pressure_model='dw_outlet'):
        self.flow_velocity = flow_velocity
        self.pipe_diameter = pipe_diameter
        self.TKb = TKb
        self.TKw = TKb if TKw is None else TKb
        self.pressure = pressure
        self.pressure_model = pressure_model