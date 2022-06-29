# -*- coding: utf-8 -*-
import numpy as np


class DataRecorder(object):
    def __init__(self):
        self.recorder = dict()
        self.reset_inner_recorders()

    def reset_inner_recorders(self):
        self.inner_recorder = dict()
        self.inner_concentrations_recorder = dict()

    def confirm_records(self):
        for name, value in self.inner_recorder.items():
            if name in self.recorder:
                self.recorder[name].append(value)
            else:
                self.recorder[name] = [value]
        self._confirm_concentration_records()

    def record(self, name, value):
        self.inner_recorder[name] = value

    def record_concentration(self, name, concentrations):
        self.inner_concentrations_recorder[name] = concentrations

    def dump(self):
        return dict([(name, np.array(value))
                     for name, value in self.recorder.items()])

    def _confirm_concentration_records(self):
        for name, concentrations in self.inner_concentrations_recorder.items():
            if name in self.recorder:
                for key, value in self.recorder[name].items():
                    self.recorder[name][key] = np.append(
                        value, concentrations[key])
            else:
                self.recorder[name] = \
                    dict([(specie, np.array([concentration]))
                          for specie, concentration in concentrations.items()])
        return
