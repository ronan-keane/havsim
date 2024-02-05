#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:37:01 2020

@author: rlk268
"""

import havsim.simulation.calibration as hc
import math
import scipy.optimize as sc

testres = [(bfgs[1], count) for count, bfgs in enumerate(relax_lc_res)]
badlist = []
for i in testres:
    if i[0] > 200:
        badlist.append(i)


def training(veh_id, plist, bounds, meas, platooninfo, dt, vehicle_object, cutoff = 6):
    """Runs bfgs with multiple initial guesses to fit parameters for a CalibrationVehicle"""
    #veh_id = float vehicle id, plist = list of parameters, bounds = bounds for optimizer (list of tuples),
    #vehicle_object = (possibly subclassed) CalibrationVehicle object, cutoff = minimum mse required for
    #multiple guesses
    cal = hc.make_calibration([veh_id], meas, platooninfo, dt, vehicle_object)
    bestmse = math.inf
    best = None
    for guess in plist:
        bfgs = sc.fmin_l_bfgs_b(cal.simulate, guess, bounds = bounds, approx_grad=1)
        if bfgs[1] < bestmse:
            best = bfgs
            bestmse = bfgs[1]
        if bestmse < cutoff:
            break
    return best

class NoRelaxIDM(hc.CalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
        # initial conditions
        self.lead = None
        self.pos = self.initpos
        self.speed = self.initspd
        # reset relax
        self.in_relax = False
        self.relax = None
        self.relax_start = None
        # memory
        self.leadmem = []
        self.posmem = [self.pos]
        self.speedmem = [self.speed]
        self.relaxmem = []
        # parameters
        self.cf_parameters = parameters


veh_id = 969
# plist = [[40,1,1,3,10,25], [60,1,1,3,10,5], [80,1,15,1,1,35], [40,1,1,3,10,.1], [30,2.5,15,1,2,35]]
plist = [[40,1,1,3,10,25], [60,1,1,3,10,5], [80,1,15,1,1,35]]
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
# bounds = [(20,120),(.1,3),(.1,20),(.1,10),(.1,10),(.1,60)]

out = training(veh_id, plist, bounds, meas, platooninfo, .1, hc.CalibrationVehicle, cutoff = 6)

plist = [[40,1,1,3,10], [60,1,1,3,10], [80,1,15,1,1]]
# bounds = [(20,120),(.1,3),(.1,20),(.1,10),(.1,10)]
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
out2 = training(veh_id, plist, bounds, meas, platooninfo, .1, NoRelaxIDM, cutoff = 6)
