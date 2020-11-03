#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:58:29 2020

@author: rlk268
"""
from havsim.simulation.models import IDM_parameters
import havsim.simulation as hs
import matplotlib.pyplot as plt

cf_p, unused = IDM_parameters()
relaxp = 15
leadveh = hs.Vehicle(-1, None, cf_p, None, maxspeed = cf_p[0]-1e-6)
folveh = hs.Vehicle(0, None, cf_p, None, maxspeed = cf_p[0]-1e-6, relax_parameters=relaxp)
leadspeed = 29 #speed of 29 and headway of 30 choosen to represent typical merging scenario at maximum capacity for the IDM parameters = [33.5, 1.3, 3, 1.1, 1.5]
inithd = 30
initrelax = leadveh.get_eql(leadspeed) - inithd  # this the relax amount as calculated in  set_relax

leadveh.posmem = []
leadveh.speedmem = []
leadveh.pos = 0
leadveh.speed = leadspeed
leadveh.acc = 0

folveh.leadmem = [(leadveh, None,), (None, None)] #oldv = newv = leadspeed
folveh.posmem = []
folveh.speedmem = []
folveh.pos = leadveh.pos-inithd-leadveh.len # new headway = inithd
folveh.speed = leadspeed
folveh.lead = leadveh
folveh.hd = leadveh.get_eql(leadspeed)  #old headway
folveh.set_relax(0, .1)
for i in range(1000):
    folveh.set_cf(i, .1)
    leadveh.update(i, .1)
    folveh.update(i, .1)
    folveh.hd = leadveh.pos - leadveh.len - folveh.pos

plt.subplot(1,2,1)
plt.plot(np.array(leadveh.posmem) - leadveh.len - folveh.posmem)
plt.subplot(1,2,2)
plt.plot(folveh.speedmem)






