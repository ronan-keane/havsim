
"""
@author: rlk268@cornell.edu
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import matplotlib.pyplot as plt


import havsim
from havsim.simulation.simulationold2 import simulate_step, eq_circular, simulate_cir, update_cir, update2nd_cir
from havsim.simulation.models import drl_reward8, IDM_b3, IDM_b3_eql, FS
from havsim.plotting import plotformat, platoonplot
from toysimulation import debugenv

import copy
import math
import gym
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import time

p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 0, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)
testingtime = 1500
#create simulation environment
testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25)
agent = ACagent(PolicyModel(num_actions=3), ValueModel())

times3=[]
out3 = []
for _ in range(5):
    start = time.time()
    agent.train(testenv, updates=10)
    end = time.time()
    times3.append(end-start)
    out3.append(agent.timecounter)
print("Average time over 5 runs is {:.4f}".format(np.mean(times3)))  #25.1353 eager
print('average time to run environment step method is '+str(np.mean(out3)))
testenv.reset()
start = time.time()
for i in range(64*10):
    out = testenv.step(2, 0, 0, False)
print('time to run environment step method outside training is '+str(time.time()-start))