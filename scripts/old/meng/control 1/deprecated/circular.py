
"""
@author: rlk268@cornell.edu
"""

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import *

import matplotlib.pyplot as plt 
from havsim.plotting import plotformat, platoonplot

import scipy.optimize as sc
import math 


#%%
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers 
#specify the model as IDM_b3, have 41 vehicles with a vehicle length of 2 (meters), vehicles start evenly spaced at 15 m/s 
#the length of the circular road is chosen so that it is possible for all vehicles to stay to 15 m/s, this is 
#called the equilibrium solution and corresponds to the highest possible flow state
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road 

sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25) #perform 10000 timesteps of simulation 

#%% plot results 

#all timesteps plotted, density = 2 means every other vehicle
def myplot(sim, auxinfo, roadinfo):
    meas, platooninfo = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1) 
    platoonplot(meas,None,platooninfo,platoon=[], lane=1, colorcode= True, speed_limit = [0,25]) 
    plt.ylim(0,roadinfo[0])

myplot(sim,auxinfo, roadinfo)

#%% control with a parametrized policy example 

#specify the control vehicle as the minimum speed vehicle (arbitrary)
vlist = {i: curstate[i][1] for i in curstate.keys()}
indlist = [min(vlist, key=vlist.get)]
#initial run with default parameters for FS 
obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj([1.5, 1.0, .5, 4.5, 5.25, 6.0, 15,1],curstate, auxinfo, roadinfo,indlist, 
                                                           FS, update2nd_cir, l2v_obj, update_cir, 1500, .25, False)

print('objective function value is '+str(obj))
myplot(testsim,auxinfo2,roadinfo)

#%% optimize parameters

p2 = [1.5, 1.0, .5, 4.5, 5.25, 6.0, 15,1] #guess some parameters for follower stopper controller 
args = (curstate, auxinfo, roadinfo,indlist, FS, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
bounds = [(.4,2),(.4,2),(.4,2),(3,7),(3,7),(3,7),(15,22),(.4,2)]
#can uncomment below 
#res = sc.minimize(simcir_obj, p2, args = args, method = 'l-bfgs-b', bounds = bounds) #optimization problem is solved here, takes some minutes 

#here is the output you need to see the result 
res = {}
res['x'] = [ 2.        ,  0.40000002,  0.4       ,  3.        ,  3.        ,
        6.99999994, 15       ,  1.99999997]

#%%
#run with the optimized parameters for FS controller 
obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(res['x'],curstate, auxinfo, roadinfo,indlist, 
                                                           FS, update2nd_cir, l2v_obj, update_cir, 1500, .25, False)

print('optimized objective value is '+str(obj))
myplot(testsim, auxinfo2, roadinfo)