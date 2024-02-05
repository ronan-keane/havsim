
"""
@author: rlk268@cornell.edu

For the best control model we found (IDM) evaluate its performance on different equilibrium setups
"""

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import *
import matplotlib.pyplot as plt 
import scipy.optimize as sc

#%%
#define function that will do this testing for us 
def robusttest(v):
    
    #first get the initial state
    p = [33.33, 1.2, 2, 1.1, 1.5]
    initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = v, perturb = 1)
    sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
    vlist = {i: curstate[i][1] for i in curstate.keys()}
    indlist = [min(vlist, key=vlist.get)]
    
    p2 = [33.33, 1.2, 2, 1.1, 1.5] #IDK just guess some parameters for IDM
    args = (curstate, auxinfo, roadinfo,indlist, IDM_b3_b, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
    bounds = [(v,20+v), (.5, 1.8), (1, 3), (.5, 3), (.5, 3)]
    res = sc.minimize(simcir_obj, p2, args = args, method = 'l-bfgs-b', bounds = bounds)
    
    return res, curstate, auxinfo, roadinfo, indlist

for v in range(5,15):
    out = robusttest(v)
    filepath = 'C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/'
    filepath = filepath+str(v)+'robust.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(out,f)
#%%
with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/5robust.pkl', 'rb') as f:
    out = pickle.load(f)