
"""
@author: rlk268@cornell.edu

test the new shiny simulation module 

some experiments just to test stabilization of traffic on a circular road. 
I did also some experiments testing using the control policy on a straight road
(these are housed in script simulationtesting.py) and those simulations 
showed that IDM_b3 strategy learned here did not work well in the 
infinite road. Examing the output showed that the learned IDM_b3 strategy
basically consists of accelerating very slowly once you become very slow
in the circular test track. This works fine in the circular test track because
of the periodic boundary. In the infinite road this works terribly, it
actually makes things worse. What you need to do is instead you need to ANTICIPATE
the big oscillation coming and start slowing down far in advance (I think)
#using linearCAV didn't work either so yea

related experiments in robusttesting where I optimize for different
equilibrium speeds (with the same human driver parameters)
"""

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import *
import matplotlib.pyplot as plt 
import scipy.optimize as sc
#%%
#create initial conditions of stop and go traffic on circular road 
#p = [33.33, 1.5, 2, 1.1, 1.5] #4th parameter can change between .9 for absolutely unstable, 1.1 for convectively unstable. 
p = [33.33, 1.2, 2, 1.1, 1.5]
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 1)
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)


#%%
vlist = {i: curstate[i][1] for i in curstate.keys()}
indlist = [min(vlist, key=vlist.get)]
#verify objective function works - also get baseline of original scenario
#obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(p,curstate, auxinfo, roadinfo,indlist, IDM_b3, update2nd_cir, l2v_obj, timesteps = 5000, objonly = False, dt = .25)
#verify objective function works 
#obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(res2['x'],curstate, auxinfo, roadinfo,indlist, linearCAV, update2nd_cir, l2v_obj, update_cir, 5000, .25, False)

#obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(res['x'],curstate, auxinfo, roadinfo,indlist, FS, update2nd_cir, l2v_obj, update_cir, 5000, .25, False)

obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj([1.5, 1.0, .5, 4.5, 5.25, 6.0, 16.4,1],curstate, auxinfo, roadinfo,indlist, FS, update2nd_cir, l2v_obj, update_cir, 5000, .25, False)

#obj, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(res3['x'],curstate, auxinfo, roadinfo,indlist, IDM_b3_b, update2nd_cir, l2v_obj, update_cir, 5000, .25, False)

#%%
#optimize

#using IDM as a control model
#p2 = [33.33, 1.2, 2, 1.1, 1.5] #IDK just guess some parameters for IDM
#args = (curstate, auxinfo, roadinfo,indlist, IDM_b3_b, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
#bounds = [(15,35), (.5, 1.8), (1, 3), (.5, 3), (.5, 3)]
#res = sc.minimize(simcir_obj, p2, args = args, method = 'l-bfgs-b', bounds = bounds)

#follower stopper
p2 = [1.5, 1.0, .5, 4.5, 5.25, 6.0, 15,1] #IDK just guess some parameters for follower stopper
args = (curstate, auxinfo, roadinfo,indlist, FS, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
bounds = [(.4,2),(.4,2),(.4,2),(3,7),(3,7),(3,7),(15,22),(.4,2)]
res = sc.minimize(simcir_obj, p2, args = args, method = 'l-bfgs-b', bounds = bounds)

#linear CAV controller from experimental validation of cav design paper
p2 = [1, .6, 16, .2, .4, 30,70] #IDK just guess some parameters for linearCAV
args = (curstate, auxinfo, roadinfo,indlist, linearCAV, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
bounds = [(.5,5), (.4,.8), (15,22), (.15,.4), (.3,.6),(10,70),(50,200)]
res2 = sc.minimize(simcir_obj, p2, args = args, method = 'l-bfgs-b', bounds = bounds)
#%%

objopt, testsim, curstate2, auxinfo2, modelinfo2 = simcir_obj(res2['x'],curstate, auxinfo, roadinfo,indlist, linearCAV, update2nd_cir, avgv_obj, update_cir, 5000, .25, False)

#%%
def plothelper(sim, cur = 40):
    plt.close('all')
    x, v, s = [], [], []
    for i in range(len(sim[cur])):
        x.append(sim[cur][i][0])
        v.append(sim[cur][i][1])
        s.append(sim[cur][i][2])
    plt.figure()
    plt.plot(x)
    plt.figure()
    plt.plot(v)
    plt.figure()
    plt.plot(s)
    
plothelper(testsim, cur = 21)

#def headwayhelper(sim, auxinfo, L, cur = 41):
#    slist = []
#    for j in range(len(sim[cur])): #penality for collisions
#        lead = auxinfo[cur][1]
#        leadx = sim[lead][j][0]
#        leadlen = auxinfo[lead][0]
#        s = leadx - leadlen - sim[cur][j][0]
#        slist.append(s)
#    plt.figure()
#    plt.plot(slist)
#    return 
#headwayhelper(sim,auxinfo, cur = 7)

#%%
def v_metric(testsim,ind=0):
    v = []
    for i in testsim.keys(): 
        for j in testsim[i][ind:]:
            v.append(j[1])
    print(np.mean(v))
    return
v_metric(testsim)
v_metric(sim)

#%%
import pickle
#results for using IDM_b3 to stabilize ring road
#args = (curstate, auxinfo, roadinfo,indlist, IDM_b3_b, update2nd_cir, avgv_obj, update_cir, 5000, .25, True)
#bounds = [(15,35), (.5, 1.8), (1, 3), (.5, 3), (.5, 3)]
#with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb3.pkl', 'wb') as f:
#    pickle.dump(res,f)

#IDM_b3 with l2v objective
#use l2v instead of avgv
#args = (curstate, auxinfo, roadinfo,indlist, IDM_b3_b, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
#with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb32.pkl', 'wb') as f:
#    pickle.dump(res,f)
with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb32.pkl', 'rb') as f:
    res3 = pickle.load(f)


#linearCAV
#args = (curstate, auxinfo, roadinfo,indlist, linearCAV, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
#with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb33.pkl', 'wb') as f:
#    pickle.dump(res2,f)
with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb33.pkl', 'rb') as f:
    res2 = pickle.load(f)

#Follower stopper
#p2 = [1.5, 1.0, .5, 4.5, 5.25, 6.0, 15,1] #IDK just guess some parameters for follower stopper
#args = (curstate, auxinfo, roadinfo,indlist, FS, update2nd_cir, l2v_obj, update_cir, 5000, .25, True)
#bounds = [(.4,2),(.4,2),(.4,2),(3,7),(3,7),(3,7),(12.5,22),(.4,2)]
#with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb34.pkl', 'wb') as f:
#    pickle.dump(res,f)
with open('C:/Users/rlk268/OneDrive - Cornell University/fall 2019/IFAC conference/IDMb34.pkl', 'rb') as f:
    res = pickle.load(f)