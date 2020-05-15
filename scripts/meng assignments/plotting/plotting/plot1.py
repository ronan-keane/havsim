
"""
@author: rlk268@cornell.edu
"""
#imports 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy 
import math 
#stuff for calibration 
from havsim.calibration.calibration import calibrate_tnc2, calibrate_GA
from havsim.calibration.helper import makeleadfolinfo, obj_helper
from havsim.calibration.models import OVM, OVMadjsys, OVMadj
from havsim.calibration.opt import platoonobjfn_obj, platoonobjfn_objder
#simulation 
#from havsim.simulation.simulation import eq_circular, simulate_cir, update2nd_cir, update_cir
from havsim.simulation.models import IDM_b3, IDM_b3_eql
#plotting 
from havsim.plotting import platoonplot, plotflows, animatevhd, plotvhd, animatetraj, meanspeedplot, optplot, selectoscillation, plotformat, selectvehID, selectvehID_v2
#data processing
from havsim.calibration.algs import makeplatoonlist
#%% #load data

with open('reconngsim.pkl','rb') as f:
    data = pickle.load(f)[0]
with open('platoonlist.pkl','rb') as f:
    platoonlist = pickle.load(f) #platoonlist = list of platoons 
with open('testcalout.pkl','rb') as f:
    out, rmse = pickle.load(f) #calibration results for the testplatoon 
meas, platooninfo = makeplatoonlist(data,1,False)
testplatoon =[[904.0, 907.0, 914.0, 926.0, 927.0, 939.0],[967.0, 906.0, 928.0, 931.0],[973.0, 983.0, 987.0, 997.0, 1004.0, 1025.0, 1032.0]]

#example output from calibration module 
sim = copy.deepcopy(meas)
sim = obj_helper(out,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,testplatoon,makeleadfolinfo,platoonobjfn_obj,(True,6))

#animatetraj(meas, platooninfo, testplatoon[0])

#%%
#example call signature of platoonplot 
#platoonplot(meas,None, platooninfo, platoon = testplatoon[0], colorcode=True, speed_limit=[20,35]) #single platoon, can specify colorcoding speeds
#platoonplot(meas,None,platooninfo, platoon=platoonlist[48:60], colorcode = True) #list of platoons
#platoonplot(meas,None,platooninfo, platoon=platoonlist[48:60], colorcode = True, lane = 2, opacity = .1) #list of platoons with specific lane
platoonplot(meas,None,platooninfo, platoon=platoonlist[48:60], colorcode = False, lane = 2, opacity = .1, fulltraj=False, timerange=[2800,3250]) #list of platoons with specific lane

#other main call signature is when you give meas and sim, and then you can compare the two
platoonplot(meas, sim, platooninfo, platoon = testplatoon[0:2], colorcode = False, lane = 4 )
#platoonplot(meas, None, platooninfo, platoon = testplatoon[0:2], colorcode = True, lane = 4 )


#%% plotvhd and animatevhd\_list
#meas, platooninfo, platoonlist = makeplatoonlist(data,10) #this takes like 3 minutes to run

#plotvhd(meas,None,platooninfo,[928,931,967],draw_arrow=True, arrow_interval=20)
plotvhd(meas,None,platooninfo,platoonlist[50][:4],plot_color_line=True, draw_arrow = True, arrow_interval = 20)
#plotvhd(meas,None,platooninfo,platoonlist[50][:4],draw_arrow=True, arrow_interval=10)


#animatevhd(meas, None, platooninfo, [928, 931, 967])

#animatetraj(meas, platooninfo, platoon = testplatoon[0])
#animatetraj(meas, platooninfo, platoon = testplatoon[0], interval = 10)


#%%
#example output from simulation module
#p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
#initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
#sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 15000, dt = .25)
#%%
#right now to plot something from simulation looks like this
#sim2, platooninfo2 = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1)
#platoonplot(sim2,None,platooninfo2,platoon=[], lane=1, colorcode= True, speed_limit = [0,25])
#plt.ylim(0,roadinfo[0])

#meanspeedplot(sim2, 50, 8)
#%%
#plotvhd(sim2,None,platooninfo,1)
#%% #for selectoscillation/selectvehID
#option 1 - select rectangular area, press n to call selectvehID with chosen region (will automatically get vehicles in chosen region top left vertex)
times, x, lane, veh = selectoscillation(meas,xint = 40,lane=3)
#make box, press n to enter selectvehID
#a and z keys add vehicles before and after, respecitvely

#option 2 - call selectvehID directly, but need to specify a starting vehicle list and also the rectangular region 
#gives error? 
test2 = [[(5562.474476963611, 1476.8050669428), (6311.045414797408, 164.0527611552), (7203.064516129032, 164.0527611552), (6454.493578295235, 1476.8050669428)]]
test3 = [224.0, 194.0, 244.0, 240.0, 249.0, 255.0, 260.0, 267.0, 257.0]
test4 = [1763, 1756, 1791, 1794]
selectvehID_v2(data,times,x,3,veh,test2,test4)
