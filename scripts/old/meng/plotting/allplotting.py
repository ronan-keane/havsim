
"""
@author: rlk268@cornell.edu
summary of plotting api
"""
#imports, load/process data 
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
from havsim.simulation.simulation import eq_circular, simulate_cir, update2nd_cir, update_cir
from havsim.simulation.models import IDM_b3, IDM_b3_eql
#plotting 
from havsim.plotting import  platoonplot, plotflows, plotvhd, animatevhd_list, animatetraj, meanspeedplot, optplot, selectoscillation, plotformat, selectvehID
#data processing
from havsim.calibration.algs import makeplatoonlist
#%%
#load the data from pickle file (you should uncomment this and put in the full/path/to/pickle.pkl as path_)
## ngsim data
with open('reconngsim.pkl', 'rb') as f:
    reconngsim = pickle.load(f)[0]
# highd data
with open('highd26.pkl', 'rb') as f:
    highd = pickle.load(f)[0]

#takes maybe 60 seconds to get platoonlist
meas, platooninfo, platoonlist = makeplatoonlist(reconngsim, 6)
#meas, platooninfo = makeplatoonlist(data,1,False) #you can call like this to get meas/platooninfo fast

testplatoon =[[904.0, 907.0, 914.0, 926.0, 927.0, 939.0],[967.0, 906.0, 928.0, 931.0],[973.0, 983.0, 987.0, 997.0, 1004.0, 1025.0, 1032.0]] #here are some test platoons 
#corresponds to platoonlist[48:51]

#%%
#current frontend api for calibration
plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ], [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
out, unused, rmse = calibrate_tnc2(plist, bounds, meas, platooninfo, testplatoon, makeleadfolinfo, platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, True, 6, order = 1)

#with open('testcalout.pkl','rb') as f:
#    out, rmse = pickle.load(f)
#out = 
    
#%%
#basic example for simulation on circular road 
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers 
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road 
sim2, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25) 

#%%
#platoonplot is basically the main plotting api - it has different use cases
platoonplot(meas,None, platooninfo, platoon = testplatoon[0], colorcode=True, speed_limit=[20,35]) #single platoon, can specify colorcoding speeds
platoonplot(meas,None,platooninfo, platoon=platoonlist[48:60], colorcode = True) #list of platoons
platoonplot(meas,None,platooninfo, platoon=platoonlist[48:60], colorcode = True, lane = 2, opacity = .1) #can specify specific lane, if colorcode = True you probably want the opacity turned off
#another good feature would be able to select vehicles and see their IDs when colorcode is True- currently you can only do that when colorcode is False. 
#another idea is to use different colorkeys for both simulation/measurments so you can compare that way 

#other main use of platoonplot is when sim is not None, this version is meant to be used with colorcode=False
sim = copy.deepcopy(meas)
sim = obj_helper(out,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,testplatoon,makeleadfolinfo,platoonobjfn_obj,(True,6))
platoonplot(meas, sim, platooninfo, testplatoon, colorcode=False, lane = 4, opacity = .1)

#right now I have some code linking simulation code to plotting format but its super hacky and needs to be fixed. 
#
def myplot(sim, auxinfo, roadinfo, platoon= []):
    #note to self: platoon keyword is messed up becauase plotformat is hacky - when vehicles wrap around they get put in new keys
    meas, platooninfo = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1)
    platoonplot(meas,None,platooninfo,platoon=platoon, lane=1, colorcode= True, speed_limit = [0,25])
    plt.ylim(0,roadinfo[0])
    
myplot(sim2, auxinfo, roadinfo)
#want a non hacky version of plotformat and a keyword for putting in time for platoonplot
#need code to handle wrap around for circular roads 
#need code to handle calculating position for networks with multiple roads 

#%%
#here's another plotting function. I don't want to add any to this one 
plotflows(meas,[[800,1200]],[0,10*60*14.5],30*10,type = 'line',lane = 6, method = 'flow')

#%%
plotvhd(meas,sim,platooninfo,904) #want a list version of this like for animate_list - basically just accept list input, do multiple plots
plotvhd(meas,None,platooninfo,904) #the keywrod arguments are messed up - this should not throw an error. 
plotvhd(meas,None,platooninfo, 904, show_sim = False)
#also the plotvhd/animatevhd code is especially bad - it would really be useful to do some clean up. 
#%%
animatevhd_list(meas,None,platooninfo,testplatoon[0],show_sim=False)

animatevhd_list(meas,sim,platooninfo,testplatoon[0]) #some issues - numbering is not being done correctly, (some lines have no labels)
#the scaling on axis isn't right. 
#also, would like to get rid of those horizontal lines which occur when vehicles change lanes. 
#show_meas = False keyword is broken, 
#if meas or sim are None, it should automatically infer its corresponding keyword as false.

animatevhd_list(meas,sim,platooninfo,testplatoon[0], show_meas=False, ) 

animatevhd_list(meas,sim,platooninfo,testplatoon[0], show_meas=False, usestart = 2700,useend = 2900) 
animatevhd_list(meas,sim,platooninfo,testplatoon[0], show_meas=False, usestart = 2400,useend = 2900) 
animatevhd_list(meas,sim,platooninfo,testplatoon[0], show_meas=False, usestart = 2700,useend = 4100) 
#there is some kind of issue with usestart keyword and useend keywords. 
#major job would be getting rid of the jumps in headway which are caused when a vehicle changes lanes

#%%
animatetraj(meas,platooninfo,testplatoon, usetime = list(range(2700,3000)))
animatetraj(meas,platooninfo,testplatoon, speed_limit = [0,30])
#the time is messed up on this one too. Should automatically make the time range/shorter for maximum time and longer for minimum time
#so there are no errors when the time range is too long/short
#i'm happy with this function but eventually there needs to be an extension which can plot networks - need some updated strategy then 
#some nice features would be some way to control the speed of animation, or being able to jump ahead in the animations

#%%
meanspeedplot(data, 50,12, lane = 2)
#happy with this - how to make it work for simulation output though? 
#maybe make it take meas as an input? or is it easier to put simulation output into raw data format ?
#%%
times, x, lane, veh = selectoscillation(data,50,20,lane=3)
#this functino is a mess (the second part of it, selectvehID), it is missing documentation and features, 
#and has some logic bugs 
test2 = [[(5562.474476963611, 1476.8050669428), (6311.045414797408, 164.0527611552), (7203.064516129032, 164.0527611552), (6454.493578295235, 1476.8050669428)]]
test3 = [224.0, 194.0, 244.0, 240.0, 249.0, 255.0, 260.0, 267.0, 257.0] 
selectvehID(data,times,x,3,veh,test2,test3)

#would be nice to have a more general interactive plotting for looking at simulation results
#%%
optplot(out,meas,sim,platooninfo,testplatoon,OVM,OVMadjsys,OVMadj,makeleadfolinfo,platoonobjfn_obj,(True,6), lane = 4)
#this used to work but now it's throwing errors about round method - please fix that 
#there are some other problems too just with the selection/evaluation being buggy 
#in general I would say that a better design is to pass in the simulated data and any metrics, and we just give a way 
#to visually inspect the output and the corresponding metrics, as opposed to trying to have a monolithic function 
#that is trying to do too much 
