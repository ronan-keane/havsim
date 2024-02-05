# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 01:44:13 2018
Testing and debugging functions.
I also use this script to benchmark the speed and accuracy of the adjoint method.
There are also examples of how to setup the optimization problem and call different algorithms. However these examples are pretty deprecated, refer
to files like adjointcontent, relaxcontent or algtesting2 for more up to date optimization routines being called

can verify that in discrete time the gradient isn't necessarily continuous, specifically there are issues with how many
timesteps get put into each gradient calculation, simplest example is relaxatino phenomenon
around 5 i.e. 5-1e-8 adjoint gets 4 timesteps with sensitivity, 5+1e-8 it gets 5 timesteps
with sensitivity, and so you get an extra kick to the total gradient when this happens because
of the extra timestep.
@author: rlk268
"""

from havsim.old.opt import platoonobjfn_obj, platoonobjfn_objder, platoonobjfn_fder
from havsim.helper import makeleadfolinfo
from havsim.old.models import IDM_b3, IDMadjsys_b3, IDMadj_b3, OVM, OVMadjsys, OVMadj, daganzo, daganzoadjsys, daganzoadj
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sc

#make the platoons and platooninfo, as well as get the measurements in dictionary form
#meas, platooninfo, platoonlist = makeplatoonlist(data, 10)
#meas, platooninfo, platoonlist = makeplatoonlist(rawdata, 22, False, [5,11,14,15,8,13,12])

#sim = copy.deepcopy(meas) #simulation is initialized to be the same as the measurements

#pguess = [16.8*3.3,.086/3.3, 1.545, 2, .175 ] #this is what we are using for the initial guess for OVM; it comes from the bando et al paper 'phenomological study ...'
##mybounds = [(30,100),(.01,.05),(.3,2),(1,3),(0,.5)] #conservative bounds
#mybounds = [(30,200),(.001,.1),(.1,10),(.1,5),(0,2)] #less conservative bounds



#%%#get a specific platoon, get the platoon length, make the lead/fol info, and initialize the parameters for all the vehicles in the platoon
sim = copy.deepcopy(meas)
# pguess = [16.8*3.3,.086/3.3, 1.545, 2, .175 ] #OVM

pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5]
#mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] #less conservative bounds #can mess with the bounds to make them loser (may get weird stuff but in general better) or tighter (can cause higher values)
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds



# pguess =  [40,1,1,3,10,25] #IDM
# mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]

#pguess =  [24,1,1,3,15] #IDM
#mybounds = [(20,120),(.1,5),(.1,25),(.1,20),(.1,20)]

args = (True,6)
#args = (False,5)

# args = (True,4)
#
# pguess = [1,40,100,5]
# mybounds = [(.1,10),(0,100),(40,120),(.1,75)]

#curplatoon = platoonlist[93]
#curplatoon = [[], 995,998,1013,1023,1030]  #[[],995,998,1013,1023,1030] this is a good test platoon
#curplatoon = [[],995,998,1013,1023] #995 good for testing lane changing #1003 1014 was original pair we used for testing where 1014 was the follower
#curplatoon = [[],581, 611]
# curplatoon = [381.0, 391.0, 335.0, 326.0, 334.0]
curplatoon = [1602]
#curplatoon = [335, 326]
#curplatoon = platoonlist[17]
n = len(curplatoon)

leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,meas)
#leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,meas)
p = np.tile(pguess, n)
# p = [10*3.3,.086/3.3, 1.545, 2, .175, 5.01, 9*3.3,.083/3.3, 2, 1.5, .275, 15.01, 11*3.3,.075/3.3, 1.545, 2.2, .175, 25.01,
#           10.5*3.3,.086/3.3, 1.6, 2, .175, 10.01, 9.6*3.3,.095/3.3, 1.6, 2.1, .255, 8.01]
#p = p[0:12]
bounds = np.tile(mybounds,(n,1))
#p = finitebfgs['x']
##########################################################################################
################test objective and gradient evaluation##################
# model = IDM_b3
# modeladjsys = IDMadjsys_b3
# modeladj = IDMadj_b3
model = OVM
modeladjsys = OVMadjsys
modeladj = OVMadj
# model = daganzo
# modeladjsys = daganzoadjsys
# modeladj = daganzoadj
start = time.time()
obj = platoonobjfn_obj(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo,*args)
end = time.time()
objtime = end-start #note that the obj is supposed to be around 500k for the initial platoon with 5 vehicles, initial guess for ovm

start = time.time()
adjder = platoonobjfn_objder(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,*args)
#adjder = platoonobjfn_der(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,*args)
end = time.time()
adjdertime = end-start
adjder = adjder[1]

start = time.time()
finder = platoonobjfn_fder(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,*args)
end = time.time()
findertime = end-start

acc = np.linalg.norm(adjder-finder)/np.linalg.norm(finder)
acc2 = np.divide(adjder-finder,finder)
print('accuracy in norm is '+str(acc))
print(acc2)

#############test calibration#####################

start = time.time()
# bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,*args),0,bounds,maxfun=200)
GA = sc.differential_evolution(platoonobjfn_obj,mybounds,(model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo, *args))
end = time.time()
bfgstime = end-start

# ### comparison purposes
print('time for objective is '+str(objtime))
t_nstar, t_n, T_nm1 = platooninfo[curplatoon[0]][:3]
# print('time for calibration is '+str(bfgstime)+' with mse '+str((bfgs[1]/((T_nm1-t_n+1)*.1))))
print('time for calibration is '+str(bfgstime)+' with mse '+str(GA['fun']/((T_nm1-t_n+1)*.1)))
plt.plot(sim[curplatoon[0]][t_n-t_nstar:T_nm1+1-t_nstar,3])
plt.plot(meas[curplatoon[0]][t_n-t_nstar:T_nm1+1-t_nstar,3])

start = time.time()
#sqp = SQP2(platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der,p,bounds,nmbacktrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,5), maxit = 200, t=2, eps=5e-7)
end = time.time()
sqptime = end-start

start = time.time()
#gd = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,nmbacktrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,5),t=3, eps=5e-7,srch_type=1,proj_type=0, maxit = 1000, c1=1e-4)
end = time.time()
gdtime = end-start

#re_diff(sim,platooninfo,curplatoon)

#bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,*args),0,mybounds)

#GA = sc.differential_evolution(platoonobjfn_obj,mybounds,(model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo))


#start = time.time()
#bfgs = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'L-BFGS-B',None,None,None)
#end = time.time()
#bfgstime = end-start
#
#start = time.time()
#bfgs2 = sc.minimize(platoonobjfn_obj,p,(OVM_b, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'L-BFGS-B',platoonobjfn_fder,None,None)
#end = time.time()
#bfgstime2 = end-start
#
#start = time.time()
#bfgs = sc.minimize(platoonobjfn_obj,p,(OVM_b, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'L-BFGS-B',platoonobjfn_fder,None,None)
#end = time.time()
#bfgstime = end-start

#start = time.time()
#bfgs3 = sc.minimize(platoonobjfn_obj,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'L-BFGS-B',platoonobjfn_der,None,None)
#end = time.time()
#bfgs3time = end-start

#start = time.time()
#bfgs4 = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
#end = time.time()
#bfgstime4 = end-start
#
#start = time.time()
#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,mybounds)
#end = time.time()
#bfgstime = end-start
#
#start = time.time()
#bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,mybounds)
#end = time.time()
#bfgstime2 = end-start
#
#start = time.time()
#NM2 = sc.minimize(platoonobjfn_obj,p,(OVM_b, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'Nelder-Mead',options = {'maxfev':3000})
#end = time.time()
#NMtime2 = end-start
#
#start = time.time()
#NM = sc.minimize(platoonobjfn_obj_b,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,mybounds),'Nelder-Mead',options = {'maxfev':10000})
#end = time.time()
#NMtime = end-start
#
#start = time.time()
#NM2 = sc.minimize(platoonobjfn_obj_b,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,mybounds),'Nelder-Mead',options = {'maxfev':10000})
#end = time.time()
#NMtime2 = end-start

#start = time.time()
#finitebfgs = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo, True, 6),'L-BFGS-B',None,None,None,bounds)
#end = time.time()
#finitebfgstime = end-start

#start = time.time()
#NM = sc.minimize(platoonobjfn_obj,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'Nelder-Mead')
#end = time.time()
#NMtime = end-start
#
#start = time.time()
#GA = sc.differential_evolution(platoonobjfn_obj,bounds,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo))
#end = time.time()
#GAtime = end-start

#start = time.time()
#GA2 = sc.differential_evolution(platoonobjfn_obj,bounds,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6))
#end = time.time()
#GAtime2 = end-start
#
#start = time.time()
#slsqp = sc.minimize(platoonobjfn_obj,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'SLSQP',platoonobjfn_der,None,None,bounds)
#end = time.time()
#sqptime = end-start
#
#start = time.time()
#finitesqp = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'SLSQP',None,None,None,bounds)
#end = time.time()
#finitesqptime = end-start

#start = time.time()
#
##TNC = sc.fmin_tnc(platoonobjfn_obj,p,platoonobjfn_fder, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo, *args),0, bounds)
#TNC = sc.fmin_tnc(platoonobjfn_objder,p,None, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo, *args),0, bounds)
#
#end = time.time()
#TNCtime = end-start
#
#obj = platoonobjfn_obj(TNC[0],OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#
##obj = platoonobjfn_obj(TNC[0],OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo, *args)
#print(obj)
#start = time.time()
#newtonCG = sc.minimize(platoonobjfn_obj,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'Newton-CG',platoonobjfn_der)
#end = time.time()
#newtonCGtime = end-start

#start = time.time()
#newtonCG = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'trust-constr',platoonobjfn_der, '2-point', None, bounds)
#end = time.time()
#newtonCGtime = end-start



