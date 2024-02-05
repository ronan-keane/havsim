# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:14:06 2019

@author: rlk268
"""
#find good bounds / initial guesses for IDM. There are lists of difficult vehicles to calibrate in randomtesting "LCbad" is a list of veh IDs
from calibration import * 
plt.close('all')
pguess = [1,20,100,5] #daganzo guess 1 


mybounds = [(.1,10),(0,100),(40,120),(.1,75)] #fairly conservative bounds

sim = copy.deepcopy(meas)
#curplatoon = [[],1821] #vehicle 435 (10) very wonky for some reason #754 very wonky (13)

#testveh = 1 #21
#curplatoon = LClist[LCbad[testveh]]

#curplatoon = noLClist[1]

#curplatoon=LClist[15]
#print(curplatoon)
curplatoon = [[],1013] #603 a good example of many lane changes #50 an example of calibration not working well #156
#curplatoon =LClist2[1]
print(curplatoon)
n = len(curplatoon[1:])

#print('testing vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s).')
#print(' originally adjoint found an obj. value of '+str(out[LCbad[testveh]][1])+' no relax adjoint found '+str(out2[LCbad[testveh]][1]))

leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

start = time.time()
bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(daganzo, daganzoadjsys, daganzoadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4),0,mybounds)
end = time.time()
bfgstime = end-start
re_diff(sim,platooninfo,curplatoon)
plt.figure()
plotspeed(meas,sim,platooninfo,curplatoon[1])
plt.figure()
plotdist(meas,sim,platooninfo,curplatoon[1])

#############
saver = rinfo.copy()
print(rinfo)

#start = time.time()
#bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
#end = time.time()
#bfgstime3 = end-start

pguess = [1,20,100] #daganzo guess 1 
mybounds = [(.1,10),(0,100),(40,120)] #fairly conservative bounds
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,sim) 

start = time.time()
bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(daganzo, daganzoadjsys, daganzoadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,3),0,mybounds)
end = time.time()
bfgstime2 = end-start
re_diff(sim,platooninfo,curplatoon)
plt.figure()
plotspeed(meas,sim,platooninfo,curplatoon[1])
plt.figure()
plotdist(meas,sim,platooninfo,curplatoon[1])

print('adjoint with new guess found '+str(bfgs[1]))
#print(' finite with new guess found '+str(bfgs3[1]))
print(' adjoint with new guess no relax found '+str(bfgs2[1]))


#%%


#%%
#
#mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
#leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 
#start = time.time()
#GA = sc.differential_evolution(platoonobjfn_obj,mybounds,(IDM_b3, IDMadjsys_b3, IDMadj_b3, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6))
#end = time.time()
#GAtime = end-start

