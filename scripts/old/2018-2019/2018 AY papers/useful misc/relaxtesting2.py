# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:15:55 2018

@author: rlk268
"""
#find good bounds / initial guesses for IDM. There are lists of difficult vehicles to calibrate in randomtesting "LCbad" is a list of veh IDs
from calibration import * 
plt.close('all')
pguess =  [40,1,1,3,10,25] #IDM guess 1 
#pguess =  [60,1,1,3,10,5] #IDM guess 2 
#pguess =  [80,1,15,1,1,5] #IDM guess 3 
#mybounds = [(10,100),(.1,5),(.1,25),(.1,20),(.1,20),(.1,75)]
mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]

sim = copy.deepcopy(meas)
#curplatoon = [[],1821] #vehicle 435 (10) very wonky for some reason #754 very wonky (13)

#testveh = 1 #21
#curplatoon = LClist[LCbad[testveh]]

#curplatoon = noLClist[1]

#curplatoon=LClist[15]
#print(curplatoon)
curplatoon = [[],603] #603 a good example of many lane changes #50 an example of calibration not working well #156
#curplatoon =LClist2[1]
print(curplatoon)
n = len(curplatoon[1:])

#print('testing vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s).')
#print(' originally adjoint found an obj. value of '+str(out[LCbad[testveh]][1])+' no relax adjoint found '+str(out2[LCbad[testveh]][1]))

leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

start = time.time()
bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(IDM_b3, IDMadjsys_b3, IDMadj_b3, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
end = time.time()
bfgstime = end-start
plotspeed(meas,sim,platooninfo,curplatoon[1])
plotdist(meas,sim,platooninfo,curplatoon[1])

#############
saver = rinfo.copy()
print(rinfo)

#start = time.time()
#bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
#end = time.time()
#bfgstime3 = end-start

pguess =  [40,1,1,3,10] #IDM 
mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,sim) 

start = time.time()
bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(IDM_b3, IDMadjsys_b3, IDMadj_b3, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,mybounds)
end = time.time()
bfgstime2 = end-start
plotspeed(meas,sim,platooninfo,curplatoon[1])
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