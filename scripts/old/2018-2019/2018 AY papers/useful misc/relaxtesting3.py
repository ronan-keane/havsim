# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:15:55 2018

@author: rlk268
"""
#find good bounds / initial guesses for IDM. There are lists of difficult vehicles to calibrate in randomtesting "LCbad" is a list of veh IDs
from calibration import * 
plt.close('all')

pguess =  [1.5,60,5,5] #guess 1 
#pguess =  [2.5,100,60] #guess 2
#pguess =  [2,150,60] #guess 3 
mybounds = [(0,5),(0,200),(.1,75),(.1,75)]

sim = copy.deepcopy(meas)
#curplatoon = [[],1821] #vehicle 435 (10) very wonky for some reason #754 very wonky (13)
#testveh = 19 #21
#curplatoon = LClist[LCbad[testveh]]
#curplatoon=LClist[15]
#print(curplatoon)
curplatoon = [[],158] #603 a good example of many lane changes #50 an example of calibration not working well #156
curplatoon = [[],1014]
#curplatoon =LClist2[1]
print(curplatoon)
n = len(curplatoon[1:])

#print('testing vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s).')
#print(' originally adjoint found an obj. value of '+str(out[LCbad[testveh]][1])+' no relax adjoint found '+str(out2[LCbad[testveh]][1]))

leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

start = time.time()
bfgs = sc.fmin_l_bfgs_b(TTobjfn_obj,p,TTobjfn_fder,(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4,True,True),0,mybounds)
end = time.time()
bfgstime = end-start
re_diff(sim,platooninfo,curplatoon,delay = bfgs[0][0])
plt.figure()
plotspeed(meas,sim,platooninfo,curplatoon[1],delay = bfgs[0][0])
plt.figure()
plotdist(meas,sim,platooninfo,curplatoon[1],delay = bfgs[0][0])

#############
#saver = rinfo.copy()
#print(rinfo)
#sim = copy.deepcopy(meas)
##start = time.time()
##bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
##end = time.time()
##bfgstime3 = end-start
#
#pguess =  [1.5,60]
#mybounds = [(0,5),(0,200)]
#p = np.tile(pguess, n)
#bounds = np.tile(mybounds,(n,1))
#
#leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,sim) 
#
#start = time.time()
#bfgs2 = sc.fmin_l_bfgs_b(TTobjfn_obj,p,TTobjfn_fder,(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,2),0,mybounds)
#end = time.time()
#bfgstime2 = end-start
#re_diff(sim,platooninfo,curplatoon,delay = bfgs2[0][0])
#plt.figure()
#plotspeed(meas,sim,platooninfo,curplatoon[1],delay = bfgs2[0][0])
#plt.figure()
#plotdist(meas,sim,platooninfo,curplatoon[1],delay = bfgs2[0][0])
#
#print('Newell with new guess found '+str(bfgs[1]))
##print(' finite with new guess found '+str(bfgs3[1]))
#print('Newell with new guess no relax found '+str(bfgs2[1]))


#%%


#%% compare to global optimum for reference
#
#sim = copy.deepcopy(meas)
#mybounds = [(0,5),(5,200),(.1,75)]
#leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 
#start = time.time()
#GA = sc.differential_evolution(TTobjfn_obj,mybounds,(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,3))
#end = time.time()
#GAtime = end-start
#
#print('Newell with GA found '+str(GA['fun']))