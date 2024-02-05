# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 08:55:13 2019

@author: rlk268

debugging script seeing if the merging strategy was actually helping the fit or not. 
"""

#sim = copy.deepcopy(meas)
#mergelist = []
#merge_from_lane = 7 
#merge_lane = 6
#for i in meas.keys():
#    curveh = i
#    t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
#    lanelist = np.unique(sim[curveh][:t_n-t_nstar,7])
#    if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][t_n-t_nstar,7]==merge_lane:
#        mergelist.append([[],i])
        
        
#%%
from calibration import * 
plt.close('all')
sim = copy.deepcopy(meas)
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5,5  ] #original guess


mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75), (.1, 75)] #less conservative bounds 

curplatoon = mergelist[21]

print(curplatoon)
n = len(curplatoon[1:])

#print('testing vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s).')
#print(' originally adjoint found an obj. value of '+str(out[LCbad[testveh]][1])+' no relax adjoint found '+str(out2[LCbad[testveh]][1]))

leadinfo, folinfo, rinfo = makeleadfolinfo_r5(curplatoon, platooninfo,sim) 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

print(rinfo)

start = time.time()
bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder2,p,None,(OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,7),0,mybounds)
end = time.time()
bfgstime = end-start
plotspeed(meas,sim,platooninfo,curplatoon[1])

sim[curplatoon[1]] = meas[curplatoon[1]].copy()
pguess = [10*3.3, .086/3.3, 1.545, 2, .175, 5, 5 ]
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75), (.1,75)] #less conservative bounds 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

leadinfo, folinfo, rinfo = makeleadfolinfo_r4(curplatoon, platooninfo,sim) 
print(rinfo)
start = time.time()
bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder2,p,None,(OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,7),0,mybounds)
end = time.time()
bfgstime2 = end-start
plotspeed(meas,sim,platooninfo,curplatoon[1])

#############
#saver = rinfo.copy()
#print(rinfo)

#start = time.time()
#bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
#end = time.time()
#bfgstime3 = end-start
sim[curplatoon[1]] = meas[curplatoon[1]].copy()
pguess = [10*3.3, .086/3.3, 1.545, 2, .175, 5, 5 ]
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75), (.1,75)] #less conservative bounds 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 

start = time.time()
bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_objder2,p,None,(OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,7),0,mybounds)
end = time.time()
bfgstime2 = end-start
plotspeed(meas,sim,platooninfo,curplatoon[1])

print('adjoint 2r with merger 2 found '+str(bfgs[1]))
print(' adjoint 2r with merger 1 found '+str(bfgs2[1]))  
print('adjoint 2r with no merger found '+str(bfgs3[1]))

  
        

