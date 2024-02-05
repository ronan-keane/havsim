# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:57:02 2019

@author: rlk268
"""

from calibration import * 
testmeas, followerchain = makefollowerchain(996,data,n=5)
testsim = copy.deepcopy(testmeas)
#daganzo parameters
pguess = [1,20,30,5] #daganzo guess 1 
mybounds = [(1,10),(0,100),(40,120),(.1,75)] #fairly conservative bounds

#pguess2 =  [1.5,60,5] #newell guess
#mybounds2 = [(1,5),(0,100),(.1,75)]

#pguess = [.1,20,30,5] #daganzo guess 1 
#mybounds = [(.1,.1),(0,100),(40,120),(.1,75)] #fairly conservative bounds
#
#pguess2 =  [.1,60,5] #newell guess
#mybounds2 = [(.1,.1),(0,100),(.1,75)]

#pguess =  [40,1,1,3,10,25] #IDM guess 1 
#mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
#
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ] #original guess
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds 

curplatoon = [[],1003]
leadinfo,folinfo,rinfo = makeleadfolinfo_r3(curplatoon,followerchain,testsim)
print(rinfo)

#test = TTobjfn_obj(bfgs[0][:3],*(None, None, None, testmeas,testsim,followerchain, curplatoon, leadinfo, folinfo,rinfo,True,3,False,True))
#re_diff(testsim,followerchain,curplatoon)
#plotvhd(testmeas,testsim,followerchain,curplatoon[1])

#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(daganzo, daganzoadjsys, daganzoadj, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,4),0,mybounds)

#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(IDM_b3, IDMadjsys_b3, IDMadj_b3, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)

#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(OVM, OVMadjsys, OVMadj, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)

bfgs = sc.fmin_l_bfgs_b(TTobjfn_obj,pguess2,TTobjfn_fder,(None, None, None, testmeas,testsim,followerchain, curplatoon, leadinfo, folinfo,rinfo,True,3,False,True),0,mybounds2)

re_diff(testsim,followerchain,curplatoon)
T_nm1 = platooninfo[curplatoon[1]][2]
plt.close('all')

#plotvhd(testmeas,testsim,followerchain,curplatoon[1], end = T_nm1-1)
#plt.figure()
#plotspeed(testmeas,testsim,followerchain,1003)

plotvhd(testmeas,testsim,followerchain,curplatoon[1],delay=bfgs[0][0])
plt.figure()
plotspeed(testmeas,testsim,followerchain,1003,delay=bfgs[0][0])

curplatoon=[[],1013]
leadinfo,folinfo,rinfo = makeleadfolinfo_r3(curplatoon,platooninfo,meas)



#bfgs2 = sc.fmin_l_bfgs_b(TTobjfn_obj,pguess2,TTobjfn_fder,(None, None, None, testmeas,testsim,followerchain, curplatoon, leadinfo, folinfo,rinfo,True,3,False,True),0,mybounds2)

#bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(daganzo, daganzoadjsys, daganzoadj, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,4),0,mybounds)

#bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(IDM_b3, IDMadjsys_b3, IDMadj_b3, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)

bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(OVM, OVMadjsys, OVMadj, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
#re_diff(testsim,followerchain,curplatoon)
#plotvhd(testmeas,testsim,followerchain,curplatoon[1])
#bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(daganzo, daganzoadjsys, daganzoadj, testmeas,  testsim, followerchain, curplatoon, leadinfo, folinfo,rinfo,True,4),0,mybounds)
#T_nm1 = platooninfo[curplatoon[1]][2]
#re_diff(testsim,followerchain,curplatoon)
#plotvhd(testmeas,testsim,followerchain,curplatoon[1], end = T_nm1-1)
#plt.figure()
#plotspeed(testmeas,testsim,followerchain,1014)

re_diff(testsim,followerchain,curplatoon,delay=bfgs2[0][0])
plotvhd(testmeas,testsim,followerchain,curplatoon[1],delay=bfgs2[0][0])
plt.figure()
plotspeed(testmeas,testsim,followerchain,1014,delay=bfgs2[0][0])