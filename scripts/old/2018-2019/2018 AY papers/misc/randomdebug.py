# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 16:03:45 2018

@author: rlk268
"""
from calibration import * 
##out,out2 = r_constant(rinfo[0],platooninfo[562][1:3],platooninfo[562][3],45)
#
##leadinfo,folinfo,rinfo = makeleadfolinfo_r5(curplatoon,platooninfo,sim)
#sim = copy.deepcopy(meas)
#platoons = [[],1013]
#
#pguess = [20,1,8,3.3,12]
#leadinfo,folinfo,rinfo = makeleadfolinfo(platoons, platooninfo, meas)
##platoonobjfn_objder(p,*(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo,True,6))
#
#blah = platoonobjfn_objder(test,*(IDM, IDMadjsys, IDMadj, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo))
#print(blah)

#%%


#sim = copy.deepcopy(meas)
#platoons = [[],1013]
#
#pguess = [0,60,5]
#leadinfo,folinfo,rinfo = makeleadfolinfo_r3(platoons, platooninfo, meas)
##platoonobjfn_objder(p,*(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo,True,6))
#
#blah = TTobjfn_obj(pguess,*(None, None, None, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo,True,3))

#%% make sure newell works with the calibrate_bfgs function
#from calibration import * 
#platoonlist = [[[],969]]
#
#plist = [[1.5,60,5],[2.5,100,60],[2,150,60]]
#mybounds = [(0,5),(5,200),(.1,75)]
#
#test = calibrate_bfgs(plist,mybounds,meas,platooninfo,platoonlist,makeleadfolinfo_r3,TTobjfn_obj,TTobjfn_fder,None,None,None,True,3,cutoff = 0,delay = True,dim=1)


#from calibration import * 
#platoonlist = [[[],603]]
#
#plist = [[1.5,60,5,5],[2.5,100,60,60],[2,150,60,60]]
#mybounds = [(0,5),(5,200),(.1,75),(.1,75)]
#
#test = calibrate_bfgs(plist,mybounds,meas,platooninfo,platoonlist,makeleadfolinfo_r3,TTobjfn_obj,TTobjfn_fder,None,None,None,True,4,True,cutoff = 0,delay = True,dim=1)

#%%
#from calibration import * 
#
#platoonlist = [[[],603]]
#plist = [[40,1,1,3,10],[60,1,1,3,10],[80,1,15,1,1]]
##plist = [[40,1,1,3,10,25],[60,1,1,3,10,5],[80,1,15,1,1,5]]
##plist = [[40,1,1,3,10,25,25],[60,1,1,3,10,5,5],[80,1,15,1,1,5,5]]
#mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
#
#test = calibrate_bfgs(plist,mybounds,meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn_objder,None,IDM_b3,IDMadjsys_b3,IDMadj_b3,False,5,cutoff = 0,delay = False,dim=2)

#%%
#need to be a little bit careful with plotting when the linesearch fails because of the time delay. 
#with open('LCtest5.pkl','rb') as f:
#    merge_nor, merge_r, merge_2r,mergeLC_r, mergeLC_2r = pickle.load(f)
#sim = copy.deepcopy(meas)
#obj = TTobjfn_obj(bfgs[0],*(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4,True,True))
#re_diff(sim,platooninfo,curplatoon,delay = bfgs[0][0])

#%%

#SEobj_pervehicle(meas,sim,platooninfo,curplatoon)
from calibration import * 

meas2,followerchain = makefollowerchain(956,data,15)
