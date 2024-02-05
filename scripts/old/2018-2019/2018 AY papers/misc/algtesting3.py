# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:57:10 2019

@author: rlk268
"""
from calibration import * 

meas, platooninfo, platoonlist = makeplatoonlist(data,n=1,lane=2,vehs=[582,1146])
sim = copy.deepcopy(meas)

#bounds, guesses 
pguess = [10*3.3,.086/3.3, 1.545, 2, .175,5]
plist = [[10*3.3,.086/3.3, 1.545, 2, .175,5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175,60],[10*3.3,.086/3.3/2, .5, .5, .175,60]]
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3),(.1,75)] #less conservative bounds 

#this stuff needs to be defined 
makeleadfolfun = makeleadfolinfo_r3
model = OVM
modeladjsys = OVMadjsys
modeladj = OVMadj 
platoonobjfn = platoonobjfn_obj
args = (True,6)

platoonlisttest = platoonlist[0:5]

#optplot(out3test,meas,None,platooninfo,platoonlisttest,model,modeladjsys,modeladj,makeleadfolfun,platoonobjfn,args,lane = 2)

#%%

out3test2 = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlisttest, makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
                       cutoff=4,cutoff2=5.5, order = 1, dim = 2, budget = 1)

#%%
optplot(out3test2,meas,None,platooninfo,platoonlisttest,model,modeladjsys,modeladj,makeleadfolfun,platoonobjfn,args,lane = 2)
