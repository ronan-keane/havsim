# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:09:11 2019
#the purpose of this script is to test different optimization algorithms

be careful when analyzing results of optimization, make sure the RMSE observed is consistent with what was recorded, otherwise you might 
have a bug which is making the simulation slightly different than the optimized simulation. Causes of this in the past have been: 
    -forgetting to initialize sim as equal to meas, 
    -problem with how sim is being updated
    -inconsistent makeleadfolinfo functions used 

@author: rlk268
"""
from calibration import * 

#make test platoons to calibrate
meas, platooninfo, platoonlist = makeplatoonlist(data,n=5,lane=2,vehs=[582,1146])
sim = copy.deepcopy(meas)

#bounds, guesses 
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5]
plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds 

#this stuff needs to be defined 
makeleadfolfun = makeleadfolinfo_r3
model = OVM
modeladjsys = OVMadjsys
modeladj = OVMadj 
platoonobjfn = platoonobjfn_obj
args = (True,6)


platoonlisttest = platoonlist[0:3]

optplot(out2,meas,None,platooninfo,platoonlisttest,model,modeladjsys,modeladj,makeleadfolfun,platoonobjfn,args,lane = 2)

#%%%

out2 = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlisttest, makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
                       cutoff=0,cutoff2=0, order = 1, dim = 2, budget = 1)

out3 = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlisttest, makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
                       cutoff=4,cutoff2=5.5, order = 1, dim = 2, budget = 3)

out4 = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlisttest, makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
                       cutoff=4,cutoff2=5.5, order = 1, dim = 2, budget = 3, reguess = False)


#out2test = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlisttest, makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
#                       cutoff=4,cutoff2=5.5, order = 0, dim = 2, budget = 3)
#
#out3test = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlisttest, makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
#                       cutoff=4,cutoff2=5.5, order = 1, dim = 2, budget = 3)


#%% #visualize and test results 
#old 
#sim = obj_helper(out2[0],model,modeladjsys,modeladj,meas,sim,platooninfo,platoonlisttest,makeleadfolinfo_r3,args)
#platoonplot(meas,sim,platooninfo,platoonlisttest,lane =2)
# old 

sim = obj_helper(out2[0],model,modeladjsys,modeladj,meas,sim,platooninfo,platoonlisttest,makeleadfolfun,platoonobjfn,args) #load in what the sim is supposed to be

obj = SEobj_pervehicle(meas,sim,platooninfo,platoonlisttest[2]) #convert sim to RMSE and check that this is consistent with output. 
convert_to_rmse(sum(obj),platooninfo,platoonlisttest[2])

#%%%

out5 = calibrate_tnc2(plist,mybounds,sim,platooninfo,[platoonlisttest[1]], makeleadfolfun,platoonobjfn_objder,None,model,modeladjsys,modeladj,*args,
                       cutoff=4,cutoff2=5.5, order = 1, dim = 2, budget = 3)