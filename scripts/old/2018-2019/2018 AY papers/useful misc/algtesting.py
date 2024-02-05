# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 23:56:11 2019

@author: rlk268
"""

import nlopt 
import numpy as np 

from scipy.optimize import rosen, rosen_der

from calibration import * 
#import ipopt
##############MWE OF NLOPT 

def testnlopt():
    counter = 0 #note that this is how you are supposed to return func evals with nlopt.It doesn't keep track and only returns the answer in terms of parameters
    countergrad = 0
    def nltest(x,grad):
        nonlocal counter
        nonlocal countergrad
        if len(grad) > 0:
            countergrad += 1
            grad[:] = rosen_der(x)
            return grad
        counter += 1
        return rosen(x)
    
    opt = nlopt.opt(nlopt.GN_DIRECT_L,2)
    opt.set_min_objective(nltest)
    opt.set_lower_bounds([-1,-1])
    opt.set_upper_bounds([1,1])
    opt.set_maxeval(1000)
    #
#    test = opt.optimize([.5,-.5])
    test = opt.optimize([.5,-.5])
    
    return test, counter, countergrad
#execute above 
test, counter, countergrad = testnlopt()
#########################
#%%
#
#opt = nlopt.opt(nlopt.GN_DIRECT_L,5)
#
#
#
#sim = copy.deepcopy(meas)
#pguess = [10*3.3,.086/3.3, 1.545, 2, .175]

#how this is gonna work: have inside function does calibration stuff for a given platoon. Then we will have an outside function that will deal with the other stuff.

def calibrate_nlopt_test(alg, pguess, bounds, meas, sim, platooninfo, platoon, makeleadfolinfo, platoonobjfn, platoonobjfn_der, model, modeladjsys, modeladj, *args, evalper = 20 ):
    #this is going to take an specified platoon and calibrate it using one of the NLopt algorithms. 
    
    #refer to calibrate_bfgs2 for the most up to date documentation of what all the parameters are. 
    #note that nlopt doesn't support grad and obj being returned at the same time (this is not ideal for using automatic differentiation or the adjoint method because its slower) 
    #so this means that platoonobjfn_der needs to only return the grad. 
    
    N = len(pguess) #total number of parameters
    m = args[1] #number of parameters per vehicle 
    opt = nlopt.opt(alg,N)
    if alg == nlopt.G_MLSL_LDS or alg == nlopt.G_MLSL: 
        optlocal = nlopt.opt(nlopt.LD_TNEWTON_PRECOND_RESTART, N)
        opt.set_local_optimizer(optlocal)
    lb = []; ub = [];
    for i in bounds: 
        lb.append(i[0]); ub.append(i[1])
    
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    
    maxfun = max(200,evalper*N) #
    opt.set_maxeval(maxfun)
    
    leadinfo, folinfo, rinfo = makeleadfolinfo(platoon, platooninfo, sim) #note that this needs to be done with sim and not meas 
    
    count = 0
    countgrad = 0
    
    def nlopt_fun(p, grad):
        nonlocal count
        nonlocal countgrad
        if len(grad) > 0 :
            newgrad = platoonobjfn_der(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoon, leadinfo, folinfo, rinfo, *args)
            newgrad = newgrad
            grad[:] = newgrad
            countgrad += 1
            return grad
        obj = platoonobjfn(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoon, leadinfo, folinfo, rinfo, *args)
        count += 1
        return obj
            
    opt.set_min_objective(nlopt_fun)
    
    ans = opt.optimize(pguess) #returns answer
    
    return ans, count, countgrad



sim = copy.deepcopy(meas)
curplatoon = [[],581, 611]
n = len(curplatoon)-1
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5]
mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
pguess = np.tile(pguess, n)
mybounds = np.tile(mybounds,(n,1))

model = OVM
modeladjsys = OVMadjsys
modeladj = OVMadj


args = (True, 6)


#model = daganzo
#modeladjsys = daganzoadjsys
#modeladj = daganzoadj
#args = (True,4)
#pguess = [1,20,100,5] #daganzo guess 1 
#mybounds = [(.1,10),(0,100),(40,120),(.1,75)] #fairly conservative bounds




#alg list for nlopt
#GN_DIRECT_L #works
#GN_CRS2_LM #works

#G_MLSL_LDS #need also to set local optimizer LD_TNEWTON_PRECOND_RESTART #nlopt invalid argument 
#GD_STOGO #nlopt failure 
#GN_AGS #nlopt invalid argument 

#GN_ISRES #works
#GN_ESCH #works 

start = time.time()
ans, count, countgrad = calibrate_nlopt_test(nlopt.GN_DIRECT_L, pguess, mybounds, meas, sim, platooninfo, curplatoon, 
                                       makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, model, modeladjsys, modeladj, *args, evalper = 30)
end = time.time()
dt = end-start; 
#opt.set_lower_bounds([20,.001,.1,.1,0])
#opt.set_upper_bounds([120,.1,2,5,3])
#
#args = (False,5)
#

leadinfo,folinfo,rinfo = makeleadfolinfo_r3(curplatoon,platooninfo,meas)
#
obj = platoonobjfn_obj(ans,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args) #note the benchmark here is 360 for vehicle 1013 using TNC 
##which is done in 3.26 seconds or 1118 in 3.82 seconds for vehicle 581 with relaxation (note that this is on laptop)
#print('found objective of '+str(obj)+' in a time of '+str(dt))

outtest = calibrate_nlopt(nlopt.GN_DIRECT_L,pguess, mybounds, meas, platooninfo, [[[],581, 611]], makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           model, modeladjsys, modeladj, *args, evalper = 30, dim = 1)