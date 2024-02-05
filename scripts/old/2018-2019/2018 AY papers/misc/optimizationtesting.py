# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:16:11 2019

@author: rlk268
"""
sim = copy.deepcopy(meas)

pguess = [1.5,60,5,5] #newell
mybounds = [(1e-8,5),(0,200),(.1,75),(.1,75)]

pguess = [1,20,100,5] #daganzo guess 1 
mybounds = [(.1,10),(0,100),(40,120),(.1,75)] #fairly conservative bounds

curplatoon = [[],1013]

leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim) 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

#args = (None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4,True,True)
#fnc = TTobjfn_obj
#fnc_der = TTobjfn_fder

args = (daganzo,daganzoadjsys,daganzoadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4)
fnc = platoonobjfn_obj
fnc_der = platoonobjfn_der
fnc_objder = platoonobjfn_objder

#spsa = SPSA(fnc,p,bounds,args,maxit= 10000, c1=5e-4, c2=1)
#sd = SD(fnc,fnc_der, p,bounds,args,maxit= 10000, c1=5e-4, c2=1)
#bfgs = sc.fmin_l_bfgs_b(TTobjfn_obj,p,TTobjfn_fder,(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4,True,True),0,mybounds)
bfgs = sc.fmin_l_bfgs_b(fnc_objder,p,None,args,0,bounds)

gd = pgrad_descent(fnc,fnc_objder,None,p,bounds,backtrack,args,srch_type=1,proj_type=1)

gd1 = pgrad_descent(fnc,fnc_objder,None,p,bounds,weakwolfe,args,srch_type=1,proj_type=1)
