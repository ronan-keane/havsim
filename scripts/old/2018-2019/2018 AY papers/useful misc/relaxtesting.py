# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:15:55 2018

@author: rlk268

in all of the relaxtesting scripts, we tested different initial guesses/bounds for the calibration problems
we were solving for the OVM, IDM, Daganzo, and Newell models, including adding the relaxation parameters.
So these scripts may be useful since we found some initial guesses that empirically worked well for all the vehicles. 
"""
from calibration import * 
#pguess = [16.8*3.3,.086/3.3, 1.545, 2, .175 ]
plt.close('all')
#pguess = [20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ] #this seems like a very good second guess
pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ] #original guess
#pguess = [10*3.3,.086/3.3/2, .5, .5, .175, 60 ] #seems to work a good bit of the time

mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds 

pguess = [10*3.3,.086/3.3, 1.545, 2, .175 ] #original guess
#pguess = [10*3.3,.086/3.3/2, .5, .5, .175, 60 ] #seems to work a good bit of the time

mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] #less conservative bounds 

#pguess = bfgs[0]*1.5
pguess = np.asarray(pguess)

sim = copy.deepcopy(meas)
#curplatoon = [[],1821] #vehicle 435 (10) very wonky for some reason #754 very wonky (13)
testveh = 3 #21
#curplatoon = LClist[LCbad[testveh]]
#curplatoon=LClist[15]
#print(curplatoon)
curplatoon = [[],1013] #603 a good example of many lane changes #50 an example of calibration not working well #156
#curplatoon =LClist2[1]
print(curplatoon)
n = len(curplatoon[1:])

#print('testing vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s).')
#print(' originally adjoint found an obj. value of '+str(out[LCbad[testveh]][1])+' no relax adjoint found '+str(out2[LCbad[testveh]][1]))

leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,sim,False) 
p = np.tile(pguess, n)
bounds = np.tile(mybounds,(n,1))

#rescaling seems to actually make things worse
#rescale = np.asarray([1,1000,100,100,100,1])
#pr = np.multiply(p,rescale)
#boundsr = bounds.copy()
#for i in range(len(boundsr)):
#    boundsr[i,:] = boundsr[i,:]*rescale[i]


print(rinfo)

start = time.time()
bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False, 5),0,mybounds)
plotvhd(meas, sim,platooninfo,curplatoon[1])
#bfgsr = sc.fmin_l_bfgs_b(rescaledobjfn_objder,pr,None,(rescale,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,boundsr)
#gd = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder, None,p,bounds,backtrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6), eps=1e-6)
end = time.time()
bfgstime = end-start
start = time.time()
#spsa = SPSA(platoonobjfn_obj,None, None, p,bounds,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),maxit= 3000, c1=5e-4, c2=1)

#spsa2 = SPSA(platoonobjfn_obj,None, None, p,bounds,None, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),maxit= 5000, c1=5e-4, c2=1)

#sd = SD(platoonobjfn_obj,platoonobjfn_der, p,bounds,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),maxit= 10000, c1=1e-4, c2=1)
end = time.time()
spsatime = end-start

#gd = pgrad_descent(platoonobjfn_obj,platoonobjfn_fder,None,p,bounds,backtrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-10,srch_type=0,proj_type=0, c1=1e-2,gamma=.5, der_only = True, maxit = 200)
start = time.time() #this seems to work best 
#gd = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-6,srch_type=2,proj_type=0, maxit = 1000, c1=1e-4,gamma=.3, der_only = False)
end = time.time()
gdtime = end-start
#gd2 = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,weakwolfe,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-7,srch_type=1,proj_type=0, maxit = 1000, c1=1e-4, c2=.5) 
#gd3 = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,weakwolfe2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-8,srch_type=1,proj_type=0, maxit = 1000, c1=1e-4, c2=.5, alo=.1, ahi=.9)
#gd4 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-6,srch_type=1,proj_type=0, maxit = 1000, c1=1e-4, gamma=.3, alo=.1, ahi=.6)

#.3 .1 .6 seems to work the best (srch type, proj type = 2 0 ), eps = 1e-6, c1 = 1e-4,  in terms of the function value you can reach 
#.5, .1, .9 eps = 1e-6, c1 = 1e-4 seems to work very well with srch type, proj type = (1, 0)
#.5, .1, .9, eps = 5e-7, c1=1e-4, srchtype, projtype  = 1, 0 with nmbacktrack t = 3 overall seems to give the best results. 
start = time.time() #gives slightly better answer than backtracking but since you have more gradient evaluations it ends up being slower
#gd2 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-7,t=2, srch_type=1,proj_type=0, maxit = 1000, eps2=1e-6, c1 = 1e-4, c2 = .5, der_only = False)
end = time.time()
gdtime2 = end-start

#gd4 = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,nmbacktrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),t=3, eps=1e-6,srch_type=1,proj_type=0, maxit = 1000, c1=1e-4, gamma=.5, alo=.1, ahi=.9)



###############this one is good##################
#gd2 = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,weakwolfe2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=5e-7,t=0, srch_type=1,proj_type=0, maxit = 1000, c1 = 1e-4, c2 = .9, der_only = False)

#gd3 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=5e-7,t=1, srch_type=1,proj_type=0, maxit = 1000, c1 = 1e-4, c2 = .9, der_only = False)

#gd4 = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,nmweakwolfe,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=5e-7, t=2, srch_type=1,proj_type=0, maxit = 1000, c1 = 1e-4, c2 = .9, der_only = False)
 
#this is probably the best sqp 
start = time.time()
#sqp = SQP2(platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der,p,bounds,nmbacktrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,5), maxit = 200, t=2, eps=5e-7)
end = time.time()
sqptime = end-start

#sqp2 = SQP(platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der,p,bounds,backtrack2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True, 6),maxit = 200, t=2, eps=5e-7)

#sqp2 = SQP(platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der,p,bounds,backtrack2,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True, 6),maxit = 200, t=3, eps=5e-7)


#this one is still the best. 
start = time.time()
#gd5 = pgrad_descent2(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,nmbacktrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,5),t=3, eps=5e-7,srch_type=1,proj_type=0, maxit = 1000, c1=1e-4)
end = time.time()
gd5time = end-start
##################################################

plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
vehlist=  [[[],1013]]
kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3}

#out2 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
#                               nmbacktrack, kwarg, False, 5, cutoff = 0)


start = time.time() #gives slightly better answer than backtracking but since you have more gradient evaluations it ends up being slower
#gd3 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,weakwolfe,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-10,srch_type=2,proj_type=0,maxit = 1000, c1 = 1e-4, c2=.5, der_only = False)
end = time.time()
gdtime3 = end-start


#gd2 = pgrad_descent(platoonobjfn_obj,platoonobjfn_fder,None,p,bounds,weakwolfe,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-10,srch_type=0,proj_type=0, maxit = 200, der_only = True)

#gd2 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-10,srch_type=1,proj_type=1, c1=1e-2, gamma=.5)
#gd3 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-10,srch_type=1,proj_type=1)
#gd4 = pgrad_descent(platoonobjfn_obj,platoonobjfn_objder,None,p,bounds,backtrack,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),eps=1e-10,srch_type=1,proj_type=0)
#plotspeed(meas,sim,platooninfo,curplatoon[1])
#plotdist(meas,sim,platooninfo,curplatoon[1])
#%%
##############
#saver = rinfo.copy()
#print(rinfo)
#
##start = time.time()
##bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_obj,p,platoonobjfn_fder,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,6),0,mybounds)
##end = time.time()
##bfgstime3 = end-start
#
#pguess = [10*3.3, .086/3.3, 1.545, 2, .175 ]
#mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] #less conservative bounds 
#p = np.tile(pguess, n)
#bounds = np.tile(mybounds,(n,1))
#
#leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,sim) 
#
#start = time.time()
##bfgs2 = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,mybounds)
#
#end = time.time()
#bfgstime2 = end-start
#
#print('adjoint with new guess found '+str(bfgs[1]))
##print(' finite with new guess found '+str(bfgs3[1]))
#print(' adjoint with new guess no relax found '+str(bfgs2[1]))


#%%
#plotspeed(meas,sim,platooninfo,curplatoon[1]) #first is no relax 
#plotdist(meas,sim,platooninfo,curplatoon[1])

#im_ani = animatevhd(meas,sim,platooninfo,1013)
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

#plotvhd(meas,sim,platooninfo,1013)

#im_ani.save('eg.mp4',writer=writer)

##next do initial guess
#leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,sim) 
#obj = platoonobjfn_obj(pguess,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)
#plotspeed(meas,sim,platooninfo,curplatoon[1]) #initial guess
#plotdist(meas,sim,platooninfo,curplatoon[1])

##plot relax
#sim = copy.deepcopy(meas)
#leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon,platooninfo,sim)
#obj = platoonobjfn_obj(bfgs[0],OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, saver,True,6)
##obj = platoonobjfn_obj(out2[LCbad[testveh]][0],OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)
#plotspeed(meas,sim,platooninfo,curplatoon[1])


#optimalvelocityplot(bfgs[0])


#%%
from calibration import * 
#plist,bounds, meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn, platoonobjfn_der, model, 
#modeladjsys, modeladj, *args, cutoff=7.5, delay = False, dim = 2, objder = True
def adjointtest8(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3}
    
    custom1 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = 0)
    
    custom2 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = 7.5)
    
    custom3 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = float('inf'))
    
    return custom1, custom2, custom3, 
    
#test = adjointtest8(platooninfo,meas)