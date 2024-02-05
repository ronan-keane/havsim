# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:11:54 2018

@author: rlk268
"""
import numpy as np
import time

#pguess = [16.8*3.3,.086/3.3, 1.545, 2, .175 ] #this is what we are using for the initial guess for OVM; it comes from the bando et al paper 'phenomological study ...'
##mybounds = [(30,100),(.01,.05),(.3,2),(1,3),(0,.5)] #conservative bounds 
#mybounds = [(30,200),(.001,.1),(.1,10),(.1,5),(0,2)] #less conservative bounds 
##sim = copy.deepcopy(meas) #to reset simulation to the measurements
##curplatoon = [[],998] #do the calibration with this platoon; save as bfgs
#curplatoon = [[],1433] #do the claibraiton with this, save as bfgs1
##curplatoon = [[],1057,1216,1347,1349,1433]
##put in the calibration for 998 into simulation 
##curplatoon = [[],1013] #now do the calibration for vehicle 1013
#
##curplatoon = platoonlist[294]
#n = len(curplatoon[1:])
#leadinfo, folinfo = makeleadfolinfo(curplatoon, platooninfo,meas) 
#p = np.tile(pguess, n)
#bounds = np.tile(mybounds,(n,1))
###p = finitebfgs['x']
#
#obj = platoonobjfn_noder(p,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo)
#
##############test calibration#####################
#
#start = time.time()
#bfgs = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo),'L-BFGS-B',platoonobjfn_der,None,None, bounds)
#end = time.time()
#bfgstime = end-start
#
##sim = copy.deepcopy(meas)
##
#start = time.time()
#bfgs1 = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo),'L-BFGS-B',None,None,None, bounds)
#end = time.time()
#bfgstime = end-start
#
#start = time.time()
#NM = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo),'Nelder-Mead')
#end = time.time()
#NMtime = end-start

#start = time.time()
#bfgs2 = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo),'L-BFGS-B',platoonobjfn_der,None,None, bounds)
#end = time.time()
#bfgstime = end-start

#start = time.time()
#bfgs3 = sc.minimize(platoonobjfn_noder,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo),'L-BFGS-B',platoonobjfn_der,None,None, bounds)
#end = time.time()
#bfgstime = end-start




#%%
#ans = [bfgs, finitebfgs, NM]
#speedRMSE = []
#distRMSE = []
#myveh = curplatoon[-1]
#sim = platoon_sim(p,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo)
#plotspeed(meas[myveh],sim[myveh],platooninfo[myveh])
#speedRMSE.append(rmse_speed(meas[myveh],sim[myveh],platooninfo[myveh]))
#distRMSE.append(rmse_dist(meas[myveh],sim[myveh],platooninfo[myveh]))
#for i in ans: 
#    myp = i['x']
#    sim = platoon_sim(myp,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo)
#    plotspeed(meas[myveh],sim[myveh],platooninfo[myveh])
#    speedRMSE.append(rmse_speed(meas[myveh],sim[myveh],platooninfo[myveh]))
#    distRMSE.append(rmse_dist(meas[myveh],sim[myveh],platooninfo[myveh]))
    
#%%
#investigate why some calibration attempts fail

#curplatoon = platoonlist[failed2[1]]
#curplatoon = [[],1014]
#p = [10*3.3,.086/3.3, 1.545, 2, .175 ]
#
#leadinfo, folinfo,rinfo = makeleadfolinfo(curplatoon, platooninfo,meas)
#
#start = time.time()
#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,None)
#end = time.time()
#bfgstime = end-start
#
#start = time.time()
#bfgs2 = sc.minimize(platoonobjfn_obj,p,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),'L-BFGS-B',platoonobjfn_fder,None,None,mybounds)
#end = time.time()
#bfgs2time = end-start

#%% test some stuff with the initial tests we ran 

#rmse3 = []
#rmse4 = []
#rmse5 = []
#rmse6 = []
#rmse7 = []
#rmse8 = []
#rmse9 = []
#rmse10 = []
#
#for i in range(len(platoonlist)): 
#    curplatoon = platoonlist[i]
#    currmse = convert_to_rmse(out3[i][1],platooninfo,curplatoon)
#    rmse3.append(currmse)
#    
#    currmse = convert_to_rmse(out4[i]['fun'],platooninfo,curplatoon)
#    rmse4.append(currmse)
#    
#    currmse = convert_to_rmse(out5[i][1],platooninfo,curplatoon)
#    rmse5.append(currmse)
#    
#    currmse = convert_to_rmse(out6[i]['fun'],platooninfo,curplatoon)
#    rmse6.append(currmse)
#    
#    currmse = convert_to_rmse(out7[i][1],platooninfo,curplatoon)
#    rmse7.append(currmse)
#    
#    currmse = convert_to_rmse(out8[i]['fun'],platooninfo,curplatoon)
#    rmse8.append(currmse)
#    
#    currmse = convert_to_rmse(out9[i][1],platooninfo,curplatoon)
#    rmse9.append(currmse)
#    
#    currmse = convert_to_rmse(out10[i]['fun'],platooninfo,curplatoon)
#    rmse10.append(currmse)
    
#rmse3 = np.asarray(rmse3) #convert to numpy array and you can do boolean masking useful for looking at stuff
#rmse3[rmse3<15]
    
    
#%% #check the initial relax content results  
##get vehicles which are hard to calibrate so we can test on those; require results from initial relaxation test
#noLCbad = []
#LCbad = []
#for i in range(len(rmse3)):
#    if rmse3[i]>15: 
#        noLCbad.append(i)
#        
#for i in range(len(rmse)):
#    if rmse[i]>15:
#        LCbad.append(i)
#        
#noLClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])==1:
#        noLClist.append([[],i])
#            
#LClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])>1:
#        LClist.append([[],i])
#        
##rmse = np.asarray(rmse)
##rmse2 = np.asarray(rmse2)
##rmse3 = np.asarray(rmse3)
##
##plt.hist(rmse[rmse<15])
##plt.figure()
##plt.hist(rmse2[rmse2<15])
##plt.figure()
##plt.hist(rmse3[rmse3<15])
#        
        
        
#%% #investigate why some vehicles have high rmse

#%%

#noLClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])==1:
#        noLClist.append([[],i])
#        
#plist = []
#for i in noLC[0]:
#    plist.append(i[0])
#    
#real = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
            
#%%
#       #test 2 parameter relax  and merger stuff
#from calibration import * 
#sim = copy.deepcopy(meas)
##pguess = [16.8*3.3,.086/3.3, 1.545, 2, .175 ]
#pguess = [10*3.3, .086/3.3, 1.545, 2, .175 ,5.01,5.01]
##pguess = [10*3.3,.086/3.3, 1.545, 2, .175, .11 ]
#mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3),(.1,75),(.1,75)] #less conservative bounds #can mess with the bounds to make them loser (may get weird stuff but in general better) or tighter (can cause higher values)
##mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds 
#
##curplatoon = platoonlist[93]
##curplatoon = [[], 995,998,1013,1023,1030]  #[[],995,998,1013,1023,1030] this is a good test platoon 
##curplatoon = [[],995,998,1013,1023] #995 good for testing lane changing #1003 1014 was original pair we used for testing where 1014 was the follower
#curplatoon = LClist2[1]
##curplatoon = [[],1320]
#n = len(curplatoon[1:])
#
#leadinfo, folinfo, rinfo = makeleadfolinfo_r3(curplatoon, platooninfo,meas) 
##leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,meas)
#p = np.tile(pguess, n)
#bounds = np.tile(mybounds,(n,1))
##p = finitebfgs['x']
###########################################################################################
#start = time.time()
#obj = platoonobjfn_obj2(p,OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo,True,7)
#end = time.time()
#objtime = end-start #note that the obj is supposed to be around 500k for the initial platoon with 5 vehicles, initial guess for ovm 
#
#start = time.time()
#adjder = platoonobjfn_objder2(p,OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,7)
#adjder = adjder[1]
#end = time.time()
#adjdertime = end-start
#
#start = time.time()
#finder = platoonobjfn_fder2(p,OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,7)
#end = time.time()
#findertime = end-start
#
#acc = np.linalg.norm(adjder-finder)/np.linalg.norm(finder)
#acc2 = np.divide(adjder-finder,finder)
#print('accuracy in norm is '+str(acc))
#print(acc2)
#
## 
#################test objective and gradient evaluation##################
#start = time.time()
#bfgs3 = sc.fmin_l_bfgs_b(platoonobjfn_objder2,p,None,(OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,7),0,mybounds)
#end = time.time()
#bfgstime = end-start
#
#plotspeed(meas,sim,platooninfo,curplatoon[1])

#%%

#LClist2 = []
#for i in meas.keys():
#    if len(platooninfo[i][4])>1:
#        unused,unused,rinfo = makeleadfolinfo_r3([[],i],platooninfo,sim)
#        curpos = False
#        curneg = False
#        for j in rinfo[0]: 
#            if j[1]>0:
#                curpos = True
#            else:
#                curneg = True
#            if curpos and curneg: 
#                LClist2.append([[],i])
#                break
                
            

#%%
##list of merging vehicles
#from calibration import * 
#mergeLClist = []
#for i in meas.keys():
#    unused, unused, rinfo = makeleadfolinfo_r3([[],i],platooninfo,meas)
#    unused,unused,rinfo2 = makeleadfolinfo_r6([[],i],platooninfo,meas)
#    if len(rinfo[0])>0:
#        if len(rinfo2[0])==0:
#            mergeLClist.append([[],i])
#        elif rinfo[0][0] != rinfo2[0][0]:
#            mergeLClist.append([[],i])
            
#%%%
            #debug the SPSA and other new optimization algorithms
from calibration import * 
from scipy.optimize import rosen
from scipy.optimize import rosen_der
def myf(x):
    return (x[0]-1)**2 + 3*(x[1]-2)**2
    
    
def mydf(x):
    return np.asarray([2*(x[0]-1), 6*(x[1]-2)])

#p =[3,4]
#hess = approx_hess([3,4],mydf)
#test = []
#for i in range(500):
#    spsader = SPSA_grad(p,myf)
#    test.append(spsader)
    
p = [5.5,4.5]
bounds = [(1,6),(-2,6)]
#ans = SPSA(myf,p,bounds,(),maxit = 10000)
#ans2 = pgrad_descent(myf,mydf,None, p,bounds,weakwolfe, (),maxit = 10000, der_only = True, srch_type = 0, proj_type = 1)

#ans = SPSA(rosen,p,bounds,(),maxit = 10000, c1=1e-2,c2 = 1)
#ans2 = SD(rosen,rosen_der,p,bounds,(),maxit = 10000)
#ans2 = pgrad_descent(rosen, rosen_der, None, p, bounds, backtrack,  (), der_only = True)
#ans3 = pgrad_descent(rosen, rosen_der, None, p, bounds, backtrack,  (), der_only = True, srch_type = 1)


#
#ans42 = pgrad_descent(rosen, rosen_der, None, p, bounds, backtrack2,  (), t = 3, der_only = True, srch_type = 1, proj_type = 0)
#
#ans4 = pgrad_descent2(rosen, rosen_der, None, p, bounds, backtrack2,  (), der_only = True, srch_type = 2, proj_type = 0)
#
#ans43 = pgrad_descent2(rosen, rosen_der, None, p, bounds, backtrack,  (), der_only = True, srch_type = 1, proj_type = 0)
#
#ans44 = pgrad_descent2(rosen, rosen_der, None, p, bounds, nmbacktrack,  (), t = 3, der_only = True, srch_type = 1, proj_type = 0)



#ans5 = pgrad_descent(rosen, rosen_der, None, p, bounds, backtrack,  (), der_only = True, proj_type = 1)

ans41 = pgrad_descent2(rosen, rosen_der, None, p, bounds, weakwolfe,  (), BBlow = 1e-5, c2 = .9, der_only = True, srch_type = 1, proj_type = 0)

ans42 = pgrad_descent(rosen, rosen_der, None, p, bounds, weakwolfe2,  (), t=2, BBlow = 1e-5, eps2 = 1e-10, c2 = .9, der_only = True, srch_type = 1, proj_type = 0)

ans43 = pgrad_descent2(rosen, rosen_der, None, p, bounds, nmweakwolfe,  (), t = 1, BBlow = 1e-5, eps2 = 1e-10, c2 = .9, der_only = True, srch_type = 1, proj_type = 0)

ans51 = SQP2(rosen, rosen_der, rosen_der, p, bounds, nmbacktrack, (), t = 2, der_only=True, hessfn=False)


#ans42 = pgrad_descent(rosen,rosen_der,None,p,bounds,backtrack,(), der_only=True, t=2,srch_type =2,c0=1e-2)

#ans2 = pgrad_descent(rosen, rosen_der, None, p, bounds, fixedstep,  (), der_only = True, c1=1e-1, c2=1,srch_type = 0, proj_type = 1)

#%%
#test = [1]
#def randomtest(test):
#    test[0] = 2
#    return
#
#randomtest(test)
#print(test)

def testing(*args):
    
    for i in args:
        print(i)
        
    return
testing(())