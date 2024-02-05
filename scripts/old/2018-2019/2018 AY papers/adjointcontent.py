# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 02:11:16 2019

@author: Pc

main data which was used in the adjoint paper. 
pickle files - adjointtest1 - adjointtest11
"""
#from calibration import *  #old code was all in calibration file 
import os
os.chdir('C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/2018 AY papers') 

def adjointtest(platooninfo,meas):
            
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    bfgs_1 = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,cutoff = 0)
    bfgs_2 = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,cutoff = 7.5)
    bfgs_3 = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,cutoff = float('inf'))
    
    GA = calibrate_GA(bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj,None,OVM,OVMadjsys,OVMadj)
    
    NM = calibrate_NM(plist_nor[0],meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj_b,None,OVM,OVMadjsys,OVMadj,bounds_nor)
    
    return bfgs_1,bfgs_2,bfgs_3, GA, NM

#bfgs_1,bfgs_2,bfgs_3, GA, NM = adjointtest(platooninfo,meas)

#with open('adjointtest.pkl','wb') as f:
#    pickle.dump([bfgs_1,bfgs_2,bfgs_3, GA, NM],f)
    
with open('adjointtest.pkl','rb') as f:
    bfgs_1,bfgs_2,bfgs_3, GA, NM = pickle.load(f)
    
def adjointtest2(platooninfo,meas):
            
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    bfgs_1 = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj,platoonobjfn_fder,OVM,OVMadjsys,OVMadj,cutoff = 0)
    bfgs_2 = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj,platoonobjfn_fder,OVM,OVMadjsys,OVMadj,cutoff = 7.5)
    bfgs_3 = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj,platoonobjfn_fder,OVM,OVMadjsys,OVMadj,cutoff = float('inf'))
    
    return bfgs_1,bfgs_2,bfgs_3

#bfgsf_1,bfgsf_2,bfgsf_3 = adjointtest2(platooninfo,meas)

#with open('adjointtest2.pkl','wb') as f:
#    pickle.dump([bfgsf_1,bfgsf_2,bfgsf_3],f)
    
with open('adjointtest2.pkl','rb') as f:
    bfgsf_1,bfgsf_2,bfgsf_3 = pickle.load(f)
    
def adjointtest3(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 3000, 'c1':5e-4, 'c2' : 1}
    
    custom1 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SPSA,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               backtrack, kwarg, False, 5, cutoff = 0)
    
    return custom1


    
def adjointtest4(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3}
    
    custom1 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 0)
    
    custom2 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 7.5)
    
    custom3 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = float('inf'))
    
    return custom1, custom2, custom3, 
    
#custom11, custom12, custom13 = adjointtest4(platooninfo,meas)

#with open('adjointtest4.pkl','wb') as f:
#    pickle.dump([custom11, custom12, custom13],f)
    
with open('adjointtest4.pkl','rb') as f:
    custom11, custom12, custom13 = pickle.load(f)

def adjointtest5(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_nor = np.asarray(plist_nor)
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 200, 'srch_type':1, 'eps':5e-7, 't':2}
    
    custom1 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SQP2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 0)
    
    custom2 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SQP2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 7.5)
    
    custom3 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SQP2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = float('inf'))
    
    return custom1, custom2, custom3

#custom21, custom22, custom23 = adjointtest5(platooninfo,meas)

#with open('adjointtest5.pkl','wb') as f:
#    pickle.dump([custom21, custom22, custom23],f)
    
with open('adjointtest5.pkl','rb') as f:
    custom21, custom22, custom23 = pickle.load(f)
    
    
#custom1 = adjointtest3(platooninfo,meas)  

#with open('adjointtest3.pkl','wb') as f:
#    pickle.dump([custom1],f)
    
with open('adjointtest3.pkl','rb') as f:
    custom1 = pickle.load(f)
    
def adjointtest6(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3, 'der_only':True}
    
    custom3 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_fder, platoonobjfn_fder, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = float('inf'))
    
    return custom3

#custom4 = adjointtest6(platooninfo,meas)

#with open('adjointtest6.pkl','wb') as f:
#    pickle.dump([custom4],f)
    
with open('adjointtest6.pkl','rb') as f:
    custom4 = pickle.load(f)
    
def adjointtest7(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_nor = np.asarray(plist_nor)
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 200, 'srch_type':1, 'eps':5e-7, 't':2, 'der_only':True}
    
    custom3 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SQP2,platoonobjfn_obj,platoonobjfn_fder, platoonobjfn_fder, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = float('inf'))
    
    return custom3

#custom5 = adjointtest7(platooninfo,meas)

#with open('adjointtest7.pkl','wb') as f:
#    pickle.dump([custom5],f)
    
with open('adjointtest7.pkl','rb') as f:
    custom5 = pickle.load(f)
    
def adjointtest8(platooninfo,meas):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
#    kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3}
    
    custom1 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = 0)
    
    custom2 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = 7.5)
    
    custom3 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_objder, None, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = float('inf'))
    
    return custom1, custom2, custom3, 
    
#custom61, custom62, custom63 = adjointtest8(platooninfo,meas)


#with open('adjointtest8.pkl','wb') as f:
#    pickle.dump([custom61, custom62, custom63],f)
    
with open('adjointtest8.pkl','rb') as f:
    custom61, custom62, custom63 = pickle.load(f)
    
def adjointtest9(platooninfo,meas,objder = False):
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
#    kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3}
    
    custom3 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj, platoonobjfn_fder, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = float('inf'), objder=False)
    
    return custom3

#custom64= adjointtest9(platooninfo,meas)


#with open('adjointtest9.pkl','wb') as f:
#    pickle.dump([custom64],f)
    
with open('adjointtest9.pkl','rb') as f:
    custom64 = pickle.load(f)
    
def adjointtest10(platooninfo,meas): #need to redo these to correct the results at some point. This will take roughly 45 hours to run. 
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_nor = np.asarray(plist_nor)
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    kwarg = {'maxit' : 200, 'srch_type':1, 'eps':5e-7, 't':2}
    
    custom1 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SQP2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 0)
    
    custom2 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,SQP2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 7.5)
    
    
    kwarg = {'maxit' : 1000, 'srch_type':1, 'eps':5e-7, 't':3}
    
    custom3 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 0)
    
    custom4 = calibrate_custom(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,pgrad_descent2,platoonobjfn_obj,platoonobjfn_objder, platoonobjfn_der, OVM, OVMadjsys, OVMadj, 
                               nmbacktrack, kwarg, False, 5, cutoff = 7.5)
    
    return custom1, custom2, custom3, custom4


    
def adjointtest11(platooninfo,meas): #would also like to get some results for the finite difference applied to TNC since that algorithm worked quite well. 
    #this should take about 8-10 hours to run
    vehlist = []
    for i in meas.keys():
        if len(platooninfo[i][4]) >=1:
            vehlist.append([[],i])
            
#plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
#                     linesearch, kwargs, *args, cutoff=7.5,
    
#    vehlist = [[[],1013]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_nor = np.asarray(plist_nor)
    
#    prearg = (platoonobjfn_obj,)
#    postarg = (bounds_nor, (OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, i, leadinfo, folinfo, rinfo, False, 5))
    custom1 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj, platoonobjfn_fder, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = 0, objder=False)
    custom2 = calibrate_tnc(plist_nor,bounds_nor,meas,platooninfo,vehlist,makeleadfolinfo,platoonobjfn_obj, platoonobjfn_fder, OVM, OVMadjsys, OVMadj, 
                               False, 5, cutoff = 7.5, objder=False)
    
    
    return custom1, custom2

#custom66, custom65 = adjointtest11(platooninfo,meas)

#with open('adjointtest11.pkl','wb') as f:
#    pickle.dump([custom66, custom65],f)

with open('adjointtest11.pkl','rb') as f:
    custom66, custom65 = pickle.load(f)
    
#custom11, custom12, custom21, custom22 = adjointtest10(platooninfo,meas)

#with open('adjointtest10.pkl','wb') as f:
#    pickle.dump([custom11, custom12, custom21, custom22],f)
