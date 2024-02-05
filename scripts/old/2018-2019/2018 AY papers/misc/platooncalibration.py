# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 03:56:12 2019

@author: rlk268
"""
from calibration import * 
from findoscillations import * 
import nlopt

def platooncalibration(meas, platooninfo, platoonlist): 
    #calibrate a set number of platoons with 1 parameter relax 
    
    
    
    pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5]
    plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds 
    
    out1 = calibrate_bfgs2(plist,mybounds,meas,platooninfo,platoonlist, makeleadfolinfo_r3,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,True,6,
                           cutoff=0,cutoff2=0, order = 1, dim = 2, budget = 1)
    
    out2 = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlist, makeleadfolinfo_r3,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,True,6,
                           cutoff=0,cutoff2=0, order = 1, dim = 2, budget = 1)
    
    
    
    pguess = [1,20,100,5] #daganzo guess 1 
    plist = [[1,20,100,5]]
    mybounds = [(.1,10),(0,100),(40,120),(.1,75)] #fairly conservative bounds
    
    
    out11 = calibrate_bfgs2(plist,mybounds,meas,platooninfo,platoonlist, makeleadfolinfo_r3,platoonobjfn_objder,None,daganzo, daganzoadjsys, daganzoadj, True, 4,
                           cutoff=0,cutoff2=0, order = 1, dim = 2, budget = 1)
    
    out21 = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlist, makeleadfolinfo_r3,platoonobjfn_objder,None,daganzo, daganzoadjsys, daganzoadj, True, 4,
                           cutoff=0,cutoff2=0, order = 1, dim = 2, budget = 1)
    
    return out1, out2, out11, out21
#    return out4

def platooncalibration2(meas, platooninfo, platoonlist): 
    #calibrate a set number of platoons with 1 parameter relax 
    
    
    
    pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5]
    plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)] #less conservative bounds 

    
    out3 = calibrate_nlopt(nlopt.GN_DIRECT_L,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           OVM, OVMadjsys, OVMadj, True,6, evalper = None, order = 1, dim = 2)
    
    out4 = calibrate_nlopt(nlopt.GN_CRS2_LM,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           OVM, OVMadjsys, OVMadj, True,6, evalper = None, order = 1, dim = 2)
    
    out5 = calibrate_nlopt(nlopt.GN_ISRES,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           OVM, OVMadjsys, OVMadj, True,6, evalper = None, order = 1, dim = 2)
    
    out6 = calibrate_nlopt(nlopt.GN_ESCH,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           OVM, OVMadjsys, OVMadj, True,6, evalper = None, order = 1, dim = 2)
    
    
    
    pguess = [1,20,100,5] #daganzo guess 1 
    plist = [[1,20,100,5]]
    mybounds = [(.1,10),(0,100),(40,120),(.1,75)] #fairly conservative bounds
    
    
    out31 = calibrate_nlopt(nlopt.GN_DIRECT_L,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           daganzo, daganzoadjsys, daganzoadj, True, 4, evalper = 30, order = 1, dim = 1)
    
    out41 = calibrate_nlopt(nlopt.GN_CRS2_LM,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           daganzo, daganzoadjsys, daganzoadj, True, 4, evalper = 30, order = 1, dim = 1)
    
    out51 = calibrate_nlopt(nlopt.GN_ISRES,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           daganzo, daganzoadjsys, daganzoadj, True, 4, evalper = 30, order = 1, dim = 1)
    
    out61 = calibrate_nlopt(nlopt.GN_ESCH,pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo_r3, platoonobjfn_obj, platoonobjfn_der, 
                           daganzo, daganzoadjsys, daganzoadj, True, 4, evalper = 30, order = 1, dim = 1)
    
    return out3, out4, out5, out6, out31, out41, out51, out61
#    return out4

#platoonlist = [[[],581], [[], 611, 1013]] #test 
    
def dothestuff(data):
    for i in range(6):
        meas, platooninfo, platoonlist = makeplatoonlist(data,n=i+1,lane=2,vehs=[582,1146])
        out1, out2, out11, out21 = platooncalibration(meas,platooninfo,platoonlist)
        
        curstring = 'platooncal'+str(i+1)+'.pkl'
        with open(curstring,'wb') as f:
            pickle.dump([out1, out2, out11, out21], f)
            
    for i in range(6):
        meas, platooninfo, platoonlist = makeplatoonlist(data,n=i+1,lane=2,vehs=[582,1146])
        out3, out4, out5, out6, out31, out41, out51, out61 = platooncalibration2(meas,platooninfo,platoonlist)
        
        curstring = 'platooncalglobal'+str(i+1)+'.pkl'
        with open(curstring,'wb') as f:
            pickle.dump([out3, out4, out5, out6, out31, out41, out51, out61], f)
        
    
    return 0

blahblah = dothestuff(data)


#out1, out2, out3, out4, out5, out6, out11, out21, out31, out41, out51, out61 = platooncalibration(meas,platooninfo,platoonlist)


#with open('platooncal.pkl','wb') as f:
#    pickle.dump([out1, out2, out3, out4, out5, out6, out11, out21, out31, out41, out51, out61],f)

#%%


