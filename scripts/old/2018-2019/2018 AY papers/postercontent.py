# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 21:44:13 2019

@author: rlk268

did some testing on the NGSim data and saved the results, some of these were useful in the papers because 
we were looking at the no lane changing vehicles and saved the results

ASSOCIATED PKL FILES - POSTERTEST AND POSTERTEST2
"""

from calibration import * 

def postertest(platooninfo,meas,noLC):
    
    noLClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])==1:
            noLClist.append([[],i])
    
#    noLClist = [[[],1014]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    #    noLC = calibrate_bfgs(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,7.5) #this has already been run in the relax testing results
    
    noLC_bfgs_f = calibrate_bfgs_f(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,7.5)
    
    noLC_GA = calibrate_GA(meas,platooninfo,noLClist,makeleadfolinfo,bounds_nor)
    
    noLC_NM = calibrate_NM(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor[0],bounds_nor)
    
    noLC2 = calibrate_bfgs(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,0)
    
    noLC_bfgs_f2 = calibrate_bfgs_f(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,0)
    
    plist = []
    for i in noLC[0]:
        plist.append(i[0])
        
    real = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    plist = []
    for i in noLC_bfgs_f[0]:
        plist.append(i[0])
        
    real_f = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    plist = []
    for i in noLC_GA[0]:
        plist.append(i['x'])
        
    real_GA = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    plist = []
    for i in noLC_NM[0]:
        plist.append(i['x'])
        
    real_NM = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    plist = []
    for i in noLC2[0]:
        plist.append(i[0])
        
    real2 = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    plist = []
    for i in noLC_bfgs_f2[0]:
        plist.append(i[0])
        
    real_f2 = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    
    
    return noLC, noLC_bfgs_f, noLC_GA, noLC_NM, noLC2, noLC_bfgs_f2, real, real_f, real_GA, real_NM, real2, real_f2
#################

#####################

#noLC, noLC_bfgs_f, noLC_GA, noLC_NM, noLC2, noLC_bfgs_f2, real, real_f, real_GA, real_NM, real2, real_f2 = postertest(platooninfo,meas,noLC)
#
#with open('postertest.pkl','wb') as f:
#    pickle.dump([noLC, noLC_bfgs_f, noLC_GA, noLC_NM, noLC2, noLC_bfgs_f2, real, real_f, real_GA, real_NM, real2, real_f2], f)
    
with open('postertest.pkl','rb') as f:
    noLC, noLC_bfgs_f, noLC_GA, noLC_NM, noLC2, noLC_bfgs_f2, real, real_f, real_GA, real_NM, real2, real_f2 = pickle.load(f)
    
    
    
def postertest2(platooninfo,meas):
    
    noLClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])==1:
            noLClist.append([[],i])
    
#    noLClist = [[[],1014]]
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    #    noLC = calibrate_bfgs(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,7.5) #this has already been run in the relax testing results
    
    noLC = calibrate_bfgs(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,float('inf'))
    
    noLC_bfgs_f = calibrate_bfgs_f(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,float('inf'))
    
    plist = []
    for i in noLC[0]:
        plist.append(i[0])
        
    real = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
    
    plist = []
    for i in noLC_bfgs_f[0]:
        plist.append(i[0])
        
    real_f = calibrate_check_realistic(meas,platooninfo,noLClist,makeleadfolinfo,plist)
        
    return noLC, noLC_bfgs_f, real, real_f


#noLC3, noLC_bfgs_f3, real3, real_f3 = postertest2(platooninfo,meas)

#with open('postertest2.pkl','wb') as f:
#    pickle.dump([noLC3, noLC_bfgs_f3, real3, real_f3], f)
    
#with open('postertest2.pkl','rb') as f:
#    noLC3, noLC_bfgs_f3, real3, real_f3 = pickle.load(f)
    
#%%
#test = []
##for i in range(len(noLC_bfgs_f2[0])):
##    test.append(noLC_GA[0][i]['nfev'])
#
##for i in range(len(noLC_bfgs_f2[0])):
##    test.append(noLC_bfgs_f2[0][i][2]['funcalls'])
#
#for i in range(len(noLC[0])):
#    test.append(noLC_bfgs_f[0][i][-1])
