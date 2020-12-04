
"""
@author: rlk268@cornell.edu
"""

import copy 
from havsim.calibration.calibration import calibrate_tnc2, calibrate_GA
from havsim.calibration.helper import makeleadfolinfo, obj_helper, calculate_rmse, re_diff
from havsim.calibration.models import OVM, OVMadjsys, OVMadj, IDM_b3, IDMadjsys_b3, IDMadj_b3, daganzo, daganzoadjsys, daganzoadj
from havsim.calibration.opt import platoonobjfn_obj, platoonobjfn_objder
#plotting 
from havsim.plotting import  platoonplot, plotvhd, animatevhd

def benchmark(meas, platooninfo, platoonlist, budget = 3, usemodel = 'IDM', order = 1, cutoff = 7.5, cutoff2 = 4.5, lane = None):
    #does not work for high D because the dt/h is not being passed correctly in calibrate_tnc
    #platoonlist = nested list of platoons e.g. [[1013]] or [[1013, 1019], [1123,124]]
    
    #can decrease budget for speed increase in some instances. default parameters are essentially for state of the art parametric model
    #usemodel = 'OVM' - less accurate than IDM
    #usemodel = 'IDM' most accurate, a bit slower than OVM
    #usemodel = 'LWR' - least accurate, fastest
    if usemodel == 'OVM':
        plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ], [10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
        bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
        model, modeladjsys, modeladj = OVM, OVMadjsys, OVMadj
        args = (True,6)
    elif usemodel == 'IDM': 
        plist = [[40,1,1,3,10,25], [60,1,1,3,10,5], [80,1,15,1,1,5]]
        bounds = [(20,120),(.1,3),(.1,35),(.1,5),(.1,5),(.1,75)]
        model, modeladjsys, modeladj = IDM_b3, IDMadjsys_b3, IDMadj_b3
        args = (True,6)
    else: 
        budget = 1
        plist = [[1,20,100,5]]
        bounds = [(.1,10), (0,100), (40, 120),(.1, 75)]
        model, modeladjsys, modeladj = daganzo, daganzoadjsys, daganzoadj
        args = (True,4)
        
    out = calibrate_tnc2(plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo,
                                      platoonobjfn_objder, None, model, modeladjsys, modeladj, *args, cutoff = cutoff, 
                                      cutoff2 = cutoff2, order = order, budget = budget)

    sim = copy.deepcopy(meas)
    sim = obj_helper(out[0], model, modeladjsys, modeladj, meas, sim, platooninfo, platoonlist, makeleadfolinfo, platoonobjfn_obj, args)
    
    vehlist = []
    [vehlist.extend(platoonlist[i]) for i in range(len(platoonlist))]
    
    rmselist = []
    [rmselist.append(calculate_rmse(meas,sim,platooninfo,vehlist[i])) for i in range(len(vehlist))]
    
    print('vehicles '+str(vehlist))
    print('rmse '+str(rmselist))
    
#    platoonplot(meas,sim,platooninfo,platoonlist,colorcode = False, lane = lane, opacity = .1)
    
    if usemodel == 'LWR':
        re_diff(sim,platooninfo,vehlist)
    
    #output is dictionary with simulated vehicles, list of vehicles, list of RMSEs
    return out, sim, vehlist, rmselist 


#out = benchmark(meas,platooninfo,[[1133,1145,1137,1153]]) #sometimes rmse can be very high (this takes ~1 minute to run)
#benchmark(meas,platooninfo,[[1133]]) #sometimes it can be very low 
#benchmark(meas,platooninfo,[[1137]]) #sometimes the problem is using larger platoons causes algorithm to get stuck in local minimum 
    
#out = benchmark(meas,platooninfo,[[2222]],usemodel='OVM')
#here is trajectories in speed/headway plane - maybe you find it interesting to look at trajectories this way 
#plotvhd(meas,out[1], platooninfo,1013) #like the below but no animation - in case animation is laggy 
#ani = animatevhd_list(meas,out[1],platooninfo,[1013])
#
#out = benchmark(meas,platooninfo,[[1013]],usemodel='LWR')
#ani2 = animatevhd_list(meas,out[1],platooninfo,[1013])

