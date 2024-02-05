
"""
@author: rlk268@cornell.edu

problem with optplot rmse not making sense; verified fixed 11/29, pushed to master. 
bug was for multiple guesses being used, when order = 1, if latter guesses are worse, the simulation used in 
future calibration would be the worse simulations, which messed up the results for calibrate_tnc2 function (main function used at that time)
"""
#%%

def test(opt,lists):
    sim = copy.deepcopy(meas)
    sim = helper.obj_helper(opt[0],OVM,OVMadj,OVMadjsys,meas,sim,platooninfo,lists,makeleadfolinfo,platoonobjfn_obj,(True,6))
    for count, i in enumerate(lists):
        obj = helper.SEobj_pervehicle(meas,sim,platooninfo,i)
        print('optimization result is '+str(opt[0][count][-2])+', our result is '+str(obj[0]))
        
test(out[0],lists[0])
#%% reproduce problem 
platoonlist = lists[0][:5]
outtest = calibrate_tnc2(plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,True,6,cutoff=0,cutoff2=4.5,order=1,budget = 3)
test(outtest,platoonlist)
#%%test fix 
from havsim.calibration.calibration import calibrate_tnc2 
platoonlist = lists[0][:5]
outtest2 = calibrate_tnc2(plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,True,6,cutoff=0,cutoff2=4.5,order=1,budget = 3)
test(outtest2,platoonlist)
