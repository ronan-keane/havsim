
"""
@author: rlk268@cornell.edu
"""
import havsim.calibration.calibration as hc
import time
import scipy.optimize as sc
import matplotlib.pyplot as plt
import math
import pickle
import havsim.calibration.calibration_models as hm

# load data
try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

# categorize vehicles
veh_list = meas.keys()
merge_list = []
lc_list = []
nolc_list = []
for veh in veh_list:
    t_nstar, t_n = platooninfo[veh][0:2]
    if t_n > t_nstar and meas[veh][t_n-t_nstar-1,7]==7 and meas[veh][t_n-t_nstar,7]==6:
        merge_list.append(veh)
    elif len(platooninfo[veh][4]) > 1:
        lc_list.append(veh)
    elif len(platooninfo[veh][4]) == 1:
        nolc_list.append(veh)


# define training loop
def training_ga(veh_id_list, bounds, meas, platooninfo, dt, workers = 2, kwargs = {}):
    """Runs differential evolution to fit parameters for a list of CalibrationVehicle's"""
    #veh_id_list = list of float vehicle id, bounds = bounds for optimizer (list of tuples),
    #kwargs = dictionary with keyword arguments for hc.make_calibration
    out = []
    for veh_id in veh_id_list:
        cal = hc.make_calibration([veh_id], meas, platooninfo, dt, **kwargs)
        ga = sc.differential_evolution(cal.simulate, bounds = bounds, workers = workers)
        out.append(ga)

    return out


def training(plist, veh_id_list, bounds, meas, platooninfo, dt, vehicle_object, cutoff = 6, kwargs = {}):
    """Runs bfgs with multiple initial guesses to fit parameters for a CalibrationVehicle"""
    #veh_id = float vehicle id, plist = list of parameters, bounds = bounds for optimizer (list of tuples),
    #cutoff = minimum mse required for multiple guesses
    #kwargs = dictionary with keyword arguments for hc.make_calibration
    out = []
    for veh_id in veh_id_list:
        cal = hc.make_calibration([veh_id], meas, platooninfo, dt, **kwargs)
        bestmse = math.inf
        best = None
        for guess in plist:
            bfgs = sc.fmin_l_bfgs_b(cal.simulate, guess, bounds = bounds, approx_grad=1)
            if bfgs[1] < bestmse:
                best = bfgs
                bestmse = bfgs[1]
            if bestmse < cutoff:
                break
        out.append(best)
    return out


class NoRelaxIDM(hc.CalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):  # just need to set parameters correctly
        super().initialize(parameters)
        self.cf_parameters = parameters

class NoRelaxOVM(hm.OVMCalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters

class NoRelaxNewell(hm.NewellCalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters

#%%  # updated, but not tested, after the 'refactored calibration + added calibration_models' commit
"""Used GA + ballistic update for paper results. Using euler update is probably better in terms of mse.
Can use BFGS instead of GA, which is significantly faster, but can have problems with local minima."""
"""
Run 1: IDM with no accident-free relax, no max speed bound, no acceleration bound (only for merge, lc)
"""
plist = [[40,1,1,3,10,25], [60,1,1,3,10,5], [80,1,15,1,1,35], [70,2,10,2,2,15]]
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
relax_lc_res = training_ga(lc_list,  bounds, meas, platooninfo, .1)
relax_merge_res = training_ga(merge_list, bounds, meas, platooninfo, .1)

with open('IDMrelax.pkl','wb') as f:
    pickle.dump((relax_lc_res,relax_merge_res), f)

# """
# Run 2: Like Run 1, but with relax disabled. (for all vehicles)
# """
# plist = [[40,1,1,3,10], [60,1,1,3,10], [80,1,15,1,1], [70,2,10,2,2]]
# bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
# kwargs = {'vehicle_class': NoRelaxIDM}
# norelax_lc_res = training_ga(lc_list, bounds, meas, platooninfo, .1 , kwargs = kwargs)
# norelax_merge_res = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs = kwargs)
# norelax_nolc_res = training_ga(nolc_list, bounds, meas, platooninfo, .1, kwargs = kwargs)

# with open('IDMnorelax.pkl','wb') as f:
#     pickle.dump((norelax_lc_res,norelax_merge_res,norelax_nolc_res),f)

"""
Run 3: OVM with no accident-free relax, no max speed bound, no acceleration bound (only for merge, lc)
"""
plist = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ], [20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],
         [10*3.3,.086/3.3/2, .5, .5, .175, 60 ], [25,.05, 1,3, 1, 25]]
bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
kwargs = {'vehicle_class': hm.OVMCalibrationVehicle}
relax_lc_res_ovm = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs = kwargs)
relax_merge_res_ovm = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs = kwargs)

with open('OVMrelax.pkl', 'wb') as f:
    pickle.dump((relax_lc_res_ovm, relax_merge_res_ovm),f)


# """
# Run 4: Like Run 3, but with relax disabled. (for all vehicles)
# """
# plist = [[10*3.3,.086/3.3, 1.545, 2, .175], [20*3.3,.086/3.3/2, 1.545, .5, .175 ],
#          [10*3.3,.086/3.3/2, .5, .5, .175 ], [25,.05, 1,3, 1]]
# bounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
# kwargs = {'vehicle_class': NoRelaxOVM}
# norelax_lc_res_ovm = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs)
# norelax_merge_res_ovm = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs)
# norelax_nolc_res_ovm = training_ga(nolc_list, bounds, meas, platooninfo, .1, kwargs)

# with open('OVMnorelax.pkl', 'wb') as f:
#     pickle.dump((norelax_lc_res_ovm, norelax_merge_res_ovm, norelax_nolc_res_ovm),f)


"""
Run 7: Try existing Relaxation model due to Schakel, Knoop, van Arem (2012)
"""

plist = [[40,1,1,3,10,1, 25], [60,1,1,3,10,1,5], [80,1,15,1,1,1,35], [70,2,10,2,2,2,15]]
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,5),(.101,75)]
kwargs = {'vehicle_class': hm.SKA_IDM}
relax_lc_res_ska = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
relax_merge_res_ska = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

with open('SKArelax.pkl', 'wb') as f:
    pickle.dump([relax_lc_res_ska, relax_merge_res_ska],f)

"""
2 Parameter positive/negative relax IDM
"""
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,5),(.1,75),(.1,75)]
kwargs = {'vehicle_class': hm.Relax2IDM}
relax_lc_res_2p = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
relax_merge_res_2p = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

with open('2pIDM.pkl', 'wb') as f:
    pickle.dump([relax_lc_res_2p, relax_merge_res_2p],f)

# """
# 2 parameter shape/time relax IDM
# """
# bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,5),(.1,75),(-1,1)]
# kwargs = {'vehicle_class': hm.RelaxShapeIDM}
# relax_lc_res_2ps = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
# relax_merge_res_2ps = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

# with open('2psIDM.pkl', 'wb') as f:
#     pickle.dump([relax_lc_res_2ps, relax_merge_res_2ps],f)


"""
Run 5: Newell with no accident free
"""
bounds = [(.1,10),(0,100),(40,120),(.1,75)]
kwargs = {'vehicle_class': hm.NewellCalibrationVehicle}
relax_lc_res_newell = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
relax_merge_res_newell = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

with open('Newellrelax.pkl','wb') as f:
    pickle.dump([relax_lc_res_newell, relax_merge_res_newell], f)


"""
Run 6: Like Run 5, but with no relax
"""
bounds = [(.1,10),(0,100),(40,120)]
kwargs = {'vehicle_class': NoRelaxNewell}
norelax_lc_res_newell = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs = kwargs)
norelax_merge_res_newell = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs = kwargs)
norelax_nolc_res_newell = training_ga(nolc_list, bounds, meas, platooninfo, .1, kwargs = kwargs)

with open('Newellnorelax.pkl','wb') as f:
    pickle.dump([norelax_lc_res_newell, norelax_merge_res_newell, norelax_nolc_res_newell], f)


#%%
"""
LL Relaxation Model
"""
bounds = [(1,100),(1,120),(40,120),(.5, 20)]
kwargs = {'vehicle_class': hm.NewellLL, 'event_maker':hm.make_ll_lc_event, 'lc_event_fun':hm.ll_lc_event}
relax_lc_res_ll = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
relax_merge_res_ll = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

with open('NewellLL.pkl', 'wb') as f:
    pickle.dump([relax_lc_res_ll, relax_merge_res_ll], f)

#%%
"""
Exponential Relaxation
"""
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
kwargs = {'vehicle_class': hm.RelaxExpIDM}
relax_lc_res_exp = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
relax_merge_res_exp = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

with open('ExpIDM.pkl', 'wb') as f:
    pickle.dump([relax_lc_res_exp, relax_merge_res_exp], f)

#%%
"""
"""
bounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,5),(.1,75),(.1,75)]
kwargs = {'vehicle_class': hm.Relax2vhdIDM}
relax_lc_res_2p = training_ga(lc_list, bounds, meas, platooninfo, .1, kwargs= kwargs)
relax_merge_res_2p = training_ga(merge_list, bounds, meas, platooninfo, .1, kwargs= kwargs)

with open('2pvhdIDM.pkl', 'wb') as f:
    pickle.dump([relax_lc_res_2p, relax_merge_res_2p],f)
