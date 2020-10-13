# imports and load data
import matplotlib.pyplot as plt
from havsim.calibration import calibration
from havsim.simulation import road_networks
import pickle
import numpy as np
import tensorflow as tf
import math
import time
import havsim.helper as helper
from havsim import simulation
import scipy.optimize as sc

try:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

try:
    with open('/Users/nathanbala/Downloads/platoonlist.pkl', 'rb') as f:
        platoon_list = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

# print(meas)

#%%
# make downstream boundaries
lanes = {}
for i in range(1,7):
    unused, unused, exitspeeds, unused = helper.boundaryspeeds(meas, [], [i],.1,.1)
    exitspeeds = road_networks.timeseries_wrapper(exitspeeds[0])
    downstream = {'method': 'speed', 'time_series':exitspeeds}
    lane_i = simulation.Lane(None, None, {'name':'idk'}, 0, downstream=downstream)
    lanes[i] = lane_i



#%%
# curplatoon = [3244.0, 3248.0, 3258.0, 3259.0, 3262.0]
# curplatoon = [1013, 1023, 1030, 1037, 1045]
for curplatoon in platoon_list:
    average_end_position = 0
    for i in curplatoon:
        average_end_position += (meas[i][-1][2])
    average_end_position = average_end_position/len(curplatoon)
    print(average_end_position)
    calibration_args = {"parameter_dict" : None, "ending_position" : average_end_position}
    pguess =  [40,1,1,3,10,25]*len(curplatoon)
    mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)] * len(curplatoon)
    start = time.time()
    cal = calibration.make_calibration(curplatoon, meas, platooninfo, .1, calibration.CalibrationVehicle, lanes=lanes, calibration_kwargs=calibration_args)
    print(curplatoon)
    print('time to make calibrate is '+str(time.time()-start))
    start = time.time()
    cal.simulate(pguess)
    print('time to simulate once is '+str(time.time()-start))
    print("\n")



# for veh in cal.all_vehicles:
#     plt.plot(np.linspace(veh.inittime, veh.inittime+len(veh.posmem)-1, len(veh.posmem)), veh.posmem)
# plt.figure()
# for i in range(len(curplatoon)):
#     t_nstar, t_n, T_nm1 = platooninfo[curplatoon[i]][:3]
#     plt.plot(list(range(t_n, T_nm1+1)), meas[curplatoon[i]][t_n-t_nstar:T_nm1+1-t_nstar,2], 'C0')
# #     plt.plot(vec.posmem)
# plt.show()


# start = time.time()
# bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)  # BFGS
# print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))
