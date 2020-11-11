# imports and load data
import matplotlib.pyplot as plt
from havsim.calibration import calibration
from havsim.simulation import road_networks
import pickle
import numpy as np
import math
import time
import havsim.helper as helper
from havsim import simulation
import scipy.optimize as sc
import havsim.calibration.make_calibration as mc
#%%
# please put pickle loading stuff in a seperate file not on github. Also note that recon-ngsim-old is
# what holds meas/platooninfo (still needed as not everything is converted)
#recon-ngsim now holds the same data in the updated format.
#also, recon-ngsim is in units of meters now instead of feet.
try:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim-old.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim-old.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

# try:
#     with open('/Users/nathanbala/Downloads/platoonlist.pkl', 'rb') as f:
#         platoon_list = pickle.load(f) #load data
# except:
#     with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
#         meas, platooninfo = pickle.load(f) #load data

# # print(meas)

try:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim.pkl', 'rb') as f:
        data = pickle.load(f) #load data
except:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim.pkl', 'rb') as f:
        data = pickle.load(f) #load data



#%%
# make downstream boundaries
lanes = {}
for i in range(1,7):
    unused, unused, exitspeeds, unused = helper.boundaryspeeds(meas, [], [i],.1,.1)
    exitspeeds[0].extend((exitspeeds[0][-1],)*1000)
    exitspeeds = road_networks.timeseries_wrapper(exitspeeds[0])
    downstream = {'method': 'speed', 'time_series':exitspeeds}
    lane_i = simulation.Lane(None, None, {'name':'idk'}, 0, downstream=downstream)
    lanes[i] = lane_i
lanes[999] = lanes[6]
lanes[7] = lanes[6]



#%%
curplatoon = [525, 530, 537, 545] 
calibration_args = {"parameter_dict" : None, "ending_position" : 1475/3.28084}
cal = mc.make_calibration(curplatoon, data, .1, mc.make_lc_events_new, lanes=lanes, calibration_kwargs = calibration_args)

# curplatoon = [1013, 1023, 1030, 1037, 1045]
# platoon_list = [[977.0, 3366.0, 774.0, 788.0]]
# for curplatoon in platoon_list:
#     # average_end_position = 0
#     # for i in curplatoon:
#     #     average_end_position += (meas[i][-1][2])
#     # average_end_position = average_end_position/len(curplatoon)
#     # print(average_end_position)
#     calibration_args = {"parameter_dict" : None, "ending_position" : 1475/3.28084}
#     pguess =  [40/3.28084, 1, 1, 3/3.28084, 10/3.28084, 25]*len(curplatoon)
#     mybounds = [(20,120), (.1,5), (.1,35), (.1,20), (.1,20), (.1,75)]*len(curplatoon)
#     start = time.time()
#     cal = calibration.make_calibration(curplatoon, vehdict, .1, calibration.CalibrationVehicle, lanes=lanes, **calibration_args)
#     print(curplatoon)
#     print('time to make calibrate is '+str(time.time()-start))
#     start = time.time()
#     cal.simulate(pguess)
#     print('time to simulate once is '+'{:2.2f}'.format(time.time()-start)+' for '+'{:2.2f}'.format(sum([len(veh.posmem) for veh in cal.all_vehicles]))+\
#           ' timesteps = '+'{:2.2f}'.format(sum([len(veh.posmem) for veh in cal.all_vehicles])/(time.time()-start))+' updates per second')
#     print("\n")



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
