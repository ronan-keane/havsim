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

try:
    with open('/Users/nathanbala/Desktop/MENG/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
# make downstream boundaries
lanes = {}
for i in range(1,7):
    unused, unused, exitspeeds, unused = helper.boundaryspeeds(meas, [], [i],.1,.1)
    exitspeeds = road_networks.timeseries_wrapper(exitspeeds[0])
    downstream = {'method': 'speed', 'time_series':exitspeeds}
    lane_i = simulation.Lane(None, None, None, None, downstream=downstream)
    lanes[i] = lane_i


curplatoon = [1013, 1023, 1030, 1037, 1045]

# calibration_args = {"parameter_type" : "non_uniform"}
# pguess =  [6,40,1,1,3,10,25,6,40,1,1,3,10,25,6,40,1,1,3,10,25,6,40,1,1,3,10,25,6,40,1,1,3,10,25] #[80,1,15,1,1,35] #

calibration_args = {"parameter_type" : "uniform"}
pguess =  [40,1,1,3,10,25,40,1,1,3,10,25,40,1,1,3,10,25,40,1,1,3,10,25,40,1,1,3,10,25]

mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
cal = calibration.make_calibration(curplatoon, meas, platooninfo, .1, calibration.CalibrationVehicle, lanes=lanes, calibration_kwargs=calibration_args)
start = time.time()
cal.simulate(pguess)
for veh in cal.all_vehicles:
    plt.plot(np.linspace(veh.inittime, veh.inittime+len(veh.posmem)-1, len(veh.posmem)), veh.posmem)
#     plt.plot(vec.posmem)
plt.show()
# start = time.time()
# if use_method == 'BFGS':
#     bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)  # BFGS
#     print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))
