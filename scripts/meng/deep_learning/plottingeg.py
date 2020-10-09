
"""
@author: rlk268@cornell.edu
"""

#assume you have data loaded already, have access to meas and platooninfo

from havsim.plotting import plotvhd
import matplotlib.pyplot as plt
from havsim.calibration.algs import makeplatoonlist
import pickle
from sklearn.cluster import KMeans
import havsim
import numpy as np


path_highd26 = '/Users/nathanbala/Desktop/meng_project/data/highd26.pkl'
path_reconngsim = '/Users/nathanbala/Desktop/meng_project/data/reconngsim.pkl'
# with open(path_highd26, 'rb') as f:
#    data = pickle.load(f)[0]
with open(path_reconngsim, 'rb') as f:
   data = pickle.load(f)[0]

meas, platooninfo = makeplatoonlist(data,1, False)

#
with open("data/extraq_diffarch/extraq_info_diffarch.pickle", 'rb') as f:
    data = pickle.load(f)

with open("data/extraq_diffarch/extraq_diffarch.pickle", 'rb') as f:
    data1 = pickle.load(f)










high_error = []
low_error = []

lange_change_error = []
stay_error = []
for i in data:
    if i == "note":
        continue
    t_nstar, t_n, T_nm1, T_n = platooninfo[i][:4]
    curmeas = meas[i][t_n-t_nstar:T_nm1+1-t_nstar,[2,3,7]]
    curr_lanes = curmeas[:,2]
    curr = np.unique(curr_lanes)
    data_tup = data[i]
    rmse = data_tup[0]
    if rmse != None:
        error = rmse.numpy()
        lane_change = data_tup[1]
        if lane_change == True:
            lange_change_error.append(error)
        else:
            stay_error.append(error)

print(np.mean(lange_change_error))
print(np.mean(stay_error))

data["note"] = "this is the results for highd data with three inputs (vec speed, lead speed, headway) with a statemem of 5 using normal NN with L2 reg"
data1["note"] = "this is the results for highd data with three inputs (vec speed, lead speed, headway) with a statemem of 5 using normal NN with L2 reg"



with open('data/extraq_diffarch/extraq_info_diffarch.pickle', 'wb') as handle:
   pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/extraq_diffarch/extraq_diffarch.pickle', 'wb') as handle:
   pickle.dump(data1, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(data["note"])
