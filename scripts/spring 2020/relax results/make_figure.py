
"""
@author: rlk268@cornell.edu
"""
import havsim.plotting as hp
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
from havsim import calibration
from havsim.calibration import calibration_models
import tensorflow as tf
import pickle
#%%  # plot a platoon of vehicles
vehs = [1145, 1152, 1155, 1163, 1173, 1174, 1183, 1177, 1184, 1193, 1200, 1214, 1215, 1219,
        1226, 1232, 1241, 1240, 1243, 1252, 1260, 1266, 1267, 1275, 1278, 1282, 1283, 1293, 1290,
        1298, 1302, 1313, 1320, 1323, 1330, 1336, 1340, 1347, 1352, 1361, 1368, 1374,
        1375, 1379, 1384, 1396, 1391]

hp.platoonplot(meas, None, platooninfo, platoon = vehs, lane = 6, opacity = 0, colorcode = False)
plt.xlabel('time (.1 s)')
plt.savefig('relaxspacetime.png', dpi = 200)

#%%  Just for looking at different vehicles to find some pretty ones
inds = [66, 67, 68, 78, 79, 85]
for ind in inds:
    veh = [lc_list[ind]]
    hp.plotvhd(meas, None, platooninfo, veh, plot_color_line = True, draw_arrow = True)

    #%%
inds = [100, 101, 102, 103]
for ind in inds:
    veh = [nolc_list[ind]]
    hp.plotvhd(meas, None, platooninfo, veh, plot_color_line = True, draw_arrow = True)

#%%
inds = [58, 59, 60, 61, 62, 63]
for ind in inds:
    veh = [merge_list[ind]]
    hp.plotvhd(meas, None, platooninfo, veh, plot_color_line = True, draw_arrow = True)

    #%%  # for plots of linear solution - coefficients determined from mathematica notebook
def baseline_1_order(t,a,c,x0,v):
    return v-math.exp(-a*t)*(a*c+v+a*x0)
    # return -(a*c+v)/a+v*t+(a*c+v+a*x0)/a*math.exp(-a*t)

def relax_1_order(t,a,c,x0,v,g,r):
    # return (-g+a*c*r-a*g*r+r*v+a*r*x0)/(a*r)*math.exp(-a*t)+(-g+a*c*r-a*g*r+r*v)/(a*r)-(g-r*v)/r*t
    return -(g-r*v)/r-(math.exp(-a*t)*(-g+a*c*r-a*g*r+r*v+a*r*x0))/r

def relax_1_order2(t,a,c,x0,v,g,r):
    def helper(t,a,c,x0,v,g,r):
        return (-g+a*c*r-a*g*r+r*v+a*r*x0)/(a*r)*math.exp(-a*t)-(-g+a*c*r-a*g*r+r*v)/(a*r)-(g-r*v)/r*t
    x0 = helper(r,a,c,x0,v,g,r)
    # x0 = 269.7
    # return t*v-(a*c+v)/a+(math.exp(15*a-a*t)*(a*c+v-c*a*v+a*x0))/a
    return v - math.exp(15*a-a*t)*(a*c+v-15*a*v+a*x0)

plt.close('all')
plt.figure()
plt.subplot(1,2,1)
a,c,x0,v,g,r = 2/3, 2, -15, 20, 17, 15
npts = 100
pre_t = np.linspace(-5,0,npts)
pre_v = (v,)*npts
post_t = np.linspace(0,25,25*npts)
baseline_v = [baseline_1_order(t,a,c,x0,v) for t in post_t]

relax_t1 = np.linspace(0,15,15*npts)
relax_v1 = [relax_1_order(t,a,c,x0,v,g,r) for t in relax_t1]
relax_t2 = np.linspace(15,25,10*npts)
relax_v2 = [relax_1_order2(t,a,c,x0,v,g,r) for t in relax_t2]
plt.plot(pre_t, pre_v, 'k--', alpha = .8)
plt.plot(post_t, baseline_v, 'C0')
plt.plot(relax_t1, relax_v1, 'C1')
plt.legend(['Speed before LC', 'Baseline speed after LC', 'Relaxed speed after LC'])
plt.title('1st order ODE (Newell Model)')
plt.ylabel('speed')
plt.xlabel('time')
plt.plot(relax_t2, relax_v2, 'C1')

plt.subplot(1,2,2)
def baseline_2_order(t):
    return -1.632*math.exp(-.4*t) + .612*math.exp(-.15*t)
pre_t = np.linspace(-5,0,npts)
pre_v = (0,)*npts
baseline_t = np.linspace(0,40,40*npts)
baseline_a = [baseline_2_order(t) for t in baseline_t]
def relax_2_order(t):
    return .272*math.exp(-.4*t)-.272*math.exp(-.15*t)

def relax_2_order2(t):
    return -109.461*math.exp(-.4*t) + 2.30866*math.exp(-.15*t)
relax_t1 = np.linspace(0,15,15*npts)
relax_v1 = [relax_2_order(t) for t in relax_t1]
relax_t2 = np.linspace(15,40,25*npts)
relax_v2 = [relax_2_order2(t) for t in relax_t2]
plt.plot(pre_t, pre_v, 'k--', alpha = .8)
plt.plot(baseline_t, baseline_a, 'C0')
plt.plot(relax_t1, relax_v1, 'C1')
plt.legend(['Acc. before LC', 'Baseline acc. after LC', 'Relaxed acc. after LC'])
plt.plot(relax_t2, relax_v2, 'C1')

plt.title('2nd order ODE (linearized IDM)')
plt.ylabel('acceleration')
plt.xlabel('time')


#%%  need the output from /relax results/NN.py 
# _r -> the output from DL2_relax which was calibrated with relaxation on all vehicles = trained LSTM with relax 
# _nor -> the output from DL2 which was calibrated on vehicles with no lane changing = trained LSTM
# need to run sections 0, 3, 4 in NN.py in scripts/spring 2020/relax results

with open('RNNCFtraj.pkl', 'rb') as f:
    nolc_r_res, out3, lc_r_res, out4 = pickle.load(f)
with open('RNNCFtraj_nor.pkl', 'rb') as f:
    nolc_nor_res, out1, lc_nor_res, out2 = pickle.load(f)
nolc_r_res_list = make_into_analyze_res_format(nolc_r_res, list(nolc_ds.keys()), meas, platooninfo)
nolc_nor_res_list = make_into_analyze_res_format(nolc_nor_res, list(nolc_ds.keys()), meas, platooninfo)
lc_r_res_list = make_into_analyze_res_format(lc_r_res, list(lc_ds.keys()), meas, platooninfo)
lc_nor_res_list = make_into_analyze_res_format(lc_nor_res, list(lc_ds.keys()), meas, platooninfo)

#%%
C0 = plt.plot([], 'C0')
C1 = plt.plot([], 'C1')
plt.close('all')

#%%  code to plot LSTM model
sim_rnn_nor = copy.deepcopy(meas)
sim_rnn = copy.deepcopy(meas)

veh_to_plot = [1977]
try:
    for veh in veh_to_plot:
        t0, t1, t2, t3 = platooninfo[veh][:4]
        sim_rnn_nor[veh][t1-t0:t2-t0+1,2] = nolc_nor_res_list[veh]['posmem']
        sim_rnn_nor[veh][t1-t0:t2-t0+1,3] = nolc_nor_res_list[veh]['speedmem']
        sim_rnn[veh][t1-t0:t2-t0+1,2] = nolc_r_res_list[veh]['posmem']
        sim_rnn[veh][t1-t0:t2-t0+1,3] = nolc_r_res_list[veh]['speedmem']
except:
    for veh in veh_to_plot:
        t0, t1, t2, t3 = platooninfo[veh][:4]
        sim_rnn_nor[veh][t1-t0:t2-t0+1,2] = lc_nor_res_list[veh]['posmem']
        sim_rnn_nor[veh][t1-t0:t2-t0+1,3] = lc_nor_res_list[veh]['speedmem']
        sim_rnn[veh][t1-t0:t2-t0+1,2] = lc_r_res_list[veh]['posmem']
        sim_rnn[veh][t1-t0:t2-t0+1,3] = lc_r_res_list[veh]['speedmem']

# hp.plotvhd(meas, sim_rnn_nor, platooninfo, veh_to_plot, draw_arrow=True, arrow_interval=20)
# plt.legend([C1[0], C0[0]], ['LSTM no relax', 'data'])
# hp.plotvhd(meas, sim_rnn, platooninfo, veh_to_plot, draw_arrow=True, arrow_interval=20)
# plt.legend([C1[0], C0[0]], ['LSTM with relax','data'])
t = np.linspace(t1, t2, t2+1-t1)
plt.figure()
plt.plot(t, meas[veh][t1-t0:t2-t0+1,3], t, sim_rnn_nor[veh][t1-t0:t2-t0+1,3], t, sim_rnn[veh][t1-t0:t2-t0+1,3] )
plt.legend(['data', 'LSTM no relax', 'LSTM with relax'])
plt.title('speed-time for vehicle '+str(int(veh)))
plt.ylabel('speed (ft/s)')
plt.xlabel('time index (.1s)')

#%% code to plot parametric models - load results from /scripts/spring 2020/relax results
with open('IDMrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs_idm = {}

with open('IDMnorelax.pkl', 'rb') as f:
    idmres1, idmres2, res3 = pickle.load(f)
idmres1.extend(idmres2)
kwargs_idm_nor = {'vehicle_class':calibration_models.NoRelaxIDM}

with open('Newellrelax.pkl', 'rb') as f:
    nnres1, nnres2 = pickle.load(f)
nnres1.extend(nnres2)
kwargs_nn = {'vehicle_class':calibration_models.NewellCalibrationVehicle}

with open('Newellnorelax.pkl', 'rb') as f:
    nmres1, nmres2, nnres3 = pickle.load(f)
nmres1.extend(nmres2)
kwargs_nn_nor = {'vehicle_class':calibration_models.NoRelaxNewell}

with open('NewellLL.pkl', 'rb') as f:
    llres1, llres2 = pickle.load(f)
llres1.extend(llres2)
kwargs_ll = {'vehicle_class': calibration_models.NewellLL, 'event_maker':calibration_models.make_ll_lc_event, 'lc_event_fun':calibration_models.ll_lc_event}

with open('SKArelax.pkl', 'rb') as f:
    skares1, skares2 = pickle.load(f)
skares1.extend(skares2)
kwargs_ska = {'vehicle_class': calibration_models.SKA_IDM}


#%%
veh_to_plot = [1977]
veh = veh_to_plot[0]
t0, t1, t2, t3 = platooninfo[veh][:4]
sim_idm = copy.deepcopy(meas)
sim_nn = copy.deepcopy(meas)
sim_idm_nor = copy.deepcopy(meas)
sim_nn_nor = copy.deepcopy(meas)
sim_ll = copy.deepcopy(meas)
sim_ska = copy.deepcopy(meas)

if veh_to_plot[0] in lc_list:
    count = lc_list.index(veh)
    res = res1[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_idm)
    cal.simulate(res['x'])
    sim_idm[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_idm[veh][t1-t0:t2+1-t0,3] = cal.all_vehicles[0].speedmem
    
    res = nnres1[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_nn)
    cal.simulate(res['x'])
    sim_nn[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_nn[veh][t1-t0:t2-t0,3] = cal.all_vehicles[0].speedmem
    sim_nn[veh][t2-t0,3] = sim_nn[veh][t2-t0-1,3]
    
    res = llres1[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_ll)
    cal.simulate(res['x'])
    sim_ll[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_ll[veh][t1-t0:t2-t0,3] = cal.all_vehicles[0].speedmem
    sim_ll[veh][t2-t0,3] = sim_nn[veh][t2-t0-1,3]
    
    res = skares1[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_ska)
    cal.simulate(res['x'])
    sim_ska[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_ska[veh][t1-t0:t2+1-t0,3] = cal.all_vehicles[0].speedmem
    
    res = idmres1[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_idm_nor)
    cal.simulate(res['x'])
    sim_idm_nor[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_idm_nor[veh][t1-t0:t2+1-t0,3] = cal.all_vehicles[0].speedmem
    
    res = nmres1[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_nn_nor)
    cal.simulate(res['x'])
    sim_nn_nor[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_nn_nor[veh][t1-t0:t2-t0,3] = cal.all_vehicles[0].speedmem
    sim_nn_nor[veh][t2-t0,3] = sim_nn_nor[veh][t2-t0-1,3]
else:
    count = nolc_list.index(veh)
    res = res3[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_idm_nor)
    cal.simulate(res['x'])
    sim_idm[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_idm[veh][t1-t0:t2+1-t0,3] = cal.all_vehicles[0].speedmem
    
    res = nnres3[count]
    cal = calibration.make_calibration(veh_to_plot, meas, platooninfo, .1, **kwargs_nn_nor)
    cal.simulate(res['x'])
    sim_nn[veh][t1-t0:t2+1-t0,2] = cal.all_vehicles[0].posmem
    sim_nn[veh][t1-t0:t2-t0,3] = cal.all_vehicles[0].speedmem
    sim_nn[veh][t2-t0,3] = sim_nn[veh][t2-t0-1,3]
    
    print('vehicle has no lane changing')

# hp.plotvhd(meas, sim_idm, platooninfo, veh_to_plot, draw_arrow=True)
# plt.legend([C1[0], C0[0]], ['IDM with relax', 'data'])
# hp.plotvhd(meas, sim_nn, platooninfo, veh_to_plot, draw_arrow=True)
# plt.legend([C1[0], C0[0]], ['NM with relax', 'data'])
    
t = np.linspace(t1, t2, t2+1-t1)
plt.figure()
plt.plot(t, meas[veh][t1-t0:t2-t0+1,3], t, sim_idm_nor[veh][t1-t0:t2-t0+1,3], t, sim_idm[veh][t1-t0:t2-t0+1,3], t, sim_ska[veh][t1-t0:t2-t0+1,3])
plt.legend(['data', 'IDM no relax', 'IDM with relax', 'IDM-SKA'])
plt.title('speed-time for vehicle '+str(int(veh)))
plt.ylabel('speed (ft/s)')
plt.xlabel('time index (.1s)')

plt.figure()
plt.plot(t, meas[veh][t1-t0:t2-t0+1,3], t, sim_nn_nor[veh][t1-t0:t2-t0+1,3], t, sim_nn[veh][t1-t0:t2-t0+1,3], t, sim_ll[veh][t1-t0:t2-t0+1,3])
plt.legend(['data', 'NM no relax', 'NM with relax', 'NM-LL'])
plt.title('speed-time for vehicle '+str(int(veh)))
plt.ylabel('speed (ft/s)')
plt.xlabel('time index (.1s)')

# plt.plot(t, sim_idm[veh][t1-t0:t2-t0+1,3], t, sim_nn[veh][t1-t0:t2-t0+1,3])
# plt.legend(['data', 'LSTM no relax', 'IDM no relax', 'NM no relax'])



