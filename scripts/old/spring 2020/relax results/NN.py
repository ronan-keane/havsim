
from havsim.calibration import deep_learning
import pickle
import numpy as np
import tensorflow as tf
from havsim import helper
import math

try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

#%% generate training data and initialize model/optimizer

nolc_list = []
for veh in meas.keys():
    temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
np.random.shuffle(nolc_list)
train_veh = nolc_list[:-100]
test_veh = nolc_list[-100:]

training, norm = deep_learning.make_dataset(meas, platooninfo, train_veh)
maxhd, maxv, mina, maxa = norm
testing, unused = deep_learning.make_dataset(meas, platooninfo, test_veh)

model = deep_learning.RNNCFModel(maxhd, maxv, 0, 1, lstm_units=60)
loss = deep_learning.masked_MSE_loss
opt = tf.keras.optimizers.Adam(learning_rate = .0008)

#%% code for training
# the weights are also saved in havsim/scripts/meng/deep learning/saved lstm weights
deep_learning.training_loop(model, loss, opt, training, nbatches = 10000, nveh = 32, nt = 50)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 100)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 200)
deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 300)
deep_learning.training_loop(model, loss, opt, training, nbatches = 2000, nveh = 32, nt = 500)


#%%

def make_into_analyze_res_format(out, vehlist, meas, platooninfo, dt = .1):
    # want dictionary with values as a dict with keys 'posmem' 'speedmem'
    res_list = {}
    for count, veh in enumerate(vehlist):
        t0, t1 = platooninfo[veh][0], platooninfo[veh][1]
        traj_len = platooninfo[veh][2] - platooninfo[veh][1]+1
        res_list[veh] = {'posmem':None, 'speedmem':None}
        temp = list(out[0][count,:traj_len].numpy())  # note that output does not include the initial condition
        res_list[veh]['posmem'] = temp[:-1]
        res_list[veh]['posmem'].insert(0, meas[veh][t1-t0,2])
        res_list[veh]['speedmem'] = [(temp[i+1] - temp[i])/(dt) for i in range(traj_len-1)]
        res_list[veh]['speedmem'].insert(0, meas[veh][t1-t0,3])
    return res_list


def analyze_res_NN(res_list, meas, platooninfo, dt, mergeind = math.inf, times = 100, realistic = 1.1):
    out = {'mse':None, 'mse near LC':None, 'mse for merges':None, 'mse for many LC':None, 'realistic acc':None, 'short lc':None}

    mse = []
    nearlc = []
    merge = []
    manylc = []
    real = []
    out2 = []
    shortlc = []
    for count, item in enumerate(res_list.items()):
        veh, res = item

        leadinfo = helper.makeleadinfo([veh], platooninfo, meas)
        lctimes = [cur[1] for cur in leadinfo[0]]
        rinfo = helper.makerinfo([veh], platooninfo, meas, leadinfo)
        if len(lctimes) > len(rinfo[0]):  # this handles an edge case
            rinfo[0].insert(0, [None, -1])
        t_nstar, t_n, T_nm1 = platooninfo[veh][:3]
        if lctimes[0] == t_nstar:
            lctimes = lctimes[1:]

        curmeas = meas[veh][t_n-t_nstar:T_nm1+1-t_nstar,2]
        cursim = np.array(res['posmem'])
        curmse = np.mean(np.square(curmeas-cursim))/(3.28084**2)

        mse.append(curmse)
        if len(leadinfo[0]) > 3:
            manylc.append(curmse)  # many lc
        if count >= mergeind:
            merge.append(curmse)
        for count, curtime in enumerate(lctimes):
            curmse = np.square(cursim[curtime-t_n:curtime-t_n+times] - curmeas[curtime-t_n:curtime-t_n+times])
            curmse = np.sum(curmse)/len(curmse)/(3.28084**2)
            nearlc.append(curmse)  # near lc
            # try:
            curg = rinfo[0][count][1]
            if curg > 0:
                shortlc.append(curmse)
            # except:
            #     print('hello')

        simacc = [(cursim[i+2] - 2*cursim[i+1] + cursim[i])/(dt**2) for i in range(len(cursim)-2)]
        measacc = [(curmeas[i+2] - 2*curmeas[i+1] + curmeas[i])/(dt**2) for i in range(len(curmeas)-2)]
        upper = max(realistic*max(measacc), 4*3.28)
        lower = min(-6*3.28, realistic*min(measacc))
        temp1, temp2 = max(simacc), min(simacc)
        if temp1 < upper and temp2 > lower:
            real.append(1)
        else:
            real.append(0)
            out2.append(((upper, lower), (temp1, temp2)))

    out['mse'] = (np.mean(mse), np.median(mse), np.std(mse))
    out['mse near LC'] = (np.mean(nearlc), np.median(nearlc), np.std(nearlc))
    out['mse for merges'] = (np.mean(merge), np.median(merge), np.std(merge))
    out['mse for many LC'] = (np.mean(manylc), np.median(manylc), np.std(manylc))
    out['realistic acc'] = (np.mean(real))
    out['short lc'] = (np.mean(shortlc), np.median(shortlc), np.std(shortlc))
    print(out)
    return out, out2

#%%  # analyze results
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
merge_ind = len(lc_list)
lc_list.extend(merge_list)


nolc_ds, unused = deep_learning.make_dataset(meas, platooninfo, nolc_list)
lc_ds, unused = deep_learning.make_dataset(meas, platooninfo, lc_list)
#%%
nolc_res = deep_learning.generate_trajectories(model, list(nolc_ds.keys()), nolc_ds)
nolc_nor_res_list = make_into_analyze_res_format(nolc_res,  list(nolc_ds.keys()), meas, platooninfo)
out1 = analyze_res_NN(nolc_nor_res_list, meas, platooninfo, .1)


lc_nor_res = deep_learning.generate_trajectories(model, list(lc_ds.keys()), lc_ds)
lc_nor_res_list = make_into_analyze_res_format(lc_nor_res, list(lc_ds.keys()), meas, platooninfo)
out2 = analyze_res_NN(lc_nor_res_list, meas, platooninfo, .1, mergeind = merge_ind)

#%% results for applying relaxation to model
nolc_ds, unused = make_dataset(meas, platooninfo, nolc_list)  # need to use the modified model from DL2_relax = trained LSTM with relax
lc_ds, unused = make_dataset(meas, platooninfo, lc_list)

nolc_res = generate_trajectories(model, list(nolc_ds.keys()), nolc_ds)
nolc_res_list = make_into_analyze_res_format(nolc_res,  list(nolc_ds.keys()), meas, platooninfo)
out3 = analyze_res_NN(nolc_r_res_list, meas, platooninfo, .1)

lc_nor_res = generate_trajectories(model, list(lc_ds.keys()), lc_ds)
lc_res_list = make_into_analyze_res_format(lc_nor_res, list(lc_ds.keys()), meas, platooninfo)
out4 = analyze_res_NN(lc_r_res_list, meas, platooninfo, .1, mergeind = merge_ind)


