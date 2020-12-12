# imports and load data
from havsim.calibration import deep_learning
import pickle
import tensorflow as tf
import havsim.plotting as hp
import matplotlib.pyplot as plt
import numpy as np
#%%
# with open('recon-ngsim.pkl', 'rb') as f:
#     vehdict = pickle.load(f)
# with open('data/recon-ngsim.pkl', 'rb') as f:
#     vehdict = pickle.load(f)
with open('C:/Users/rlk268/OneDrive - Cornell University/havsim/data/recon-ngsim.pkl', 'rb') as f:
    vehdict = pickle.load(f)

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

#%%  decide what vehicles to train on

veh_list = []
for veh in vehdict.keys():
    if len(vehdict[veh].leads)>=1:  # exactly 1 leader => no lane changing. >1 => at least 1 lane change.
        start, end = vehdict[veh].longest_lead_times
        if end - start > 50:
            veh_list.append(veh)

train_veh = veh_list[:-300]
test_veh = veh_list[-300:]

training, norm = deep_learning.make_dataset(vehdict, train_veh, window=15)
testing, unused = deep_learning.make_dataset(vehdict, test_veh, window=15)
#%% initialize model and optimizer
def no_lc_loss(*args):  # may want to comment out the LC output in RNNCFModel so tensorflow does not spam print out
    return 0.0

params = {"lstm_units" : 64,
    "learning_rate": 0.008,
    "dropout": 0.2,
    "regularizer": 0.02,
    "batch_size": 32}

model = deep_learning.RNNSeparateModel(*norm, **params)
# model = deep_learning.RNNSeparateModel(*norm, **params)

loss = deep_learning.masked_MSE_loss
lc_loss = deep_learning.SparseCategoricalCrossentropy
# lc_loss = deep_learning.expected_LC

# lc_loss = no_lc_loss  # train CF model only
opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

#%%  training
deep_learning.training_loop(model, loss, lc_loss, opt, training, 10000, nt=50, m=100)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=100, m=100)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=200, m=100)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=300, m=100)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 2000, nt=500, m=100)


#%% generate simulated trajectories
sim_vehdict = deep_learning.generate_vehicle_data(model, train_veh, training, vehdict, deep_learning.weighted_masked_MSE_loss, lc_loss)

sim_vehdict_testing = deep_learning.generate_vehicle_data(model, test_veh, testing, vehdict, deep_learning.weighted_masked_MSE_loss, lc_loss)

for i in sim_vehdict_testing.keys():
    sim_vehdict[i] = sim_vehdict_testing[i]

# hp.plotCFErrorN(sim_vehdict, vehdict, 'outputs/cferror/')
# hp.plotTrajectoriesProb(sim_vehdict, 'outputs/lcprobs/')

# hp.plot_confmat(sim_vehdict, save=True, output_dir = 'outputs/')
#%% examples of plotting

vehid = 390
hp.plotTrajectoryProbs(sim_vehdict,vehid)

def cferror(vehid):
    temp = vehdict[vehid].longest_lead_times
    return np.abs(np.array(vehdict[vehid].posmem[temp[0]:temp[1]]) - sim_vehdict[vehid].posmem[temp[0]:temp[1]])
hp.plotCFError(cferror(vehid), vehid)

plt.figure()
temp = vehdict[vehid].longest_lead_times
plt.plot(vehdict[vehid].posmem[temp[0]:temp[1]])
plt.plot(sim_vehdict[vehid].posmem[temp[0]:temp[1]])
lctimes = [i[1] for i in vehdict[vehid].lanemem.intervals(*vehdict[vehid].longest_lead_times)[1:]]
plt.plot(np.array(lctimes)-temp[0], [vehdict[vehid].posmem[i] for i in lctimes], '*b')
plt.legend(['true', 'sim', 'lc time'])
plt.show()
