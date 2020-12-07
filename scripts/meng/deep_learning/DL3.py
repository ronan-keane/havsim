# imports and load data
from havsim.calibration import deep_learning
import pickle
import tensorflow as tf
import havsim.plotting as hp

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
    if len(vehdict[veh].leads)==1:  # exactly 1 leader => no lane changing. >1 => at least 1 lane change.
        veh_list.append(veh)

train_veh = veh_list[:-100]
test_veh = veh_list[-100:]

training, norm = deep_learning.make_dataset(vehdict, train_veh)
testing, unused = deep_learning.make_dataset(vehdict, test_veh)
#%% initialize model and optimizer
def no_lc_loss(*args):
    return 0

params = {"lstm_units" : 64,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "regularizer": 0.02,
    "batch_size": 32}

model = deep_learning.RNNCFModel(*norm, **params)
loss = deep_learning.weighted_masked_MSE_loss
# lc_loss = deep_learning.SparseCategoricalCrossentropy
lc_loss = no_lc_loss  # train CF model only
opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

#%%
deep_learning.training_loop(model, loss, lc_loss, opt, training, 10000, nt=50)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=100)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=300)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=500)
deep_learning.training_loop(model, loss, lc_loss, opt, training, 1000, nt=750)

#%%
sim_vehdict = deep_learning.generate_vehicle_data(model, train_veh, training, vehdict, loss)