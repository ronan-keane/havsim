# imports and load data
from havsim.calibration import deep_learning
import pickle
import numpy as np
import tensorflow as tf
import math

try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/home/rlk268/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

#%% generate training data and initialize model/optimizer

nolc_list = []
# train on no lc vehicles only  # good
for veh in meas.keys():
    temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
# train on all vehicles  # bad
# for veh in meas.keys():
#     temp = nolc_list.append(veh) if len(platooninfo[veh][4]) > 0 else None
np.random.shuffle(nolc_list)
train_veh = nolc_list[:-300]
test_veh = nolc_list[-300:]

training, norm = deep_learning.make_dataset(meas, platooninfo, train_veh)
maxhd, maxv, mina, maxa = norm
testing, unused = deep_learning.make_dataset(meas, platooninfo, test_veh)

model = deep_learning.RNNCFModel(maxhd, maxv, 0, 1, lstm_units=60)
loss = deep_learning.masked_MSE_loss
opt = tf.keras.optimizers.Adam(learning_rate = .0008)

#%% train and save results
early_stopping = False

# no early stopping -
if not early_stopping:
    deep_learning.training_loop(model, loss, opt, training, nbatches = 10000, nveh = 32, nt = 50)
    deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 100)
    deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 200)
    deep_learning.training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 300)
    deep_learning.training_loop(model, loss, opt, training, nbatches = 2000, nveh = 32, nt = 500)

# early stopping -
if early_stopping:
    def early_stopping_loss(model):
        return deep_learning.generate_trajectories(model, list(testing.keys()), testing,
                                                    loss=deep_learning.weighted_masked_MSE_loss)[-1]
    deep_learning.training_loop(model, loss, opt, training, nbatches=10000, nveh=32, nt=50, m=100, n=20,
                                early_stopping_loss=early_stopping_loss)
    deep_learning.training_loop(model, loss, opt, training, nbatches=1000, nveh=32, nt=100, m=50, n=10,
                                early_stopping_loss=early_stopping_loss)
    deep_learning.training_loop(model, loss, opt, training, nbatches=1000, nveh=32, nt=200, m=40, n=10,
                                early_stopping_loss=early_stopping_loss)
    deep_learning.training_loop(model, loss, opt, training, nbatches=1000, nveh=32, nt=300, m=30, n=10,
                                early_stopping_loss=early_stopping_loss)
    deep_learning.training_loop(model, loss, opt, training, nbatches=2000, nveh=32, nt=500, m=20, n=10,
                                early_stopping_loss=early_stopping_loss)



# model.save_weights('trained LSTM no relax')

# model.load_weights('trained LSTM')

#%% test by generating entire trajectories
test = deep_learning.generate_trajectories(model, list(testing.keys()), testing, loss=deep_learning.weighted_masked_MSE_loss)
test2 = deep_learning.generate_trajectories(model, list(training.keys()), training, loss=deep_learning.weighted_masked_MSE_loss)

print(' testing loss was '+str(test[-1]))
print(' training loss was '+str(test2[-1]))




