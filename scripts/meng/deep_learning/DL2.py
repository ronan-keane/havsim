# imports and load data
from havsim.calibration import deep_learning
import pickle
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import nni
import dl_model
import random
from tensorflow.python.profiler import profiler_v2 as profiler

try:
    with open('/Users/nkluke/Documents/Cornell/CS5999/havsim/data/recon-ngsim.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f) #load data
except:
    with open('/Users/nkluke/Documents/Cornell/CS5999/havsim/data/recon-ngsim.pkl', 'rb') as f:
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
# # train on no lc vehicles only
for veh in meas.keys():
    temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
# train on all vehicles
# for veh in meas.keys():
#     temp = nolc_list.append(veh) if len(platooninfo[veh][4]) > 0 else None

# platooninfo[veh][1:3] first and last time step
# np.random.shuffle(nolc_list)
random.Random(2020).shuffle(nolc_list)
train_veh = nolc_list[:-100]
val_veh = nolc_list[-100:]
test_veh = []

training, norm = deep_learning.make_dataset(meas, platooninfo, train_veh)
maxhd, maxv, mina, maxa = norm
validation, unused = deep_learning.make_dataset(meas, platooninfo, val_veh)
testing, unused = deep_learning.make_dataset(meas, platooninfo, test_veh)

default_params = {
    "lstm_units" : 64,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "regularizer": 0.02,
    "batch_size": 32
}

tuned_params = nni.get_next_parameter()

params = tuned_params if tuned_params else default_params

old_model = False
if old_model:
    model = deep_learning.RNNCFModel(maxhd, maxv, 0, 1, lstm_units=params['lstm_units'], params=params)
    loss = deep_learning.masked_MSE_loss
else:
    model = dl_model.RNNCFModel(maxhd, maxv, 0, 1, lstm_units=128, past=15, params=params)
    loss = dl_model.masked_MSE_loss
opt = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])



#%% train and save results
early_stopping = False

def test_loss():
    if old_model:
        return deep_learning.generate_trajectories(model, list(testing.keys()), testing, loss=deep_learning.weighted_masked_MSE_loss)[-1]
    else:
        return dl_model.generate_trajectories(model, list(testing.keys()), testing, loss=deep_learning.weighted_masked_MSE_loss)[-1]

def valid_loss():
    if old_model:
        return deep_learning.generate_trajectories(model, list(validation.keys()), validation, loss=deep_learning.weighted_masked_MSE_loss)[-1]
    else:
        return dl_model.generate_trajectories(model, list(validation.keys()), validation, loss=deep_learning.weighted_masked_MSE_loss)[-1]

def train_loss():
    if old_model:
        return deep_learning.generate_trajectories(model, list(training.keys()), training, loss=deep_learning.weighted_masked_MSE_loss)[-1]
    else:
        return dl_model.generate_trajectories(model, list(training.keys()), training, loss=deep_learning.weighted_masked_MSE_loss)[-1]


if old_model:
    # no early stopping -
    if not early_stopping:
        epochs = 1
        batches = [10000, 1000, 1000, 1000, 1000, 5000]
        timesteps = [50, 100, 200, 300, 500, 750]   #go up to 750
        veh = params['batch_size']
        train_losses = []
        valid_losses = []
        for i in range(epochs):
            nbatches = batches[i]
            steps = 500
            for j in range(nbatches//steps):
                deep_learning.training_loop(model, loss, opt, training, nbatches=steps, nveh=veh, nt=timesteps[i])
                valid_loss_val = valid_loss().numpy()
                train_loss_val = train_loss().numpy()
                train_losses.append(train_loss_val)
                valid_losses.append(valid_loss_val)
                nni.report_intermediate_result(valid_losses[-1])
        # plt.figure(1)
        # plt.plot(list(range(epochs)), train_losses, 'b-', valid_losses, 'r-')
        # plt.title('Training vs Validation loss')
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.legend(['training', 'validation'], loc='upper right')
        # plt.show()
        print('train loss', *train_losses)
        print('validation_loss', *valid_losses)
        nni.report_final_result(valid_losses[-1])
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

else:
    dl_model.training_loop(model, loss, opt, training, epochs=1, nveh=32, nt=25)
    # profiler.warmup()
    # profiler.start(logdir='logs')
    # dl_model.training_loop(model, loss, opt, training, epochs=1, nveh=32, nt=25, past=10)
    # print('val loss', valid_loss())
    # profiler.stop()
    # dl_model.training_loop(model, loss, opt, training, epochs=5, nveh=32, nt=50, past=15)
    # print('val loss', valid_loss())
    # dl_model.training_loop(model, loss, opt, training, epochs=2, nveh=32, nt=100, past=15)
    # print('val loss', valid_loss())
    # dl_model.training_loop(model, loss, opt, training, epochs=2, nveh=32, nt=200, past=15)
    # print('val loss', valid_loss())
    # dl_model.training_loop(model, loss, opt, training, epochs=2, nveh=32, nt=400, past=15)
    # print('val loss', valid_loss())
    # dl_model.training_loop(model, loss, opt, training, epochs=5, nveh=32, nt=800, past=15)
    # print('val loss', valid_loss())


# model.save_weights('trained LSTM no relax')

# model.load_weights('trained LSTM')

#%% test by generating entire trajectories


# print(' validation loss was '+str(valid_loss()))
# print(' training loss was '+str(train_loss()))




