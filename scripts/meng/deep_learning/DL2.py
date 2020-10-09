# imports and load data
from havsim.calibration import deep_learning
import pickle
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import nni

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
np.random.shuffle(nolc_list)
train_veh = nolc_list[:-300]
test_veh = nolc_list[-300:]

training, norm = deep_learning.make_dataset(meas, platooninfo, train_veh)
maxhd, maxv, mina, maxa = norm
testing, unused = deep_learning.make_dataset(meas, platooninfo, test_veh)

tuned_params = nni.get_next_parameter()

model = deep_learning.RNNCFModel(maxhd, maxv, 0, 1, lstm_units=tuned_params['lstm_units'])
loss = deep_learning.masked_MSE_loss
opt = tf.keras.optimizers.Adam(learning_rate = .0008)

#%% train and save results
early_stopping = False

def test_loss():
    return deep_learning.generate_trajectories(model, list(testing.keys()), testing, loss=deep_learning.weighted_masked_MSE_loss)[-1]

def train_loss():
    return deep_learning.generate_trajectories(model, list(training.keys()), training, loss=deep_learning.weighted_masked_MSE_loss)[-1]
# no early stopping -
if not early_stopping:
    epochs = 5
    batches = [10000, 1000, 1000, 1000, 2000]
    timesteps = [50, 100, 200, 300, 500]
    veh = 32
    train_losses = []
    test_losses = []
    for i in range(epochs):
        deep_learning.training_loop(model, loss, opt, training, nbatches=batches[i], nveh=veh, nt=timesteps[i])
        train_losses.append(train_loss())
        test_losses.append(test_loss())
        nni.report_intermediate_result(test_losses[-1])
    plt.figure(1)
    plt.plot(list(range(epochs)), train_losses, 'b-', test_losses, 'r-')
    plt.title('Training vs Validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()
    nni.report_final_result(test_losses[-1])
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


print(' testing loss was '+str(test_loss()))
print(' training loss was '+str(train_loss()))




