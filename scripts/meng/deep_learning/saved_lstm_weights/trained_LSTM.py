
"""
@author: rlk268@cornell.edu
"""
import tensorflow as tf
import pickle
import numpy as np
import math
import deep_learning

class RNNCFModel(tf.keras.Model):
    """Simple RNN based CF model."""

    def __init__(self, maxhd, maxv, mina, maxa, lstm_units=20, dt=.1):
        """Inits RNN based CF model.

        Args:
            maxhd: max headway (for nomalization of inputs)
            maxv: max velocity (for nomalization of inputs)
            mina: minimum acceleration (for nomalization of outputs)
            maxa: maximum acceleration (for nomalization of outputs)
            dt: timestep
        """
        super().__init__()
        # architecture
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units, dropout=.3,
                                                  kernel_regularizer=tf.keras.regularizers.l2(l=.02),
                                                  recurrent_regularizer=tf.keras.regularizers.l2(l=.02))
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=.02))

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        # other constants
        self.dt = dt
        self.lstm_units = lstm_units

    def call(self, inputs, training=False):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, cur_state, hidden_states.
                lead_inputs - tensor with shape (nveh, nt, 2), giving the leader position and speed at
                    each timestep.
                cur_state -  tensor with shape (nveh, 2) giving the vehicle position and speed at the
                    starting timestep.
                hidden_states - list of the two hidden states, each hidden state is a tensor with shape
                    of (nveh, lstm_units). Initialized as all zeros for the first timestep.
            training: Whether to run in training or inference mode. Need to pass training=True if training
                with dropout.

        Returns:
            outputs: tensor of vehicle positions, shape of (number of vehicles, number of timesteps). Note
                that these are 1 timestep after lead_inputs. E.g. if nt = 2 and lead_inputs has the lead
                measurements for time 0 and 1. Then cur_state has the vehicle position/speed for time 0, and
                outputs has the vehicle positions for time 1 and 2. curspeed would have the speed for time 2,
                and you can differentiate the outputs to get the speed at time 1 if it's needed.
            curspeed: tensor of current vehicle speeds, shape of (number of vehicles, 1)
            hidden_states: last hidden states for LSTM. Tuple of tensors, where each tensor has shape of
                (number of vehicles, number of LSTM units)
        """
        # prepare data for call
        lead_inputs, init_state, hidden_states = inputs
        lead_inputs = tf.unstack(lead_inputs, axis=1)  # unpacked over time dimension
        cur_pos, cur_speed = tf.unstack(init_state, axis=1)
        outputs = []
        for cur_lead_input in lead_inputs:
            # normalize data for current timestep
            cur_lead_pos, cur_lead_speed = tf.unstack(cur_lead_input, axis=1)
            curhd = cur_lead_pos-cur_pos
            curhd = curhd/self.maxhd
            cur_lead_speed = cur_lead_speed/self.maxv
            norm_veh_speed = cur_speed/self.maxv
            cur_inputs = tf.stack([curhd, norm_veh_speed, cur_lead_speed], axis=1)

            # call to model
            self.lstm_cell.reset_dropout_mask()
            x, hidden_states = self.lstm_cell(cur_inputs, hidden_states, training)
            x = self.dense2(x)
            x = self.dense1(x)  # output of the model is current acceleration for the batch

            # update vehicle states
            x = tf.squeeze(x, axis=1)
            cur_acc = (self.maxa-self.mina)*x + self.mina
            cur_pos = cur_pos + self.dt*cur_speed
            cur_speed = cur_speed + self.dt*cur_acc
            outputs.append(cur_pos)

        outputs = tf.stack(outputs, 1)
        return outputs, cur_speed, hidden_states

    def get_config(self):
        return {'pos_args': (self.maxhd, self.maxv, self.mina, self.maxa,), 'lstm_units': self.lstm_units, 'dt': self.dt}

    @classmethod
    def from_config(self, config):
        pos_args = config.pop('pos_args')
        return self(*pos_args, **config)

def save_model(model, filepath):
    config_filepath = filepath+' config.pkl'
    with open(config_filepath, 'wb') as f:
        pickle.dump(model.get_config(), f)

    model.save_weights(filepath)

def load_model(filepath):
    config_filepath = filepath+' config.pkl'
    with open(config_filepath, 'rb') as f:
        config = pickle.load(f)
    loaded_model = RNNCFModel.from_config(config)
    loaded_model.load_weights(filepath)
    return loaded_model
#%%
# save_model(model, 'trained LSTM')
#%%
model = load_model('trained LSTM')

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

nolc_list = []
# # train on no lc vehicles only
for veh in meas.keys():
    temp = nolc_list.append(veh) if len(platooninfo[veh][4]) == 1 else None
# train on all vehicles
# for veh in meas.keys():
#     temp = nolc_list.append(veh) if len(platooninfo[veh][4]) > 0 else None

# platooninfo[veh][1:3] first and last time step
np.random.shuffle(nolc_list)
train_veh = nolc_list[:-200]
val_veh = nolc_list[-200:-100]
test_veh = nolc_list[-100:]

training, norm = deep_learning.make_dataset(meas, platooninfo, train_veh)
maxhd, maxv, mina, maxa = norm
validation, unused = deep_learning.make_dataset(meas, platooninfo, val_veh)
testing, unused = deep_learning.make_dataset(meas, platooninfo, test_veh)

def test_loss():
    return deep_learning.generate_trajectories(model, list(testing.keys()), testing, loss=deep_learning.weighted_masked_MSE_loss)[-1]

def valid_loss():
    return deep_learning.generate_trajectories(model, list(validation.keys()), validation, loss=deep_learning.weighted_masked_MSE_loss)[-1]

def train_loss():
    return deep_learning.generate_trajectories(model, list(training.keys()), training, loss=deep_learning.weighted_masked_MSE_loss)[-1]

print(test_loss())
print(valid_loss())
print(train_loss())
