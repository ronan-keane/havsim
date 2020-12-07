"""Works with initial CF+LC model on lane_changing_refactor."""
import tensorflow as tf
import pickle
import numpy as np
import math
from havsim.calibration import deep_learning

class RNNCFModel(tf.keras.Model):
    """Simple RNN based CF model."""

    def __init__(self, maxhd, maxv, mina, maxa, lstm_units=20, dt=.1, l2reg=.02, dropout=.2, **kwargs):
        """Inits RNN based CF model.

        Args:
            maxhd: max headway (for nomalization of inputs)
            maxv: max velocity (for nomalization of inputs)
            mina: minimum acceleration (for nomalization of outputs)
            maxa: maximum acceleration (for nomalization of outputs)
            dt: timestep
            lstm_units: number of LSTM units
            l2reg: constant for l2 regularization which is applied to dense layer kernel and LSTM kernel
            dropout: dropout % applied to LSTM units
        """
        super().__init__()
        # architecture
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units, dropout=dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=l2reg),
                                    recurrent_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.lc_actions = tf.keras.layers.Dense(3)

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        # other constants
        self.dt = dt
        self.lstm_units = lstm_units
        self.l2reg = l2reg
        self.dropout = dropout
        self.num_hidden_layers = 1

    def call(self, leadfol_inputs, init_state, hidden_states, training=False):
        """Updates states for a batch of vehicles.

        Args:
            leadfol_inputs - tensor with shape (nveh, nt, 12), gives position/speed at given timestep for
                all surrounding vehicles. order is (lead, llead, rlead, fol, lfol, rfol)
                for position, then for speed
            init_state -  tensor with shape (nveh, 2) giving the vehicle position and speed at the
                starting timestep.
            hidden_states - tensor of hidden states with shape (num_hidden_layers, 2, nveh, lstm_units)
                Initialized as all zeros for the first timestep.
            training: Whether to run in training or inference mode. Need to pass training=True if training
                with dropout.

        Returns:
            pred_traj: tensor of vehicle positions, shape of (number of vehicles, number of timesteps). Note
                that these are 1 timestep after leadfol_inputs. E.g. if nt = 2 and leadfol_inputs has the lead
                measurements for time 0 and 1. Then cur_state has the vehicle position/speed for time 0, and
                outputs has the vehicle positions for time 1 and 2. curspeed would have the speed for time 2,
                and you can differentiate the outputs to get the speed at time 1 if it's needed.
            pred_lc_action: (number of vehicles, number of timesteps, 3) tensor giving the predicted logits over
                {left lc, stay in lane, right lc} classes for each timestep.
            curspeed: tensor of current vehicle speeds, shape of (number of vehicles, 1)
            hidden_states: last hidden states for LSTM. Tuple of tensors, where each tensor has shape of
                (number of vehicles, number of LSTM units)
        """
        # prepare data for call
        # self.lstm_cell.reset_dropout_mask()
        leadfol_inputs = tf.unstack(leadfol_inputs, axis=1)  # unpacked over time dimension
        cur_pos, cur_speed = tf.unstack(init_state, axis=1)

        pred_traj, pred_lc_action = [], []
        for cur_lead_fol_input in leadfol_inputs:
            # extract data for current timestep
            lead_hd = (cur_lead_fol_input[:,:3] - tf.expand_dims(cur_pos, axis=1))/self.maxhd
            fol_hd = (tf.expand_dims(cur_pos, axis=1) - cur_lead_fol_input[:,3:6])/self.maxhd
            spds = cur_lead_fol_input[:,6:]/self.maxv

            cur_inputs = tf.concat([lead_hd, fol_hd, spds], axis=1)
            cur_inputs = tf.where(tf.math.is_nan(cur_inputs), tf.ones_like(cur_inputs), cur_inputs)

            # call to model
            self.lstm_cell.reset_dropout_mask()
            x, hidden_states = self.lstm_cell(cur_inputs, hidden_states, training)
            x = self.dense2(x)
            cur_lc = self.lc_actions(x)  # logits for LC over {left, stay, right} classes for batch
            x = self.dense1(x)  # current normalized acceleration for the batch

            # update vehicle states
            x = tf.squeeze(x, axis=1)
            cur_acc = (self.maxa-self.mina)*x + self.mina
            cur_pos = cur_pos + self.dt*cur_speed
            cur_speed = cur_speed + self.dt*cur_acc
            pred_traj.append(cur_pos)
            pred_lc_action.append(cur_lc)

        pred_traj = tf.stack(pred_traj, 1)
        pred_lc_action = tf.stack(pred_lc_action, 1)
        return pred_traj, pred_lc_action, cur_speed, hidden_states

    def get_config(self):
        """Return any non-trainable parameters in json format."""
        return {'pos_args': (self.maxhd, self.maxv, self.mina, self.maxa,), \
                'lstm_units': self.lstm_units, 'dt': self.dt, 'l2reg':self.l2reg, 'dropout':self.dropout}

    @classmethod
    def from_config(self, config):
        """Inits self from saved config created by get_config."""
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

# save_model(model, 'CF_LC_initial')

model = load_model('CF_LC_initial')