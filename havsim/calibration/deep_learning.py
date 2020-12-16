"""Tensorflow.keras car following and lane changing models."""

import pickle
import tensorflow as tf
import numpy as np
from havsim import helper
import copy

# TODO can't use expected_LC loss because train_step needs to have tensor inputs instad of vehs_counter, ds
# TODO try training LC model only with expected_LC loss (make new versions of RNNCFModel, train_step, training_loop in DL3_lc_only to do this (feed true_traj into model))
# TODO ET tends to be quite low and P also tends to be low for intervals with no lane changing.
    # might want to try giving the LC model seperate layers


def generate_lane_data(veh_data, window=1):
    """
    Generates Labels for Lane-Changing Model for a given vehicle
    Args:
        veh_data: helper.VehicleData for a single vehicle
    Returns:
        lane_data: python list of 0/1/2, 0 is lane changing to the left, 1 is
            staying in the same lane, and 2 is lane changing to the right
            length = veh_data.end - veh_data.start + 1
    """
    lane_data = []

    intervals = veh_data.lanemem.intervals()
    for idx, (val, start, end) in list(enumerate(intervals)):
        lane_data += [1] * max((end - start - window),0)
        leftover = min(window, end-start)
        if idx < len(intervals) - 1:
            if val < intervals[idx + 1][0]:
                lane_data.extend([2]*leftover)
            elif val > intervals[idx + 1][0]:
                lane_data.extend([0]*leftover)
            else:
                lane_data.extend([1]*leftover)
        else:
            lane_data.extend([1]*leftover)
    return lane_data

def make_dataset(veh_dict, veh_list, dt=.1, window=1):
    """Makes dataset from meas and platooninfo.

    Args:
        veh_dict: dictionary with keys as vehicle ID, value as helper.VehicleData.
            see havsim.helper.load_dataset
        dt: timestep
    Returns:
        ds: (reads as dataset) dictionary of vehicles, values are a dictionary with keys
            'IC' - (initial conditions) list of starting position/speed for vehicle
            'times' - list of two int times. First is the first time with an observed leader. Second is the
                last time with an observed leader +1. The number of times we call the model is equal to
                times[1] - times[0], which is the length of the lead measurements.
            'longest lead times' - these
            'veh posmem' - (1d,) numpy array of observed positions for vehicle, 0 index corresponds
                to times[0]. Typically this has a longer length than the lead posmem/speedmem.
            'veh speedmem' - (1d,) numpy array of observed speeds for vehicle
            'veh lanemem' - veh_data.lanemem (helper.VehMem for lane data of veh)
            'true lc actions' - (1d,) numpy array of labels for lane changing actions. 0 is left change at
                corresponding timestep, 2 is right change, 1 is no change.
            'viable lc' - (t, 3) row index corresponds to time, column index corresponds to lc class,
                (0 = left, 2 = right). Value is 1 if that lc type is possible at the timestep, 0 otherwise.
            'lead posmem' - (1d,) numpy array of positions for leaders, corresponding to times.
                length is subtracted from the lead position.
            'lead speedmem' - (1d,) numpy array of speeds for leaders.
            'fol posmem' - (1d,) numpy array of positions of followers, with veh length subtracted
            'fol speedmem' - (1d,) numpy array of speed of followers.
            'lfol posmem' - (1d,) numpy array of positions for lfol
            'lfol speedmem' - (1d,) numpy array of speeds for lfol
            'rfol posmem' - (1d,) numpy array of positions for rfol
            'rfol speedmem' - (1d,) numpy array of speeds for rfol
            'llead posmem' - (1d,) same as lead posmem, but for left leader
            'llead speedmem' - (1d,) same as lead speedmem, but for left leader
            'rlead posmem' - (1d,) same as lead posmem, but for right leader
            'rlead speedmem' - (1d,) same as lead speedmem, but for right leader
        normalization amounts: tuple of
            maxheadway: max headway observed in training set
            maxspeed: max velocity observed in training set
            minacc: minimum acceleration observed in training set
            maxacc: maximum acceleration observed in training set

    """
    ds = {}
    maxheadway, maxspeed = 0, 0
    minacc, maxacc = 1e4, -1e4

    for veh in veh_list:
        veh_data = veh_dict[veh]
        start, end = int(veh_data.start), int(veh_data.end)
        start_sim, end_sim = veh_data.longest_lead_times

        # vehicle
        vehpos = np.array(veh_data.posmem[start_sim:end_sim+2])
        vehspd = np.array(veh_data.speedmem[start_sim:end_sim+2])
        # true_lc_actions = np.array(generate_lane_data(veh_data, window)[start_sim - start:end_sim+2])
        true_lc_actions = np.array(generate_lane_data(veh_data, window)[start_sim - start:end_sim - start + 2])

        # generate viable_lc - whether or not left/right change is possible at any given timestep
        contains_lane1 = 1.0 in veh_data.get_unique_mem(veh_data.lanemem)
        viable_lc = np.ones((vehpos.shape[0], 3))
        for time in range(start_sim, int(start_sim+vehpos.shape[0])):
            if not (veh_data.lanemem[time] > 2 or (veh_data.lanemem[time] == 2 and contains_lane1)):
                viable_lc[time - start_sim, 0] = 0
            if veh_data.lanemem[time] > 5:
                viable_lc[time - start_sim, 2] = 0

        # leaders
        leadpos = np.array(veh_data.leadmem.pos[start_sim:end_sim + 1]) - \
            np.array(veh_data.leadmem.len[start_sim:end_sim + 1])
        leadspeed = np.array(veh_data.leadmem.speed[start_sim:end_sim + 1])

        lleadpos = np.array(veh_data.lleadmem.pos.index(start_sim, end_sim + 1, np.nan)) - \
            np.array(veh_data.lleadmem.len.index(start_sim, end_sim + 1, np.nan))
        lleadspeed = np.array(veh_data.lleadmem.speed.index(start_sim, end_sim + 1, np.nan))

        rleadpos = np.array(veh_data.rleadmem.pos.index(start_sim, end_sim + 1, np.nan)) - \
            np.array(veh_data.rleadmem.len.index(start_sim, end_sim + 1, np.nan))
        rleadspeed = np.array(veh_data.rleadmem.speed.index(start_sim,end_sim + 1, np.nan))

        # followers
        folpos = np.array(veh_data.folmem.pos.index(start_sim, end_sim + 1, np.nan)) + veh_data.len
        folspeed = np.array(veh_data.folmem.speed.index(start_sim, end_sim + 1, np.nan))

        lfolpos = np.array(veh_data.lfolmem.pos.index(start_sim, end_sim + 1, np.nan)) + veh_data.len
        lfolspeed = np.array(veh_data.lfolmem.speed.index(start_sim, end_sim + 1, np.nan))

        rfolpos = np.array(veh_data.rfolmem.pos.index(start_sim, end_sim + 1, np.nan)) + veh_data.len
        rfolspeed = np.array(veh_data.rfolmem.speed.index(start_sim, end_sim + 1, np.nan))

        # normalization
        IC = [vehpos[0], vehspd[0]]
        if end_sim != start_sim:
            headway = leadpos - vehpos[:end_sim + 1 - start_sim]
            maxheadway = max(max(headway), maxheadway)
            maxspeed = max(max(vehspd), maxspeed)
        else:
            pass  # edge case of vehicle can not be simulated
        vehacc = [(vehpos[i+2] - 2*vehpos[i+1] + vehpos[i])/(dt**2) for i in range(len(vehpos)-2)]
        minacc, maxacc = min(minacc, min(vehacc)), max(maxacc, max(vehacc))

        ds[veh] = {'IC': IC, 'times': [start_sim, min(int(end_sim + 1), end)], 'veh posmem': vehpos,
                'veh speedmem': vehspd, 'true lc actions': true_lc_actions, 'viable lc': np.array(viable_lc),
                'veh lanemem': veh_data.lanemem,
                'lead posmem': leadpos, 'lead speedmem': leadspeed,
                'lfol posmem': lfolpos,'lfol speedmem': lfolspeed,
                'rfol posmem': rfolpos, 'rfol speedmem': rfolspeed,
                'llead posmem': lleadpos, 'llead speedmem': lleadspeed,
                'rlead posmem': rleadpos, 'rlead speedmem': rleadspeed,
                'fol posmem': folpos, 'fol speedmem': folspeed}

    return ds, (maxheadway, maxspeed, minacc, maxacc)

class LCOnlyModel(tf.keras.Model):
    num_lstms = 1
    lc_only = True

    def __init__(self, maxhd, maxv, mina, maxa, lstm_units=20, dt=.1, l2reg=.02, dropout=.2, **kwargs):
        """Inits RNN based LC model.

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
        self.lstm_units = lstm_units
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units, dropout=dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=l2reg),
                                    recurrent_regularizer=tf.keras.regularizers.l2(l=l2reg))

        self.dense = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.lc_actions = tf.keras.layers.Dense(3)

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        self.dt = dt

    def compute_output(self, cur_inputs, hidden_states, training=False):
        """
        Computes output of model given the cur inputs and the hidden states of the lstm_cells.
        Args:
            cur_inputs: tensor with (nveh, nt, 12), gives normalized headways in order of (lead, llead,
                rlead, fol, lfol, rfol)
            hidden_states: list of hidden_states, each a tensor with shape (2, nveh, num_hidden). The number of
                hidden_states depends on the implementation of the model
            training: Whether to run in training/inference mode. Need to pass training=True if training
                with dropout
        Returns:
            cur_cf: the predicted normalized acceleration (this model returns None)
            cur_lc: the predicted log-probabilities of lane changing
            updated hidden_states: in same structure as input
        """
        self.lstm_cell.reset_dropout_mask()
        cur_lc, hidden_states = self.lstm_cell(cur_inputs, hidden_states[0], training)
        cur_lc = self.dense(cur_lc)
        cur_lc = self.lc_actions(cur_lc)
        return None, cur_lc, [hidden_states]

    def call(self, leadfol_inputs, true_traj, hidden_states, training=False):
        """Updates states for a batch of vehicles.

        Args:
            leadfol_inputs - tensor with shape (nveh, nt, 12), gives position/speed at given timestep for
                all surrounding vehicles. order is (lead, llead, rlead, fol, lfol, rfol)
                for position, then for speed
            true_traj -  tensor with shape (nveh, nt) giving the vehicle position at each timestep
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
        true_traj = tf.unstack(true_traj, axis=1)
        leadfol_inputs = tf.unstack(leadfol_inputs, axis=1)  # unpacked over time dimension

        pred_lc_action = []
        for cur_pos, cur_lead_fol_input in zip(true_traj, leadfol_inputs):
            # extract data for current timestep
            lead_hd = (cur_lead_fol_input[:,:3] - tf.expand_dims(cur_pos, axis=1))/self.maxhd
            fol_hd = (tf.expand_dims(cur_pos, axis=1) - cur_lead_fol_input[:,3:6])/self.maxhd
            spds = cur_lead_fol_input[:,6:]/self.maxv

            cur_inputs = tf.concat([lead_hd, fol_hd, spds], axis=1)
            cur_inputs = tf.where(tf.math.is_nan(cur_inputs), tf.ones_like(cur_inputs), cur_inputs)

            _, cur_lc, hidden_states = self.compute_output(cur_inputs, hidden_states, training)

            # update vehicle states
            pred_lc_action.append(cur_lc)

        pred_lc_action = tf.stack(pred_lc_action, 1)
        return tf.stack(true_traj, axis=1), pred_lc_action, None, hidden_states

    def get_config(self):
        """Return any non-trainable parameters in json format."""
        return {'pos_args': (self.maxhd, self.maxv, self.mina, self.maxa,), \
                'lstm_units': self.lstm_units, 'dt': self.dt, 'l2reg':self.l2reg, 'dropout':self.dropout}

    @classmethod
    def from_config(self, config):
        """Inits self from saved config created by get_config."""
        pos_args = config.pop('pos_args')
        return self(*pos_args, **config)

    @classmethod
    def load_model(self, filepath):
        config_filepath = filepath + ' config.pkl'
        with open(config_filepath, 'rb') as f:
            config = pickle.load(f)
        loaded_model = self.from_config(config)
        loaded_model.load_weights(filepath)
        return loaded_model

    def save_model(self, filepath):
        config_filepath = filepath + ' config.pkl'
        with open(config_filepath, 'wb') as f:
            pickle.dump(model.get_config(), f)
        self.save_weights(filepath)

class RNNBaseModel(tf.keras.Model):
    num_lstms = None
    lc_only = False
    def __init__(self, maxhd, maxv, mina, maxa, dt=.1):
        super().__init__()
        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        self.dt = dt

    def compute_output(self, cur_inputs, hidden_states, training=False):
        """
        Computes output of model given the cur inputs and the hidden states of the lstm_cells.
        Args:
            cur_inputs: tensor with (nveh, nt, 12), gives normalized headways in order of (lead, llead,
                rlead, fol, lfol, rfol)
            hidden_states: list of hidden_states, each a tensor with shape (2, nveh, num_hidden). The number of
                hidden_states depends on the implementation of the model
            training: Whether to run in training/inference mode. Need to pass training=True if training
                with dropout
        Returns:
            cur_cf: the predicted normalized acceleration
            cur_lc: the predicted log-probabilities of lane changing
            updated hidden_states: in same structure as input
        """
        raise NotImplementedError

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

            cur_cf, cur_lc, hidden_states = self.compute_output(cur_inputs, hidden_states, training)

            # update vehicle states
            cur_cf = tf.squeeze(cur_cf, axis=1)
            cur_acc = (self.maxa-self.mina)*cur_cf + self.mina
            cur_pos = cur_pos + self.dt*cur_speed
            cur_speed = cur_speed + self.dt*cur_acc
            pred_traj.append(cur_pos)
            pred_lc_action.append(cur_lc)

        pred_traj = tf.stack(pred_traj, 1)
        pred_lc_action = tf.stack(pred_lc_action, 1)

        return pred_traj, pred_lc_action, cur_speed, hidden_states

    def get_config(self):
        """Return any non-trainable parameters in json format."""
        raise NotImplementedError

    @classmethod
    def from_config(self, config):
        """Inits self from saved config created by get_config."""
        raise NotImplementedError

    @classmethod
    def load_model(self, filepath):
        config_filepath = filepath + ' config.pkl'
        with open(config_filepath, 'rb') as f:
            config = pickle.load(f)
        loaded_model = self.from_config(config)
        loaded_model.load_weights(filepath)
        return loaded_model

    def save_model(self, filepath):
        config_filepath = filepath + ' config.pkl'
        with open(config_filepath, 'wb') as f:
            pickle.dump(model.get_config(), f)
        self.save_weights(filepath)

class RNNSeparateModel(RNNBaseModel):
    num_lstms = 2

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
        super().__init__(maxhd, maxv, mina, maxa, dt=dt)
        # architecture
        self.cf_lstm = tf.keras.layers.LSTMCell(lstm_units, dropout=dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=l2reg),
                                    recurrent_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.lc_lstm = tf.keras.layers.LSTMCell(lstm_units, dropout=dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=l2reg),
                                    recurrent_regularizer=tf.keras.regularizers.l2(l=l2reg))

        self.lc_dense = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.cf_dense = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=l2reg))

        self.cf_output = tf.keras.layers.Dense(1)
        self.lc_output = tf.keras.layers.Dense(3)

        # other constants
        self.lstm_units = lstm_units
        self.l2reg = l2reg
        self.dropout = dropout
        self.num_hidden_layers = 1

    def compute_output(self, cur_inputs, hidden_states, training=False):
        """
        Computes output of model given the cur inputs and the hidden states of the lstm_cells.
        Args:
            cur_inputs: tensor with (nveh, nt, 12), gives normalized headways in order of (lead, llead,
                rlead, fol, lfol, rfol)
            hidden_states: list of hidden_states, each a tensor with shape (2, nveh, num_hidden). The number of
                hidden_states depends on the implementation of the model
            training: Whether to run in training/inference mode. Need to pass training=True if training
                with dropout
        Returns:
            cur_cf: the predicted normalized acceleration
            cur_lc: the predicted log-probabilities of lane changing
            updated hidden_states: in same structure as input
        """
        self.cf_lstm.reset_dropout_mask()
        cur_cf, cf_hidden_states = self.cf_lstm(cur_inputs, hidden_states[0], training)
        cur_cf = self.cf_dense(cur_cf)
        cur_cf = self.cf_output(cur_cf)

        self.lc_lstm.reset_dropout_mask()
        cur_lc, lc_hidden_states = self.lc_lstm(cur_inputs, hidden_states[1], training)
        cur_lc = self.lc_dense(cur_lc)
        cur_lc = self.lc_output(cur_lc) # logits for LC over {left, stay, right} classes for batch
        return cur_cf, cur_lc, [cf_hidden_states, lc_hidden_states]

    def get_config(self):
        """Return any non-trainable parameters in json format."""
        return {'pos_args': (self.maxhd, self.maxv, self.mina, self.maxa,), \
                'lstm_units': self.lstm_units, 'dt': self.dt, 'l2reg':self.l2reg, 'dropout':self.dropout}

    @classmethod
    def from_config(self, config):
        """Inits self from saved config created by get_config."""
        pos_args = config.pop('pos_args')
        return self(*pos_args, **config)

class RNNCFModel(RNNBaseModel):
    """Simple RNN based CF model."""
    num_lstms = 1

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
        super().__init__(maxhd, maxv, mina, maxa, dt=dt)
        # architecture
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units, dropout=dropout,
                                    kernel_regularizer=tf.keras.regularizers.l2(l=l2reg),
                                    recurrent_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=l2reg))
        self.lc_actions = tf.keras.layers.Dense(3)

        # other constants
        self.lstm_units = lstm_units
        self.l2reg = l2reg
        self.dropout = dropout
        self.num_hidden_layers = 1

    def compute_output(self, cur_inputs, hidden_states, training=False):
        """
        Computes output of model given the cur inputs and the hidden states of the lstm_cells.
        Args:
            cur_inputs: tensor with (nveh, nt, 12), gives normalized headways in order of (lead, llead,
                rlead, fol, lfol, rfol)
            hidden_states: list of hidden_states, each a tensor with shape (2, nveh, num_units). The number of
                hidden_states depends on the implementation of the model
            training: Whether to run in training/inference mode. Need to pass training=True if training
                with dropout
        Returns:
            cur_cf: the predicted normalized acceleration
            cur_lc: the predicted log-probabilities of lane changing
            updated hidden_states: in same structure as input
        """
        # call to model
        self.lstm_cell.reset_dropout_mask()
        x, hidden_states = self.lstm_cell(cur_inputs, hidden_states[0], training)
        x = self.dense2(x)
        cur_lc = self.lc_actions(x)  # logits for LC over {left, stay, right} classes for batch
        cur_cf = self.dense1(x)  # current normalized acceleration for the batch
        return cur_cf, cur_lc, [hidden_states]

    def get_config(self):
        """Return any non-trainable parameters in json format."""
        return {'pos_args': (self.maxhd, self.maxv, self.mina, self.maxa,), \
                'lstm_units': self.lstm_units, 'dt': self.dt, 'l2reg':self.l2reg, 'dropout':self.dropout}

    @classmethod
    def from_config(self, config):
        """Inits self from saved config created by get_config."""
        pos_args = config.pop('pos_args')
        return self(*pos_args, **config)


def make_batch(vehs, vehs_counter, ds, nt=5):
    """Create batch of data to send to model.

    Args:
        vehs: list of vehicles in current batch
        vehs_counter: dictionary where keys are indexes, values are tuples of (current time index,
            max time index, vehid)
        ds: dataset, from make_dataset
        nt: number of timesteps in batch

    Returns:
        leadfol_inputs - tensor with shape (nveh, nt, 12), giving the position and speed of the
            the leader, follower, lfol, rfol, llead, rllead at each timestep. Padded with zeros.
            If vehicle is not available at any time, takes value of zero.
            nveh = len(vehs)
        true_traj: (nveh, nt) tensor giving the true vehicle position at each time.
            Padded with zeros
        true_lc_action: (nveh, nt) tensor giving the class label for lane changing output at each time.
        traj_mask: (nveh, nt) tensor  with either 1 or 0, corresponds to the padding. If 0, then it means
            the vehicle is not simulated at that timestep, so it needs to be masked.
        viable_lc: (nveh, nt, 3) tensor giving whether the lane change class is possible at the given time.
    """
    leadfol_inputs = []
    true_traj = []
    traj_mask = []
    true_lc_action = []
    viable_lc = []
    for count, veh in enumerate(vehs):
        t, tmax, unused = vehs_counter[count]
        leadpos, leadspeed = ds[veh]['lead posmem'], ds[veh]['lead speedmem']
        lfolpos, lfolspeed = ds[veh]['lfol posmem'], ds[veh]['lfol speedmem']
        rfolpos, rfolspeed = ds[veh]['rfol posmem'], ds[veh]['rfol speedmem']
        lleadpos, lleadspeed = ds[veh]['llead posmem'], ds[veh]['llead speedmem']
        rleadpos, rleadspeed = ds[veh]['rlead posmem'], ds[veh]['rlead speedmem']
        folpos, folspeed = ds[veh]['fol posmem'], ds[veh]['fol speedmem']
        cur_lc_action = ds[veh]['true lc actions']
        cur_viable_lc = ds[veh]['viable lc']
        posmem = ds[veh]['veh posmem']

        maxt = tmax-t
        uset = min(nt, maxt)
        leftover = nt-uset

        # make inputs
        curlead = np.stack((leadpos[t:t+uset], lleadpos[t:t+uset], rleadpos[t:t+uset], folpos[t:t+uset],
                            lfolpos[t:t+uset], rfolpos[t:t+uset], leadspeed[t:t+uset], lleadspeed[t:t+uset],
                            rleadspeed[t:t+uset], folspeed[t:t+uset], lfolspeed[t:t+uset],
                            rfolspeed[t:t+uset]), axis=1)
        curlead = np.concatenate((curlead, np.zeros((leftover,12)))) if leftover>0 else curlead
        leadfol_inputs.append(curlead)

        curtraj = posmem[t+1:t+uset+1]
        curtraj = np.concatenate((curtraj, np.zeros((leftover,)))) if leftover>0 else curtraj
        true_traj.append(curtraj)

        curmask = np.ones(uset,)
        curmask = np.concatenate((curmask, np.zeros((leftover,)))) if leftover>0 else curmask
        traj_mask.append(curmask)

        curtruelc = cur_lc_action[t:t+uset]
        curtruelc = np.concatenate((curtruelc, np.zeros((leftover,)))) if leftover>0 else curtruelc
        true_lc_action.append(curtruelc)


        curviable = cur_viable_lc[t:t+uset,:]
        temp = np.zeros((leftover,3))
        temp[:,0] = 1
        curviable = np.concatenate((curviable, temp)) if leftover>0 else curviable
        viable_lc.append(curviable)

    return [tf.convert_to_tensor(leadfol_inputs, dtype='float32'),
            tf.convert_to_tensor(true_traj, dtype='float32'),
            tf.convert_to_tensor(true_lc_action, dtype = 'float32'),
            tf.convert_to_tensor(traj_mask, dtype='float32'),
            tf.convert_to_tensor(viable_lc, dtype='float32')]


def train_step_ETP(leadfol_inputs, init_state, hidden_state, true_traj, true_lc_action, traj_mask, viable_lc,
               model, loss_fn, lc_loss_fn, optimizer, vehs_counter, ds):
    """Updates parameters for a single batch of examples.

    Args:
       leadfol_inputs - tensor with shape (nveh, nt, 12), giving the position and speed of the
            the leader, follower, lfol, rfol, llead, rllead at each timestep. Padded with zeros.
            If vehicle is not available at any time, takes value of zero.
            nveh = len(vehs)
        init_state: (nveh, 2) tensor giving the current position, speed for all vehicles in batch
        hidden_state: tensor giving initial hidden state of model
        true_traj: (nveh, nt) tensor giving the true vehicle position at each time.
            Padded with zeros
        true_lc_action: (nveh, nt) tensor giving the class label for lane changing output at each time.
        traj_mask: (nveh, nt) tensor  with either 1 or 0, corresponds to the padding. If 0, then it means
            the vehicle is not simulated at that timestep, so it needs to be masked.
        viable_lc: (nveh, nt, 3) tensor giving whether the lane change class is possible at the given time.
        model: tf.keras.Model
        loss_fn: function takes in y_true, y_pred, sample_weight, and returns the loss
        lc_loss_fn: function takes in y_true, y_pred, and returns the loss for lane changing
        optimizer: tf.keras.optimizer
    Returns:
        pred_traj: output from model
        pred_lc_actions: output from model
        cur_speeds: output from model
        hidden_state: output from model
        cf_loss: loss value from loss_fn
        lc_loss: loss value from lc_loss_fn
    """
    with tf.GradientTape() as tape:
        if model.lc_only:
            pred_traj, pred_lc_action, cur_speeds, hidden_state = \
                model(leadfol_inputs, true_traj, hidden_state, training=True)
        else:
            pred_traj, pred_lc_action, cur_speeds, hidden_state = \
                model(leadfol_inputs, init_state, hidden_state, training=True)

        lc_loss = lc_loss_fn(pred_lc_action, true_lc_action, traj_mask, viable_lc, leadfol_inputs, true_traj,
                             pred_traj, vehs_counter, ds)
        cf_loss = loss_fn(true_traj, pred_traj, traj_mask)
        loss = cf_loss + sum(model.losses) + lc_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred_traj, pred_lc_action, cur_speeds, hidden_state, cf_loss, lc_loss

@tf.function
def train_step(leadfol_inputs, init_state, hidden_state, true_traj, true_lc_action, traj_mask, viable_lc,
               model, loss_fn, lc_loss_fn, optimizer):
    """Updates parameters for a single batch of examples.

    Args:
       leadfol_inputs - tensor with shape (nveh, nt, 12), giving the position and speed of the
            the leader, follower, lfol, rfol, llead, rllead at each timestep. Padded with zeros.
            If vehicle is not available at any time, takes value of zero.
            nveh = len(vehs)
        init_state: (nveh, 2) tensor giving the current position, speed for all vehicles in batch
        hidden_state: tensor giving initial hidden state of model
        true_traj: (nveh, nt) tensor giving the true vehicle position at each time.
            Padded with zeros
        true_lc_action: (nveh, nt) tensor giving the class label for lane changing output at each time.
        traj_mask: (nveh, nt) tensor  with either 1 or 0, corresponds to the padding. If 0, then it means
            the vehicle is not simulated at that timestep, so it needs to be masked.
        viable_lc: (nveh, nt, 3) tensor giving whether the lane change class is possible at the given time.
        model: tf.keras.Model
        loss_fn: function takes in y_true, y_pred, sample_weight, and returns the loss
        lc_loss_fn: function takes in y_true, y_pred, and returns the loss for lane changing
        optimizer: tf.keras.optimizer
    Returns:
        pred_traj: output from model
        pred_lc_actions: output from model
        cur_speeds: output from model
        hidden_state: output from model
        cf_loss: loss value from loss_fn
        lc_loss: loss value from lc_loss_fn
    """
    with tf.GradientTape() as tape:
        if model.lc_only:
            pred_traj, pred_lc_action, cur_speeds, hidden_state = \
                model(leadfol_inputs, true_traj, hidden_state, training=True)
        else:
            pred_traj, pred_lc_action, cur_speeds, hidden_state = \
                model(leadfol_inputs, init_state, hidden_state, training=True)

        lc_loss = lc_loss_fn(pred_lc_action, true_lc_action, traj_mask, viable_lc, leadfol_inputs, true_traj,
                             pred_traj)
        cf_loss = loss_fn(true_traj, pred_traj, traj_mask)
        loss = cf_loss + sum(model.losses) + 300*lc_loss  # magic number for scaling lc loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred_traj, pred_lc_action, cur_speeds, hidden_state, cf_loss, lc_loss


def new_veh_indices(vehs, vehs_counter, nt):
    """
    Utilized by training_loop to calculate which vehicles need to be replaced in the next batch
    Params:
        vehs: list of vehids that are currently in the batch
        vehs_counter: stores current time index, maximum time index (length - 1) for each vehicle
            dict from vehid -> [curr time, max time]
        nt: number of timesteps in a batch
    Returns:
        indices of vehs that need to be replaced
    """
    need_new_vehs = []  # list of indices in batch we need to get a new vehicle for
    for count, veh in enumerate(vehs):
        vehs_counter[count][0] += nt
        if vehs_counter[count][0] >= vehs_counter[count][1]:
            need_new_vehs.append(count)
    return need_new_vehs

def generate_new_states(ds, vehs, vehlist, vehs_counter, need_new_vehs):
    """
    Utilized by training_loop to generate the initial conditions for the new vehicles to be
    added in the next batch.
    Params:
        ds: dataset generated from make_batch
        vehs: list of vehids that are in the current batch
        vehlist: list of all vehids
        vehs_counter: stores current time index, maximum time index (length - 1) for each vehicle
            dict from vehid -> [curr time, max time]
        need_new_vehs: list of indices that need to be replaced. For example, [0, 4] implies that
            the first and fifth vehicle in vehs need to be replaced
    Returns:
        cur_state_updates: tensor version of initial conditions for the new vehicles
            (with shape (num_new_vehs, 2)
        inds_to_update: tensor version of need_new_vehs (with shape (num_new_vehs, 1))
    """
    np.random.shuffle(vehlist)
    new_vehs = vehlist[:len(need_new_vehs)]
    cur_state_updates = []
    for count, ind in enumerate(need_new_vehs):
        new_veh = new_vehs[count]
        vehs[ind] = new_veh
        vehs_counter[ind] = [0, ds[new_veh]['times'][1]-ds[new_veh]['times'][0], new_veh]
        cur_state_updates.append(ds[new_veh]['IC'])
    cur_state_updates = tf.convert_to_tensor(cur_state_updates, dtype='float32')
    inds_to_update = tf.convert_to_tensor([[j] for j in need_new_vehs], dtype='int32')
    return cur_state_updates, inds_to_update

def update_hidden_states(hidden_states, inds_to_update, hidden_state_shape):
    """
    Updates hidden_states with zeros for the new vehicles to be added in the batch, called by
    training_loop.
    Params:
        hidden_states: list of tensors, each with shape (nveh, num_units). The length
            of this tensor depends on the model utilized (and the number of lstm units
            it utilizes)
        inds_to_update: the indices that need to be replaced (tensor w/shape (num_new_vehs,1))
        hidden_state_shape: tuple representing the shape of the hidden states
    Returns:
        updated hidden_states
    """
    hidden_state_updates = tf.zeros(hidden_state_shape)
    for idx, cur_hidden in enumerate(hidden_states):
        h, c = cur_hidden
        h = tf.tensor_scatter_nd_update(h, inds_to_update, hidden_state_updates)
        c = tf.tensor_scatter_nd_update(c, inds_to_update, hidden_state_updates)
        hidden_states[idx] = [h, c]
    return hidden_states

def initialize_states(ds, vehs, num_units, num_hidden_states):
    """
    This generates the initial conditions and hidden states at the beginning of training_loop
    and generate_trajectories.
    Params:
        ds: dataset generated from make_batch
        vehs: vehicles in the batch to be generated
        num_units: the number of units in the hidden state of each lstm cell
        num_hidden_states: the number of hidden states/lstm cells in the model
    Returns:
        cur_state: the initial conditions of each vehicle
        hidden_states: list of hidden states each with shape (nveh, num_units)
    """
    nveh = len(vehs)
    cur_state = [ds[veh]['IC'] for veh in vehs]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
    hidden_states = [[tf.zeros((nveh, num_units)), tf.zeros((nveh, num_units))]
                     for i in range(num_hidden_states)]
    return cur_state, hidden_states

def training_loop(model, loss, lc_loss_fn, optimizer, ds, nbatches=10000, nveh=32, nt=10, m=100,
                  resume_state=None):
    """Trains model by repeatedly calling train_step.

    Args:
        model: tf.keras.Model instance
        loss: tf.keras.losses or custom loss function
        lc_loss: tf.keras.losses or custom loss function for lane changing
        optimizer: tf.keras.optimzers instance
        ds: dataset from make_dataset
        nbatches: number of batches to run
        nveh: number of vehicles in each batch
        nt: number of timesteps per vehicle in each batch
        m: every m batches, print out the loss of the batch.
        resume_state: If you want to resume where you left off, pass in the output from the previous
            call (and use the same optimizer instance).
    Returns:
        resume_state: tuple of (vehs_counter, cur_state, hidden_states), can be used to resume training.
    """
    # initialization
    if not resume_state:
        # select vehicles to put in the batchnbatches=10000, nveh=32, nt=10, m=100
        vehlist = list(ds.keys())
        np.random.shuffle(vehlist)
        vehs = vehlist[:nveh].copy()
        # vehs_counter stores current time index, maximum time index (length - 1) for each vehicle
        # vehs_counter[i] corresponds to vehs[i]
        vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0], veh]
                        for count, veh in enumerate(vehs)}
        # make inputs for network
        cur_state, hidden_states = initialize_states(ds, vehs, model.lstm_units, model.num_lstms)
    else:
        vehs_counter, cur_state, hidden_states = resume_state

    leadfol_inputs, true_traj, true_lc_action, traj_mask, viable_lc = make_batch(vehs, vehs_counter, ds, nt)

    for i in range(nbatches):
        # call train_step
        if lc_loss_fn == expected_LC:
            pred_traj, pred_lc_action, cur_speeds, hidden_states, cf_loss, lc_loss = \
                train_step_ETP(leadfol_inputs, cur_state, hidden_states, true_traj, true_lc_action,
                            traj_mask, viable_lc, model, loss, lc_loss_fn, optimizer,
                            vehs_counter, ds)
        else:
            pred_traj, pred_lc_action, cur_speeds, hidden_states, cf_loss, lc_loss = \
                train_step(leadfol_inputs, cur_state, hidden_states, true_traj, true_lc_action,
                            traj_mask, viable_lc, model, loss, lc_loss_fn, optimizer)

        if i % m == 0:
            print(f'loss for {i}th batch is {cf_loss:.4f}. LC loss is {lc_loss:.4f}')

        #### update iteration by replacing any vehicles which have been fully simulated ####
        if not model.lc_only:
            cur_state = tf.stack([pred_traj[:, -1], cur_speeds], axis=1)  # current state for vehicles in batch

        # check if any vehicles in batch have had their entire trajectory simulated
        need_new_vehs = new_veh_indices(vehs, vehs_counter, nt)

        # update vehicles in batch - update hidden_states and cur_state accordingly
        if len(need_new_vehs) > 0:
            cur_state_updates, inds_to_update = \
                    generate_new_states(ds, vehs, vehlist, vehs_counter, need_new_vehs)

            # update cur_state with generated states
            if not model.lc_only:
                cur_state = tf.tensor_scatter_nd_update(cur_state, inds_to_update, cur_state_updates)

            hidden_states = update_hidden_states(hidden_states, inds_to_update, \
                    (len(need_new_vehs), model.lstm_units))

        leadfol_inputs, true_traj, true_lc_action, traj_mask, viable_lc = \
            make_batch(vehs, vehs_counter, ds, nt)

    return (vehs_counter, cur_state, hidden_states)


def generate_trajectories(model, vehs, ds, loss_fn=None, lc_loss_fn=None):
    """Generate a batch of trajectories.

    Args:
        model: tf.keras.Model
        vehs: list of vehicle IDs
        ds: dataset from make_dataset
        loss_fn: if not None, we will call loss function and return the loss
        lc_loss_fn: if not None, we will call loss function and return the loss (for lane changing)
    Returns:
        pred_traj: output from model
        pred_lc_action: output from model
        cf_loss: result of loss_fn
        lc_loss_fn: result of lc_loss_fn
    """
    # put all vehicles into a single batch, with the number of timesteps equal to the longest trajectory
    vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0], veh]
                        for count, veh in enumerate(vehs)}
    nt = max([i[1] for i in vehs_counter.values()])
    # make inputs for network
    cur_state, hidden_states = initialize_states(ds, vehs, model.lstm_units, model.num_lstms)
    leadfol_inputs, true_traj, true_lc_action, traj_mask, viable_lc = make_batch(vehs, vehs_counter, ds, nt)

    if model.lc_only:
        pred_traj, pred_lc_action, cur_speeds, hidden_state = \
            model(leadfol_inputs, true_traj, hidden_states, training=True)
    else:
        pred_traj, pred_lc_action, cur_speeds, hidden_state = \
            model(leadfol_inputs, cur_state, hidden_states, training=True)

    lc_loss = lc_loss_fn(pred_lc_action, true_lc_action, traj_mask, viable_lc, leadfol_inputs, true_traj,
                             pred_traj, vehs_counter, ds) if lc_loss_fn else 0
    cf_loss = loss_fn(true_traj, pred_traj, traj_mask) if loss_fn else 0

    return pred_traj, pred_lc_action, traj_mask, viable_lc, cf_loss, lc_loss


def generate_vehicle_data(model, vehs, ds, vehdict, loss_fn=None, lc_loss_fn=None, dt=.1):
    """Generate ouput from model for a list of vehicles in VehicleData format."""
    sim_vehdict = copy.deepcopy({veh: vehdict[veh] for veh in vehs})

    pred_traj, pred_lc_action, traj_mask, viable_lc, cf_loss, lc_loss = \
        generate_trajectories(model, vehs, ds, loss_fn, lc_loss_fn)
    print(f'cf loss is {cf_loss:.4f}. LC loss is {lc_loss:.4f}')

    if lc_loss_fn is not None:
        pred_lc_action = logits_to_probabilities(pred_lc_action, traj_mask, viable_lc)

    for count, veh in enumerate(vehs):
        # update posmem and speedmem
        start = sim_vehdict[veh].start
        starttime, endtime = ds[veh]['times']
        nt = endtime- starttime
        true_posmem = sim_vehdict[veh].posmem.data
        posmem = list(pred_traj[count,:nt].numpy())
        true_posmem[starttime+1-start:starttime+1-start+nt] = posmem

        speedmem = [(posmem[i+1] - posmem[i])/dt for i in range(len(posmem)-1)]  # no last speed
        sim_vehdict[veh].speedmem.data[starttime+1-start:starttime-start+nt] = speedmem

        # add new attribute which holds the probabilities
        if lc_loss_fn is not None:
            lcmem = pred_lc_action[count,:nt]
            # calculate ET and P and save those
            intervals = sim_vehdict[veh].lanemem.intervals(starttime, endtime-1)
            ET_P = []
            for count in range(len(intervals)):
                unused, pred_P, unused, pred_ET = calculate_ET_P(lcmem, intervals, count, starttime)
                ET_P.append((pred_ET, pred_P))

            sim_vehdict[veh].lc_actions = lcmem.numpy()
            sim_vehdict[veh].ET_P = ET_P

    return sim_vehdict


def masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns MSE over the entire batch, element-wise weighted with mask_weights."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_mean(temp)


def weighted_masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns masked_MSE over the entire batch, but we don't include 0 weight losses in the average."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_sum(temp)/tf.reduce_sum(mask_weights)


def logits_to_probabilities(pred_lc_action, traj_mask, viable_lc):
    """Converts unnormalized, unmasked logits into masked, normalized probabilities.

    To avoid issues with NaN, all predictions past the maximum time have 1 in the 0 index.
    """
    pred_lc_action = tf.math.exp(pred_lc_action)
    # pred_lc_action = pred_lc_action*tf.expand_dims(traj_mask, axis=-1)
    pred_lc_action = pred_lc_action*viable_lc
    pred_lc_action = pred_lc_action/tf.reduce_sum(pred_lc_action, axis=-1, keepdims=True)
    return pred_lc_action


def SparseCategoricalCrossentropy(pred_lc_action, true_lc_action, traj_mask, viable_lc, *args):
    """Masked 'baseline' loss for training LC model."""
    pred_lc_action = logits_to_probabilities(pred_lc_action, traj_mask, viable_lc)
    return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        true_lc_action, pred_lc_action))/tf.reduce_sum(traj_mask)


def weighted_SparseCategoricalCrossentropy(pred_lc_action, true_lc_action, traj_mask, viable_lc,
                                           lead_inputs, true_traj, pred_traj, *args):
    """Like the 'baseline' loss, but with extra weights based on the CF error."""
    pred_lc_action = logits_to_probabilities(pred_lc_action, traj_mask, viable_lc)
    weights = calculate_lc_hd_weights(lead_inputs, true_traj, pred_traj)
    return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        true_lc_action, pred_lc_action)*weights)/tf.reduce_sum(traj_mask)


def expected_LC(pred_lc_action, true_lc_action, traj_mask, viable_lc, lead_inputs, true_traj, pred_traj,
                vehs_counter, ds):
    """Computes loss based on the expected time of lane change from the logits.
    """
    pred_lc_action = logits_to_probabilities(pred_lc_action, traj_mask, viable_lc)
    nt = traj_mask.shape[-1]

    P_loss_list, E_loss_list = [], []

    for batch_index, values in vehs_counter.items():
        # get intervals for each vehicle
        curindex, maxind, vehid = values
        starttime, endtime = ds[vehid]['times']
        curtime = curindex + starttime
        endtime = curtime + min(maxind-1, nt)
        intervals = ds[vehid]['veh lanemem'].intervals(curtime, endtime)
        cur_lc_action = pred_lc_action[batch_index]

        for count in range(len(intervals)):
            # for each interval, calculate the expected time of lane change
            P, pred_P, ET, pred_ET = calculate_ET_P(cur_lc_action, intervals, count, curtime)

            P_loss_list.append(100*tf.square(pred_P-P))
            E_loss_list.append(tf.square(pred_ET-ET))
    return tf.reduce_mean(P_loss_list) + tf.reduce_mean(E_loss_list)


def calculate_ET_P(pred_lc_action, intervals, count, curtime):
    """For a specific lane changing interval, calculate ET and P.

    A lane changing interval is some sequence of timesteps for a single vehicle which has at most a
    single lane change in it. An interval with no lane change is also allowed.
    P is the probability that no lane change, or a change in the wrong side, will occur
    sometime in the interval.
    ET is the expected time of the predicted lane change, conditional on a lane change occuring.

    Args:
        pred_lc_action: (nt, 3) tensor giving the probabilities of {left, stay, right} at a given timestep
            for a single vehicle
        intervals: Interval representation of a helper.VehMem for the lane data of a single vehicle.
        count: index corresponding to current interval
        curtime: 0 index of pred_lc_action corresponds to time curtime
    Returns:
        P: target P, 0 if current interval contains lane change, 1 otherwise
        pred_P: predicted P calculated from predicted lc actions
        ET: target ET, the length of the current interval if interval contains lane change, 0 otherwise
        pred_ET: predicted ET calculated from predicted lc actions
    """
    # first, calculate targets for the current interval
    curlane, cur_start, cur_end = intervals[count]
    if count < len(intervals)-1:
        if curlane < intervals[count+1][0]:
            useind = 2  # right change
        else:
            useind = 0 # left change
        ET = cur_end-1-cur_start  # target expected time of change
        P = 0  # target probability of not making change in the interval
    else:
        useind = 1 # stay in lane
        ET = 0
        P = 1

    # calculate probability of change at each timestep in interval
    # this has 1 length more than the interval length, because the last index corresponds to the
    # probability of not making the change in the interval
    if useind==1:  # no lane change in interval
        probs = pred_lc_action[int(cur_start-curtime):int(cur_end-curtime), useind]
        probs2 = 1 - probs
    else:
        probs2 = pred_lc_action[int(cur_start-curtime):int(cur_end-curtime), useind]
        probs = 1 - probs2
    probs = tf.concat((tf.ones(1,), probs), axis=0)
    probs2 = tf.concat((probs2, tf.ones(1,)), axis=0)
    # probs = tf.math.cumprod(probs)
    # pred_ET_P = probs*probs2
    probs = tf.math.cumsum(tf.math.log(probs + 1e-5))
    pred_ET_P = probs + tf.math.log(probs2 + 1e-5)

    pred_P = pred_ET_P[-1]
    # pred_ET = tf.tensordot(pred_ET_P[:-1], tf.range(0,int(cur_end-cur_start),dtype='float32'), 1)
    # pred_ET = tf.tensordot(pred_ET_P[:-1], tf.range(0,probs.shape[0]-1,dtype='float32'), 1)
    pred_ET = tf.math.reduce_sum(pred_ET_P[:-1] +  \
            tf.math.log(tf.range(0,probs.shape[0]-1,dtype='float32') + 1e-5))

    return P, pred_P, ET, pred_ET


def calculate_lc_hd_weights(inputs, y_true, y_pred, c=1):
    """
    Calculates LC loss weights utilizing error in headway calculations.
    Args:
        inputs: tf.Tensor with shape (batch_size, nt, 12). This is the input to the RNNCF model
            and it includes information about fol/lead/lfol/rfol/llead/rlead
        y_true: tf.Tensor with shape (batch_size, nt), this is the true trajectories
        y_pred: tf.Tensor with shape (batch_size, nt), this is the output of the predicted positions
            utilizing RNNCF model.
    Returns:
        weights (tf.Tensor) with shape (batch_size, nt)
    """
    lead_true_hd = inputs[:,:,:3] - tf.expand_dims(y_true, axis=2)
    fol_true_hd = tf.expand_dims(y_true, axis=2) - inputs[:,:,3:6]
    true_hd = tf.concat([lead_true_hd, fol_true_hd], axis=2)

    # predicted headway calculation
    lead_pred_hd = inputs[:,:,:3] - tf.expand_dims(y_pred, axis=2)
    fol_pred_hd = tf.expand_dims(y_pred, axis=2) - inputs[:,:,3:6]
    pred_hd = tf.concat([lead_pred_hd, fol_pred_hd], axis=2)

    div = (true_hd - pred_hd) / (true_hd + 1e-5)
    div = -c * tf.math.reduce_sum(div, axis=2)
    return tf.math.exp(div)


def calculate_class_metric(y_true, y_pred, class_id, metric):
    """
    This computes a class-dependent metric given the true y values and the predicted values
    Args:
        y_true: (tf.Tensor) has shape (batch_size, nt)
        y_pred: (tf.Tensor) has shape (batch_size, nt, 3) representing change left, stay, change right
        class_id: (int) class index (0 for left, 1 for stay, 2 for right)
        metric: (tf.keras.metrics.Metric) could be Precision/Recall/f1.
    Returns:
        metric (float)
    """
    metric.reset_states()
    y_true_npy = y_true.numpy().flatten()
    y_pred_npy = tf.argmax(y_pred, axis=2).numpy().flatten()

    sel = y_pred_npy == class_id
    y_pred_npy[sel] = 1
    y_pred_npy[~sel] = 0

    sel = y_true_npy == class_id
    y_true_npy[sel] = 1
    y_true_npy[~sel] = 0

    metric.update_state(y_true_npy, y_pred_npy)
    return metric.result().numpy()


class Trajectories:
    """
    This object saves all information regarding the predicted trajectories and the predicted
    LC outputs from the RNNCF model. It also saves the true lc actions and true trajectories
    """
    def __init__(self, cf_pred, cur_speeds, lc_pred, loss=np.nan, lc_loss=np.nan, \
            true_cf=None, true_lc_action=None, loss_weights=None, lc_weights=None,\
            veh_ids=None, veh_times=None):
        """
        Initializes the Trajectories object.
        Args:
            cf_pred: (np.ndarray) with shape (nveh, nt) of predicted trajectories
            cur_speeds: (np.ndarray) with shape (nveh, 1) of current speeds of all vehicles
            lc_pred: (np.ndarray) with shape (nveh, nt, 3) representing predicted lc action
            loss: (float) overall loss from the model
            lc_loss: (float) lc loss from the model
            true_cf: (np.ndarray) with shape (nveh, nt) w/true trajectories for each vehicle
            true_lc_action: (np.ndarray) with shape (nveh, nt) w/true lc action for each vehicle
            loss_weights: (np.ndarray)  with shape (nveh, nt) the loss weights for the model.
                A zero indicates that the vehicle at that given timestep does not exist within
                the data so does not influence the loss.
            lc_weights: (np.ndarray) with shape (nveh, nt, 3) the lc loss weights for the model.
                A zero in the first colume (e.g. (veh_id, time_id, 0)) indicates that the vehicle
                cannot turn left (either because it's on the left-most lane or the vehicle never
                went to the left-most lane and the vehicle is on the second left-most lane).
                If all columns are zero, this is equivalent to the loss_weights entry being zero,
                meaning that the given timestep does not exist within the data for that vehicle.
            veh_ids: (list of ints) list of vehicle ids
            veh_times: (dictionary from float -> list) maps vehids to simulated times

        """
        self.veh_ids = veh_ids # list of veh ids
        self.veh_to_idx = {vehid:idx for idx, vehid in enumerate(veh_ids)}
        self.veh_times = veh_times

        self.cf_pred = cf_pred
        self.cf_predmem = self._generate_statemem(cf_pred)

        self.lc_pred = lc_pred
        self.lc_predmem = self._generate_statemem(lc_pred)

        self.true_cf = true_cf
        self.cf_truemem = self._generate_statemem(true_cf)

        self.true_lc_action = true_lc_action
        self.lc_truemem = self._generate_statemem(true_lc_action)

        self.cur_speeds = cur_speeds
        self.loss = loss
        self.lc_loss = lc_loss
        self.loss_weights = loss_weights
        self.lc_weights = lc_weights

    def __len__(self):
        return self.true_lc_action.shape[0]

    def _generate_statemem(self, data):
        """
        Converts Data into a dictionary that maps to statemem objects (from helper.py).
        The data is np.ndarray w/shape (nveh, nt,...) where data[0] corresponds to the
        1st vehicle in veh_ids.
        Args:
            data: (np.ndarray with shape (nveh, nt, ...)) data that needs to be converted
                to statemem objects
        Returns:
            dict of float (vehids) to StateMem objects
        """
        mem = {}
        for idx, vehid in enumerate(self.veh_ids):
            mem[vehid] = helper.StateMem(data[idx], self.veh_times[vehid][0])
        return mem

    def confusion_matrix(self, remove_nonsim=True, seed=42):
        """
        This calculates the confusion matrix of LC model. As a pre-processing step, rows
        that do not have any LC prediction (b/c the batch includes timesteps beyond the
        vehicle's lifetime) are removed.
        Args:
            remove_nonsim: (bool) decides whether or not rows beyond the vehicle's
                lifetime are removed
            seed: (int) seed for prediction
        Returns:
            np.ndarray (shape (3,3)) of the confusion matrix
        """
        if remove_nonsim:
            three_zeros = np.zeros((3,))
            sel = ~np.apply_along_axis(lambda x: np.array_equal(x, three_zeros), 2, self.lc_weights)
        else:
            sel = np.ones(self.lc_pred.shape[:2])

        pred_idx = self.sample_lc_outputs(seed=seed)

        # select non-zero elements
        pred_idx = pred_idx[sel]
        true_idx = self.true_lc_action[sel]

        conf_mat = np.zeros((3, 3))
        for true_label in [0, 1, 2]:
            for pred_label in [0, 1, 2]:
                conf_mat[true_label, pred_label] = np.sum((pred_idx == pred_label) \
                        & (true_idx == true_label))
        return conf_mat

    def sample_lc_outputs(self, seed=42):
        """
        Samples from the lc output predicted by the LC model.
        Args:
            seed: (int) seed for prediciton
        Returns:
            np.ndarray (shape (nveh, nt)) of the predicted lane changing action. If 0, change left
                if 1, stay in the same lane, if 2, change right.
        """
        pred = self.lc_weights * self.lc_pred
        pred = tf.unstack(pred, axis=1)

        tf.random.set_seed(seed)
        lc = [tf.squeeze(tf.random.categorical(tf.math.log(i), 1)) for i in pred]
        return tf.stack(lc).numpy().T

    def cf_error(self, vehid, remove_nonsim=True):
        """
        This returns a np.ndarray of the car following error of a particular vehicle, with
        shape (nt,)
        Args:
            vehid: (float) the vehicle id. This is converted to an index
            remove_nonsim: (bool) decides whether or not rows beyond the vehicle's
                lifetime are removed
        Returns:
            cf_error (np.ndarray with shape (nt,)) with nt depending on remove_nonsim
        """
        idx = self.veh_to_idx[vehid]
        cf_error = np.abs((self.true_cf - self.cf_pred)[idx])
        if remove_nonsim:
            sel = np.apply_along_axis(lambda x: x != 0, 0, self.loss_weights[idx])
            cf_error = cf_error[sel]
        return cf_error

    def get_true_lc_action(self, vehid, remove_nonsim=True):
        """
        This returns the lc action for a particular vehicle
        Args:
            vehid: (float) the vehicle id. This is converted to an index
            remove_nonsim: (bool) decides whether or not rows beyond the vehicle's
                lifetime are removed
        Returns
            lc_action (np.ndarray with shape (nt,)) with nt depending on remove_nonsim
                0 representing left, 1 representing stay, 2 representing right
        """
        idx = self.veh_to_idx[vehid]
        lc_action = self.true_lc_action[idx]
        if remove_nonsim:
            sel = np.apply_along_axis(lambda x: x != 0, 0, self.loss_weights[idx])
            lc_action = lc_action[sel]
        return lc_action

    def trajectory_probs(self, vehid, remove_nonsim=True):
        """
        This returns the probability of changing left and changing right (as predicted by the
        LC model) throughout a single vehicle's trajectory (indexed through idx)
        Args:
            vehid: (float) the vehicle id. This is converted to an index
            remove_nonsim: (bool) decides whether or not rows beyond the vehicle's
                lifetime are removed
        Returns:
            np.ndarray (shape (nt, 2)), the first column representing the probability of changing
                left, the second column representing the probability of changing right
        """
        idx = self.veh_to_idx[vehid]
        pred = self.lc_weights[idx] * self.lc_pred[idx]
        pred = pred[:, [0, 2]]

        if remove_nonsim:
            sel = np.apply_along_axis(lambda x: x != 0, 0, self.loss_weights[idx])
            pred = pred[sel]

        return pred



