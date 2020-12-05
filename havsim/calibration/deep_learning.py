"""Tensorflow.keras car following and lane changing models."""

import tensorflow as tf
import numpy as np
from havsim import helper
import math

def generate_lane_data(veh_data):
    """
    Generates Labels for Lane-Changing Model for a given vehicle
    Args:
        veh_data: helper.VehicleData for a single vehicle
    Returns:
        lane_data: python list of -1/0/1, -1 is lane changing to the left, 0 is
            staying in the same lane, and 1 is lane changing to the right
            length = veh_data.end - veh_data.start + 1
    """
    lane_data = []

    intervals = veh_data.lanemem.intervals()
    for idx, (val, start, end) in list(enumerate(intervals)):
        lane_data += [0] * (end - start - 1)
        if idx < len(intervals) - 1:
            if val < intervals[idx + 1][0]:
                lane_data.append(0)
            elif val == intervals[idx + 1][0]:
                lane_data.append(1)
            else:
                lane_data.append(2)
    return lane_data

def make_dataset(veh_dict, veh_list, dt=.1):
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
            'veh posmem' - (1d,) numpy array of observed positions for vehicle, 0 index corresponds
                to times[0]. Typically this has a longer length than the lead posmem/speedmem.
            'veh speedmem' - (1d,) numpy array of observed speeds for vehicle
            'veh lanemem' - intervals representation of lanes for vehicle (helper.VehMem.intervals)
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
        true_lc_actions = np.array(generate_lane_data(veh_data)[start_sim - start:end_sim+2])

        # generate viable_lc - whether or not left/right change is possible at any given timestep
        contains_lane1 = 1.0 in veh_data.get_unique_mem(veh_data.lanemem)
        viable_lc = np.ones((vehpos.shape[0], 3))
        for time in range(start_sim, end + 1):
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
            pass
        vehacc = [(vehpos[i+2] - 2*vehpos[i+1] + vehpos[i])/(dt**2) for i in range(len(vehpos)-2)]
        minacc, maxacc = min(minacc, min(vehacc)), max(maxacc, max(vehacc))

        ds[veh] = {'IC': IC, 'times': [start_sim, min(int(end_sim + 1), end)], 'veh posmem': vehpos,
                'veh speedmem': vehspd, 'true lc actions': true_lc_actions, 'viable lc': np.array(viable_lc),
                'veh lanemem': veh_data.lanemem.intervals(start_sim, end_sim+1),
                'lead posmem': leadpos, 'lead speedmem': leadspeed,
                'lfol posmem': lfolpos,'lfol speedmem': lfolspeed,
                'rfol posmem': rfolpos, 'rfol speedmem': rfolspeed,
                'llead posmem': lleadpos, 'llead speedmem': lleadspeed,
                'rlead posmem': rleadpos, 'rlead speedmem': rleadspeed,
                'fol posmem': folpos, 'fol speedmem': folspeed}

    return ds, (maxheadway, maxspeed, minacc, maxacc)

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

    def call(self, inputs, training=False):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, cur_state, hidden_states.
                lead_inputs - tensor with shape (nveh, nt, 12), order is (lead, llead, rlead, fol, lfol, rfol)
                    for position, then for speed
                init_state -  tensor with shape (nveh, 2) giving the vehicle position and speed at the
                    starting timestep.
                hidden_states - tensor of hidden states with shape (num_hidden_layers, 2, nveh, lstm_units)
                    Initialized as all zeros for the first timestep.
            training: Whether to run in training or inference mode. Need to pass training=True if training
                with dropout.

        Returns:
            outputs: tensor of vehicle positions, shape of (number of vehicles, number of timesteps). Note
                that these are 1 timestep after lead_inputs. E.g. if nt = 2 and lead_inputs has the lead
                measurements for time 0 and 1. Then cur_state has the vehicle position/speed for time 0, and
                outputs has the vehicle positions for time 1 and 2. curspeed would have the speed for time 2,
                and you can differentiate the outputs to get the speed at time 1 if it's needed.
            lc_outputs: (number of vehicles, number of timesteps, 3) tensor giving the predicted logits over
                {left lc, stay in lane, right lc} classes for each timestep.
            curspeed: tensor of current vehicle speeds, shape of (number of vehicles, 1)
            hidden_states: last hidden states for LSTM. Tuple of tensors, where each tensor has shape of
                (number of vehicles, number of LSTM units)
        """
	# prepare data for call
        lead_fol_inputs, init_state, hidden_states = inputs
        lead_fol_inputs = tf.unstack(lead_fol_inputs, axis=1)  # unpacked over time dimension
        cur_pos, cur_speed = tf.unstack(init_state, axis=1)

        pos_outputs, lc_outputs = [], []
        for cur_lead_fol_input in lead_fol_inputs:
            # extract data for current timestep
            lead_hd = (cur_lead_fol_input[:,:3] - tf.expand_dims(cur_pos, axis=1))/self.maxhd
            fol_hd = (tf.expand_dims(cur_pos, axis=1) - cur_lead_fol_input[:,3:6])/self.maxhd
            spds = cur_lead_fol_input[:,6:]/self.maxv

            cur_inputs = tf.concat([lead_hd, fol_hd, spds], axis=1)
            cur_inputs = tf.where(tf.math.is_nan(cur_inputs), tf.ones_like(cur_input), cur_input)

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
            pos_outputs.append(cur_pos)
            lc_outputs.append(cur_lc)

        pos_outputs = tf.stack(pos_outputs, 1)
        lc_outputs = tf.stack(lc_outputs, 1)
        return pos_outputs, lc_outputs, cur_speed, hidden_states

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
            max time index)
        ds: dataset, from make_dataset
        nt: number of timesteps in batch

    Returns:
        lead_inputs - tensor with shape (nveh, nt, 12), giving the position and speed of the
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
    lead_inputs = []
    true_traj = []
    traj_mask = []
    true_lc_action = []
    viable_lc = []
    for count, veh in enumerate(vehs):
        t, tmax = vehs_counter[count]
        leadpos, leadspeed = ds[veh]['lead posmem'], ds[veh]['lead speedmem']
        lfolpos, lfolspeed = ds[veh]['lfol pomems'], ds[veh]['lfol speedmem']
        rfolpos, rfolspeed = ds[veh]['rfol posmem'], ds[veh]['rfol speedmem']
        lleadpos, lleadspeed = ds[veh]['llead posmem'], ds[veh]['llead speedmem']
        rleadpos, rleadspeed = ds[veh]['rlead posmem'], ds[veh]['rlead speedmem']
        folpos, folspeed = ds[veh]['fol posmem'], ds[veh]['fol speedmem']
        cur_lc_action = ds[veh]['veh lcmem']
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
        curlead = np.concatenate((curlead, np.zeros(leftover,12))) if leftover>0 else curlead
        lead_inputs.append(curlead)

        curtraj = posmem[t+1:t+uset+1]
        curtraj = np.concatenate((curtraj, np.zeros(leftover,))) if leftover>0 else curtraj
        true_traj.append(curtraj)

        curmask = np.ones(uset,)
        curmask = np.concatenate((curmask, np.zeros(leftover,))) if leftover>0 else curmask
        traj_mask.append(curmask)

        curtruelc = cur_lc_action[t:t+uset]
        curtruelc = np.concatenate((curtruelc, np.zeros(leftover,))) if leftover>0 else curtruelc
        true_lc_action.append(curtruelc)

        curviable = cur_viable_lc[t:t+uset,:]
        curviable = np.concatenate((curviable,np.zeros(leftover,3))) if leftover>0 else curviable
        viable_lc.append(curviable)

    return [tf.convert_to_tensor(lead_inputs, dtype='float32'),
            tf.convert_to_tensor(true_traj, dtype='float32'),
            tf.convert_to_tensor(true_lc_action, dtype = 'float32'),
            tf.convert_to_tensor(traj_mask, dtype='float32'),
            tf.convert_to_tensor(viable_lc, dtype='float32')]


def masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns MSE over the entire batch, element-wise weighted with mask_weights."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_mean(temp)


def weighted_masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns masked_MSE over the entire batch, but we don't include 0 weight losses in the average."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_sum(temp)/tf.reduce_sum(mask_weights)


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


@tf.function
def train_step(x, y_true, lc_true, sample_weight, lc_weights, model, loss_fn, lc_loss_fn, optimizer):
    """Updates parameters for a single batch of examples.

    Args:
        x: input to model
        y_true: target for loss function
        sample_weight: weight for loss function to masks padded trajectory length, so that batches of
            vehicles have the same length
        lc_weights: weights for lc output which mask lane changes which aren't possible due to topology
        model: tf.keras.Model
        loss_fn: function takes in y_true, y_pred, sample_weight, and returns the loss
        lc_loss_fn: function takes in y_true, y_pred, and returns the loss for lane changing
        optimizer: tf.keras.optimizer
    Returns:
        y_pred: output from model
        lc_pred: output from lc_model
        cur_speeds: output from model
        hidden_state: hidden_states for model
        loss:
    """
    with tf.GradientTape() as tape:
        # would using model.predict_on_batch instead of model.call be faster to evaluate?
        # the ..._on_batch methods use the model.distribute_strategy - see tf.keras source code
        y_pred, lc_pred, cur_speeds, hidden_state = model(x, training=True)

        masked_lc_pred = lc_pred * lc_weights
        # use categorical cross entropy or sparse categorical cross entropy to compute loss over y_pred_lc

        # headway error calculation
        # weight = calculate_lc_hd_weights(x[0], y_true, y_pred)
        weight = calculate_lc_hd_weights(x[0], y_true, y_true)

        # lc_loss = lc_loss_fn(lc_true, lc_pred, sample_weight=loss_weights)
        lc_loss = lc_loss_fn(lc_true, masked_lc_pred, sample_weight=sample_weight*weight)
        cf_loss = loss_fn(y_true, y_pred, sample_weight)

        loss = cf_loss + sum(model.losses) + 10 * lc_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_pred, lc_pred, cur_speeds, hidden_state, loss, lc_loss


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


def training_loop(model, loss, lc_loss_fn, optimizer, ds, nbatches=10000, nveh=32, nt=10, m=100, n=20,
                  early_stopping_loss=None):
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
        m: number of batches per print out. If using early stopping, the early_stopping_loss is evaluated
            every m batches.
        n: if using early stopping, number of batches that the testing loss can increase before stopping.
        early_stopping_loss: if None, we return the loss from train_tep every m batches. If not None, it is
            a function which takes in model, returns a loss value. If the loss increases, we stop the
            training, and load the best weights.
    Returns:
        None.
    """
    # initialization
    # select vehicles to put in the batch
    vehlist = list(ds.keys())
    np.random.shuffle(vehlist)
    vehs = vehlist[:nveh].copy()
    # vehs_counter stores current time index, maximum time index (length - 1) for each vehicle
    # vehs_counter[i] corresponds to vehs[i]
    vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(vehs)}
    # make inputs for network
    cur_state = [ds[veh]['IC'] for veh in vehs]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
    hidden_states = tf.stack([tf.zeros((nveh, model.lstm_units)),  tf.zeros((nveh, model.lstm_units))])
    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, true_lane_action, loss_weights, lc_weights = make_batch(vehs, vehs_counter, ds, nt)
    prev_loss = math.inf
    early_stop_counter = 0

    precision_metric = tf.keras.metrics.Precision()
    recall_metric = tf.keras.metrics.Recall()

    for i in range(nbatches):
        # call train_step
        veh_states, lc_pred, cur_speeds, hidden_states, loss_value, lc_loss = \
            train_step([lead_inputs, cur_state, hidden_states], true_traj, true_lane_action, \
                        loss_weights, lc_weights, model, loss, lc_loss_fn, optimizer)
        if i % m == 0:
            true_la = true_lane_action.numpy()
            num_left, num_stay, num_right = np.sum(true_la == 0), np.sum(true_la == 1), np.sum(true_la == 2)

            # calculate precision/recall of changing to the left and right lanes
            left_prec = calculate_class_metric(true_lane_action, lc_pred, 0, precision_metric)
            right_prec = calculate_class_metric(true_lane_action, lc_pred, 2, precision_metric)
            left_recall = calculate_class_metric(true_lane_action, lc_pred, 0, recall_metric)
            right_recall = calculate_class_metric(true_lane_action, lc_pred, 2, recall_metric)

            if early_stopping_loss is not None:
                loss_value = early_stopping_loss(model)
                if loss_value > prev_loss:
                    early_stop_counter += 1
                    if early_stop_counter >= n:
                        print(f'loss for {i}th batch is {loss_value:.4f}. LC loss is {lc_loss:.4f}\n' + \
                                '\t(left prec, right prec, left recall, right recall):' + \
                                f' {left_prec:.4f}, {right_prec:.4f}, {left_recall:.4f}, {right_recall:.4f}\n' +
                                f'\t(num left, num stay, num right): {num_left}, {num_stay}, {num_right}')
                        model.load_weights('prev_weights')  # folder must exist
                        break
                else:
                    model.save_weights('prev_weights')
                    prev_loss = loss_value
                    early_stop_counter = 0
            print(f'loss for {i}th batch is {loss_value:.4f}. LC loss is {lc_loss:.4f}\n' + \
                    '\t(left prec, right prec, left recall, right recall):' + \
                    f' {left_prec:.4f}, {right_prec:.4f}, {left_recall:.4f}, {right_recall:.4f}\n' +
                    f'\t(num left, num stay, num right): {num_left}, {num_stay}, {num_right}')

        # update iteration
        cur_state = tf.stack([veh_states[:, -1], cur_speeds], axis=1)  # current state for vehicles in batch
        # check if any vehicles in batch have had their entire trajectory simulated
        need_new_vehs = []  # list of indices in batch we need to get a new vehicle for
        for count, veh in enumerate(vehs):
            vehs_counter[count][0] += nt
            if vehs_counter[count][0] >= vehs_counter[count][1]:
                need_new_vehs.append(count)
        # update vehicles in batch - update hidden_states and cur_state accordingly
        if len(need_new_vehs) > 0:
            np.random.shuffle(vehlist)
            new_vehs = vehlist[:len(need_new_vehs)]
            cur_state_updates = []
            for count, ind in enumerate(need_new_vehs):
                new_veh = new_vehs[count]
                vehs[ind] = new_veh
                vehs_counter[ind] = [0, ds[new_veh]['times'][1]-ds[new_veh]['times'][0]]
                cur_state_updates.append(ds[new_veh]['IC'])
            cur_state_updates = tf.convert_to_tensor(cur_state_updates, dtype='float32')
            hidden_state_updates = tf.zeros((len(need_new_vehs), model.lstm_units))
            inds_to_update = tf.convert_to_tensor([[j] for j in need_new_vehs], dtype='int32')

            cur_state = tf.tensor_scatter_nd_update(cur_state, inds_to_update, cur_state_updates)
            h, c = hidden_states
            h = tf.tensor_scatter_nd_update(h, inds_to_update, hidden_state_updates)
            c = tf.tensor_scatter_nd_update(c, inds_to_update, hidden_state_updates)
            hidden_states = [h, c]

        lead_inputs, true_traj, true_lane_action, loss_weights, lc_weights = \
                make_batch(vehs, vehs_counter, ds, nt)


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


def generate_trajectories(model, vehs, ds, loss=None, lc_loss=None, kwargs={}):
    """Generate a batch of trajectories.

    Args:
        model: tf.keras.Model
        vehs: list of vehicle IDs
        ds: dataset from make_dataset
        loss: if not None, we will call loss function and return the loss
        lc_loss: if not None, we will call loss function and return the loss (for lane changing)
        kwargs: dictionary of keyword arguments to pass to make_batch
    Returns:
        Trajectories: object (defined above) that contains information about the predicted and
            true positions, predicted/true lc actions, the loss value of the model, the loss value
            of the lane changing predictions, etc. See class for more details.
    """
    # put all vehicles into a single batch, with the number of timesteps equal to the longest trajectory
    nveh = len(vehs)
    vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(vehs)}
    nt = max([i[1] for i in vehs_counter.values()])
    veh_times = {veh: ds[veh]['times'] for veh in vehs}
    cur_state = [ds[veh]['IC'] for veh in vehs]

    hidden_states = [tf.zeros((nveh, model.lstm_units)),  tf.zeros((nveh, model.lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')

    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, true_lane_action, loss_weights, lc_weights = \
            make_batch(vehs, vehs_counter, ds, nt, **kwargs)

    y_pred, lc_pred, cur_speeds, hidden_state = model([lead_inputs, cur_state, hidden_states])

    args = [y_pred.numpy(), cur_speeds.numpy(), lc_pred.numpy()]
    kwargs = {'true_cf': true_traj.numpy(), 'true_lc_action': true_lane_action.numpy(), \
            'loss_weights': loss_weights.numpy(), 'lc_weights': lc_weights.numpy(), 'veh_ids': vehs, \
            'veh_times': veh_times}

    if loss is not None:
        out_loss = loss(true_traj, y_pred, loss_weights)
        kwargs['loss'] = out_loss.numpy()
        if lc_loss is not None:
            out_lc_loss = lc_loss(true_lane_action, lc_pred)
            kwargs['lc_loss'] = out_lc_loss.numpy()

    return Trajectories(*args, **kwargs)
