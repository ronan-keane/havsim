"""Trains a tensorflow.keras model for car following behavior."""
import tensorflow as tf
import numpy as np
from havsim import helper
import math

def generate_lane_data(veh_data):
    """
    Generates Labels for Lane-Changing Model for a given vehicle
    Args:
        veh_data: (helper.VehicleData) represents vehicle we're analyzing
    Returns:
        lane_data: python list of -1/0/1, -1 is lane changing to the left, 0 is
            staying in the same lane, and 1 is lane changing to the right
    """
    lane_data = []

    intervals = veh_data.lanemem.intervals()
    for idx, (val, start, end) in list(enumerate(intervals)):
        lane_data += [0] * (end - start - 1)
        if idx < len(intervals) - 1:
            if val < intervals[idx + 1][0]:
                lane_data.append(-1)
            elif val == intervals[idx + 1][0]:
                lane_data.append(0)
            else:
                lane_data.append(1)
    return lane_data

def make_dataset(veh_dict, veh_list, dt=.1):
    """Makes dataset from meas and platooninfo.

    Args:
        veh_dict: see havsim.helper.load_dataset
        dt: timestep
    Returns:
        ds: (reads as dataset) dictionary of vehicles, values are a dictionary with keys
            'IC' - (initial conditions) list of starting position/speed for vehicle
            'times' - list of two int times. First is the first time with an observed leader. Second is the
                last time with an observed leader +1. The number of times we call the model is equal to
                times[1] - times[0], which is the length of the lead measurements.
            'posmem' - (1d,) numpy array of observed positions for vehicle, 0 index corresponds to times[0].
                Typically this has a longer length than the lead posmem/speedmem.
            'speedmem' - (1d,) numpy array of observed speeds for vehicle
            'lead posmem' - (1d,) numpy array of positions for leaders, 0 index corresponds to times[0].
                length is subtracted from the lead position.
            'lead speedmem' - (1d,) numpy array of speeds for leaders.
            'lfolpos' - (1d,) numpy array of positions for lfol
            'lfolspeed' - (1d,) numpy array of speeds for lfol
            'rfolpos' - (1d,) numpy array of positions for rfol
            'rfolspeed' - (1d,) numpy array of speeds for rfol
            'lleadpos' - (1d,) numpy array of positions for llead
            'lleadspeed' - (1d,) numpy array of speeds for llead
            'rleadpos' - (1d,) numpy array of positions for rlead
            'rleadspeed' - (1d,) numpy array of speeds for rlead
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

        leadpos = np.array(veh_data.leadmem.pos[start_sim:end_sim + 1])
        leadlen = np.array(veh_data.leadmem.len[start_sim:end_sim + 1])
        leadpos = leadpos - leadlen # adjust by lead length

        leadspeed = np.array(veh_data.leadmem.speed[start_sim:end_sim + 1])

        # indexing for pos/spd
        vehpos = np.array(veh_data.posmem[start_sim:])
        vehspd = np.array(veh_data.speedmem[start_sim:])

        lanemem = np.array(generate_lane_data(veh_data)[start_sim - start:])

        # lfol rfol llead rllead
        contains_lane1 = 1.0 in veh_data.get_unique_mem(veh_data.lanemem)
        pos_and_spd = [ [[], []] for _ in range(len(veh_data.lcmems))]
        lc_weights = np.zeros((vehpos.shape[0], 2))
        for time in range(start_sim, end + 1):
            for mem_idx, lc_mem in enumerate(veh_data.lcmems):
                if lc_mem[time] is not None:
                    pos_and_spd[mem_idx][1].append(lc_mem.speed[time])
                    # adjust position based off of who is leader, and who is follower
                    # llead/rlead, subtract the length of the leader
                    # lfol/rfol, subtract current vehicle
                    if mem_idx > 1:
                        pos_and_spd[mem_idx][0].append(lc_mem.pos[time] - lc_mem.len[time])
                    else:
                        pos_and_spd[mem_idx][0].append(lc_mem.pos[time] + veh_data.len)
                else:
                    pos_and_spd[mem_idx][1].append(0)
                    pos_and_spd[mem_idx][0].append(0)

                if veh_data.lanemem[time] > 2 or (veh_data.lanemem[time] == 2 and contains_lane1):
                    lc_weights[time - start_sim, 0] = 1
                if veh_data.lanemem[time] < 6:
                    lc_weights[time - start_sim, 1] = 1

        # convert to np.ndarray
        for mem_idx in range(len(pos_and_spd)):
            for j in range(2):
                pos_and_spd[mem_idx][j] = np.array(pos_and_spd[mem_idx][j])

        IC = [vehpos[0], vehspd[0]]
        if end_sim != start_sim:
            headway = leadpos - vehpos[:end_sim + 1 - start_sim]
            maxheadway = max(max(headway), maxheadway)
            maxspeed = max(max(leadspeed), maxspeed)
        else:
            # if there never was a leader, there never was a headway
            headway = []

        vehacc = [(vehpos[i+2] - 2*vehpos[i+1] + vehpos[i])/(dt**2) for i in range(len(vehpos)-2)]
        minacc, maxacc = min(minacc, min(vehacc)), max(maxacc, max(vehacc))

        ds[veh] = {'IC': IC, 'times': [start_sim, min(int(end_sim + 1), end)], 'posmem': vehpos, \
                'speedmem': vehspd, 'lanemem': lanemem, 'lead posmem': leadpos,'lead speedmem': leadspeed, \
                'lfolpos': pos_and_spd[0][0],'lfolspeed': pos_and_spd[0][1], \
                'rfolpos': pos_and_spd[1][0], 'rfolspeed': pos_and_spd[1][1], \
                'lleadpos': pos_and_spd[2][0], 'lleadspeed': pos_and_spd[2][1], \
                'rleadpos': pos_and_spd[3][0], 'rleadspeed': pos_and_spd[3][1], \
                'lc_weights': np.array(lc_weights)}
    return ds, (maxheadway, maxspeed, minacc, maxacc)


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
        self.lc_actions = tf.keras.layers.Dense(3, activation='softmax')

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        # other constants
        self.dt = dt
        self.lstm_units = lstm_units

    def call(self, inputs, lc_weights, training=False):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, cur_state, hidden_states.
                lead_inputs - tensor with shape (nveh, nt, 10), giving the position and speed of the
                    the leader, lfol, rfol, llead, rllead at each timestep.
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
        lc_weights = tf.unstack(lc_weights, axis=1)
        acc_outputs, lc_outputs = [], []
        for cur_lead_input, lc_w in zip(lead_inputs, lc_weights):
            # normalize data for current timestep
            cur_lead_pos, cur_lead_speed, lfol_pos, lfol_speed, rfol_pos, rfol_speed, \
                    llead_pos, llead_speed, rlead_pos, rlead_speed = \
                    tf.unstack(cur_lead_input, axis=1)

            # headway
            curhd = cur_lead_pos-cur_pos
            curhd = curhd/self.maxhd
            cur_lfol_hd = (cur_pos - lfol_pos)/self.maxhd
            cur_rfol_hd = (cur_pos - rfol_pos)/self.maxhd
            cur_llead_hd = (llead_pos - cur_pos)/self.maxhd
            cur_rlead_hd = (rlead_pos - cur_pos)/self.maxhd

            # speed
            cur_lead_speed = cur_lead_speed/self.maxv
            norm_veh_speed = cur_speed/self.maxv
            cur_lfol_speed = lfol_speed/self.maxv
            cur_rfol_speed = rfol_speed/self.maxv
            cur_llead_speed = llead_speed/self.maxv
            cur_rlead_speed = rlead_speed/self.maxv

            cur_inputs = tf.stack([curhd, norm_veh_speed, cur_lead_speed, cur_lfol_hd, \
                    cur_lfol_speed, cur_rfol_hd, cur_rfol_speed, cur_llead_hd, cur_llead_speed, \
                    cur_rlead_hd, cur_rlead_speed], axis=1)

            # call to model
            self.lstm_cell.reset_dropout_mask()
            x, hidden_states = self.lstm_cell(cur_inputs, hidden_states, training)
            x = self.dense2(x)
            cur_lc = self.lc_actions(x)  # get outputed probabilities for LC
            cur_lc = cur_lc * lc_w 

            lc_outputs.append(cur_lc)
            x = self.dense1(x)  # output of the model is current acceleration for the batch

            # update vehicle states
            x = tf.squeeze(x, axis=1)
            cur_acc = (self.maxa-self.mina)*x + self.mina
            cur_pos = cur_pos + self.dt*cur_speed
            cur_speed = cur_speed + self.dt*cur_acc
            acc_outputs.append(cur_pos)

        acc_outputs = tf.stack(acc_outputs, 1)
        lc_outputs = tf.stack(lc_outputs, 1)
        return acc_outputs, lc_outputs, cur_speed, hidden_states


def make_batch(vehs, vehs_counter, ds, nt=5, rp=None, relax_args=None):
    """Create batch of data to send to model.

    Args:
        vehs: list of vehicles in current batch
        vehs_counter: dictionary where keys are indexes, values are tuples of (current time index,
            max time index)
        ds: dataset, from make_dataset
        nt: number of timesteps in batch
        rp: if not None, we apply relaxation using helper.get_fixed_relaxation with parameter rp.
        relax_args: if rp is not None, pass in a tuple of (meas, platooninfo, dt) so the relaxation
            can be calculated

    Returns:
        lead_inputs - tensor with shape (nveh, nt, 10), giving the position and speed of the
            the leader, lfol, rfol, llead, rllead at each timestep. Padded with zeros.
            nveh = len(vehs)
        true_traj: nested python list with shape (nveh, nt) giving the true vehicle position at each time.
            Padded with zeros
        loss_weights: nested python list with shape (nveh, nt) with either 1 or 0, used to weight each sample
            of the loss function. If 0, it means the input at the corresponding index doesn't contribute
            to the loss.
    """
    identity = np.eye(3)
    lead_inputs = []
    true_traj = []
    loss_weights = []
    true_lane_action = []
    lc_weights = []
    for count, veh in enumerate(vehs):
        t, tmax = vehs_counter[count]
        leadpos, leadspeed = ds[veh]['lead posmem'], ds[veh]['lead speedmem']
        lfolpos, lfolspeed = ds[veh]['lfolpos'], ds[veh]['lfolspeed']
        rfolpos, rfolspeed = ds[veh]['rfolpos'], ds[veh]['rfolspeed']
        lleadpos, lleadspeed = ds[veh]['lleadpos'], ds[veh]['lleadspeed']
        rleadpos, rleadspeed = ds[veh]['rleadpos'], ds[veh]['rleadspeed']

        lanemem = ds[veh]['lanemem']
        veh_lc_weights = ds[veh]['lc_weights']
        if rp is not None:
            meas, platooninfo, dt = relax_args
            relax = helper.get_fixed_relaxation(veh, meas, platooninfo, rp, dt=dt)
            leadpos = leadpos + relax
        posmem = ds[veh]['posmem']
        curlead = []
        curtraj = []
        curweights = []
        curtruelane = []
        curlcweights = []
        for i in range(nt):
            # acceleration weights
            if t+i < tmax:
                curlead.append([leadpos[t+i], leadspeed[t+i], lfolpos[t+i], lfolspeed[t+i], \
                        rfolpos[t+i], rfolspeed[t+i], lleadpos[t+i], lleadspeed[t+i], \
                        rleadpos[t+i], rleadspeed[t+i]])
                curtruelane.append(lanemem[t+i] + 1)
                curtraj.append(posmem[t+i+1])
                curweights.append(1)
                curlcweights.append([veh_lc_weights[t+i, 0], 1, veh_lc_weights[t+i, 1]])
            else:
                curlead.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                curtraj.append(0)
                curtruelane.append(0)
                curweights.append(0)
                curlcweights.append([0, 0, 0])
        lead_inputs.append(curlead)
        true_traj.append(curtraj)
        loss_weights.append(curweights)
        true_lane_action.append(curtruelane)
        lc_weights.append(curlcweights)

    return [tf.convert_to_tensor(lead_inputs, dtype='float32'),
            tf.convert_to_tensor(true_traj, dtype='float32'),
            tf.convert_to_tensor(true_lane_action, dtype = 'float32'),
            tf.convert_to_tensor(loss_weights, dtype='float32'),
            tf.convert_to_tensor(lc_weights, dtype='float32')]


def masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns MSE over the entire batch, element-wise weighted with mask_weights."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_mean(temp)


def weighted_masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns masked_MSE over the entire batch, but we don't include 0 weight losses in the average."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_sum(temp)/tf.reduce_sum(mask_weights)


@tf.function
def train_step(x, y_true, lc_true, sample_weight, lc_weights, model, loss_fn, lc_loss_fn, optimizer):
    """Updates parameters for a single batch of examples.

    Args:
        x: input to model
        y_true: target for loss function
        sample_weight: weight for loss function
        lc_weights: weights for lc output
        model: tf.keras.Model
        loss_fn: function takes in y_true, y_pred, sample_weight, and returns the loss
        lc_loss_fn: function takes in y_true, y_pred, and returns the loss for lane changing
        optimizer: tf.keras.optimizer
    Returns:
        y_pred: output from model
        lc_pred: output from lc_model
        cur_speeds: output from model
        hidden_state: hidden_state for model
        loss:
    """
    with tf.GradientTape() as tape:
        # would using model.predict_on_batch instead of model.call be faster to evaluate?
        # the ..._on_batch methods use the model.distribute_strategy - see tf.keras source code
        y_pred, lc_pred, cur_speeds, hidden_state = model(x, training=True, lc_weights=lc_weights)
        # use categorical cross entropy or sparse categorical cross entropy to compute loss over y_pred_lc
        lc_loss = lc_loss_fn(lc_true, lc_pred)
        loss = loss_fn(y_true, y_pred, sample_weight) + sum(model.losses) + 50 * lc_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_pred, lc_pred, cur_speeds, hidden_state, loss, lc_loss

def calculate_class_metric(y_true, y_pred, class_id, metric):
    metric.reset_states()
    # y_true_npy = tf.reshape(y_true, (y_true.shape[0] * y_true.shape[1],)).numpy()
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
    hidden_states = [tf.zeros((nveh, model.lstm_units)),  tf.zeros((nveh, model.lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
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

        # print out and early stopping
        if i % m == 0:
            true_la = true_lane_action.numpy()
            num_left, num_stay, num_right = np.sum(true_la == 0), np.sum(true_la == 1), np.sum(true_la == 2)

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
            # print(f'loss for {i}th batch is {loss_value:.4f}. LC loss is {lc_loss:.4f}')
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
            # hidden_state_updates = [[0 for j in range(model.lstm_units)] for k in need_new_vehs]
            # hidden_state_updates = tf.convert_to_tensor(hidden_state_updates, dtype='float32')
            hidden_state_updates = tf.zeros((len(need_new_vehs), model.lstm_units))
            inds_to_update = tf.convert_to_tensor([[j] for j in need_new_vehs], dtype='int32')

            cur_state = tf.tensor_scatter_nd_update(cur_state, inds_to_update, cur_state_updates)
            h, c = hidden_states
            h = tf.tensor_scatter_nd_update(h, inds_to_update, hidden_state_updates)
            c = tf.tensor_scatter_nd_update(c, inds_to_update, hidden_state_updates)
            hidden_states = [h, c]

        lead_inputs, true_traj, true_lane_action, loss_weights, lc_weights = \
                make_batch(vehs, vehs_counter, ds, nt)


class Trajectory:
    def __init__(self, cf_pred, cur_speeds, lc_pred, loss=np.nan, lc_loss=np.nan, \
            true_cf=None, true_lc_action=None):
        self.cf_pred = cf_pred
        self.cur_speeds = cur_speeds
        self.lc_pred = lc_pred
        self.loss = loss
        self.lc_loss = lc_loss
        self.true_cf = true_cf
        self.true_lc_action = true_lc_action

    def confusion_matrix(self):
        conf_mat = np.zeros((3, 3))
        for true_label in [0, 1, 2]:
            for pred_label in [0, 1, 2]:
                conf_mat[true_label, pred_label] = np.sum((np.argmax(self.lc_pred, axis=2) == pred_label) \
                        & (self.true_lc_action == true_label))
        return conf_mat

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
        y_pred: tensor of vehicle trajectories, shape of (number of vehicles, number of timesteps)
        cur_speeds: tensor of current vehicle speeds, shape of (number of vehicles, 1)
        out_loss: tensor of overall loss, shape of (1,)
        out_lc_loss: tensor of lane changing loss, shape of (1,)
    """
    # put all vehicles into a single batch, with the number of timesteps equal to the longest trajectory
    nveh = len(vehs)
    vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(vehs)}
    nt = max([i[1] for i in vehs_counter.values()])
    cur_state = [ds[veh]['IC'] for veh in vehs]
    hidden_states = [tf.zeros((nveh, model.lstm_units)),  tf.zeros((nveh, model.lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, true_lane_action, loss_weights, lc_weights = \
            make_batch(vehs, vehs_counter, ds, nt, **kwargs)

    y_pred, lc_pred, cur_speeds, hidden_state = model([lead_inputs, cur_state, hidden_states], \
            lc_weights=lc_weights)

    args = [y_pred.numpy(), cur_speeds.numpy(), lc_pred.numpy()]
    kwargs = {'true_cf': true_traj.numpy(), 'true_lc_action': true_lane_action.numpy()}

    if loss is not None:
        out_loss = loss(true_traj, y_pred, loss_weights)
        kwargs['loss'] = out_loss.numpy()
        if lc_loss is not None:
            out_lc_loss = lc_loss(true_lane_action, lc_pred)
            kwargs['lc_loss'] = out_lc_loss.numpy()

    return Trajectory(*args, **kwargs)
