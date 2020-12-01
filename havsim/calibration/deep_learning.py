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
                lane_data.append(1)
            elif val == intervals[idx + 1][0]:
                lane_data.append(0)
            else:
                lane_data.append(-1)
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
            'lead posmem' - (1d,) numpy array of positions for leaders, corresponding to times.
                length is subtracted from the lead position.
            'lead speedmem' - (1d,) numpy array of speeds for leaders.
            'lfolpos' - (1d,) numpy array of positions for lfol
            'lfolspeed' - (1d,) numpy array of speeds for lfol
            'rfolpos' - (1d,) numpy array of positions for rfol
            'rfolspeed' - (1d,) numpy array of speeds for rfol
            'lleadpos' - (1d,) same as lead posmem, but for left leader
            'lleadspeed' - (1d,) same as lead speedmem, but for left leader
            'rleadpos' - (1d,) same as lead posmem, but for right leader
            'rleadspeed' - (1d,) same as lead speedmem, but for right leader
        normalization amounts: tuple of
            maxheadway: max headway observed in training set
            maxspeed: max velocity observed in training set
            minacc: minimum acceleration observed in training set
            maxacc: maximum acceleration observed in training set

    """
    replace_nones = lambda arr: [x if x is not None else 0.0 for x in arr]
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

        folpos = np.array(replace_nones(veh_data.folmem.pos[start_sim:end_sim + 1]))
        folspeed = np.array(replace_nones(veh_data.folmem.speed[start_sim:end_sim + 1]))

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
                'lc_weights': np.array(lc_weights), 'fol posmem': folpos, 'fol speedmem': folspeed}
    return ds, (maxheadway, maxspeed, minacc, maxacc)

class RNNCFModel(tf.keras.Model):
    """Simple RNN based CF model."""

    def __init__(self, maxhd, maxv, mina, maxa, lstm_units=20, dt=.1, params=None):
        """Inits RNN based CF model.

        Args:
            maxhd: max headway (for nomalization of inputs)
            maxv: max velocity (for nomalization of inputs)
            mina: minimum acceleration (for nomalization of outputs)
            maxa: maximum acceleration (for nomalization of outputs)
            dt: timestep
            params: dictionary of model parameters. Used to pass nni autotune hyperparameters
        """
        super().__init__()
        # architecture
        self.lstm_cell = tf.keras.layers.LSTMCell(lstm_units, dropout=params['dropout'], 
                                    kernel_regularizer=tf.keras.regularizers.l2(l=params['regularizer']),
                                    recurrent_regularizer=tf.keras.regularizers.l2(l=params['regularizer']))
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=params['regularizer']))
        self.lc_actions = tf.keras.layers.Dense(3, activation='softmax')

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        # other constants
        self.dt = dt
        self.lstm_units = lstm_units
        self.num_hidden_layers = 1

    def call(self, inputs, training=False):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, cur_state, hidden_states.
                lead_inputs - tensor with shape (nveh, nt, 12), order is (lead, llead, rlead, fol, lfol, rfol)
                    for position, then for speed
                cur_state -  tensor with shape (nveh, 2) giving the vehicle position and speed at the
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

            # call to model
            self.lstm_cell.reset_dropout_mask()
            x, hidden_states = self.lstm_cell(cur_inputs, hidden_states, training)
            x = self.dense2(x)
            cur_lc = self.lc_actions(x)  # get outputed batch probabilities for LC
            x = self.dense1(x)  # output of the model is current acceleration for the batch

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
        return {'pos_args': (self.maxhd, self.maxv, self.mina, self.maxa,), 'lstm_units': self.lstm_units, 'dt': self.dt}

    @classmethod
    def from_config(self, config):
        pos_args = config.pop('pos_args')
        return self(*pos_args, **config)


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
        lead_inputs - tensor with shape (nveh, nt, 12), giving the position and speed of the
            the leader, follower, lfol, rfol, llead, rllead at each timestep. Padded with zeros.
            nveh = len(vehs)
        true_traj: nested python list with shape (nveh, nt) giving the true vehicle position at each time.
            Padded with zeros
        loss_weights: nested python list with shape (nveh, nt) with either 1 or 0, used to weight each sample
            of the loss function. If 0, it means the input at the corresponding index doesn't contribute
            to the loss.
    """
    # identity = np.eye(3)
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
        folpos, folspeed = ds[veh]['fol posmem'], ds[veh]['fol speedmem']
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
                curlead.append([leadpos[t+i], lleadpos[t+i], rleadpos[t+i], folpos[t+i], \
                        lfolpos[t+i], rfolpos[t+i], leadspeed[t+i], lleadspeed[t+i], \
                        rleadspeed[t+i], folspeed[t+i], lfolspeed[t+i], rfolspeed[t+i]])
                curtruelane.append(lanemem[t+i] + 1)
                curtraj.append(posmem[t+i+1])
                curweights.append(1)
                curlcweights.append([veh_lc_weights[t+i, 0], 1, veh_lc_weights[t+i, 1]])
            else:
                curlead.append([0] * 12)
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
            true_cf=None, true_lc_action=None, loss_weights=None, lc_weights=None):
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

        """
        self.cf_pred = cf_pred
        self.cur_speeds = cur_speeds
        self.lc_pred = lc_pred
        self.loss = loss
        self.lc_loss = lc_loss
        self.true_cf = true_cf
        self.true_lc_action = true_lc_action
        self.loss_weights = loss_weights
        self.lc_weights = lc_weights

    def __len__(self):
        return self.true_lc_action.shape[0]

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

    def trajectory_probs(self, idx, remove_nonsim=True):
        """
        This returns the probability of changing left and changing right (as predicted by the
        LC model) throughout a single vehicle's trajectory (indexed through idx)
        Args:
            idx: the index of the vehicle.
        Returns:
            np.ndarray (shape (nt, 2)), the first column representing the probability of changing
                left, the second column representing the probability of changing right
        """
        pred = self.lc_weights[idx] * self.lc_pred[idx]
        pred = pred[:, [0, 2]]

        if remove_nonsim:
            # remove zeros
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
    cur_state = [ds[veh]['IC'] for veh in vehs]
    hidden_states = [tf.zeros((nveh, model.lstm_units)),  tf.zeros((nveh, model.lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')

    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, true_lane_action, loss_weights, lc_weights = \
            make_batch(vehs, vehs_counter, ds, nt, **kwargs)

    y_pred, lc_pred, cur_speeds, hidden_state = model([lead_inputs, cur_state, hidden_states])

    args = [y_pred.numpy(), cur_speeds.numpy(), lc_pred.numpy()]
    kwargs = {'true_cf': true_traj.numpy(), 'true_lc_action': true_lane_action.numpy(), \
            'loss_weights': loss_weights.numpy(), 'lc_weights': lc_weights.numpy()}

    if loss is not None:
        out_loss = loss(true_traj, y_pred, loss_weights)
        kwargs['loss'] = out_loss.numpy()
        if lc_loss is not None:
            out_lc_loss = lc_loss(true_lane_action, lc_pred)
            kwargs['lc_loss'] = out_lc_loss.numpy()

    return Trajectories(*args, **kwargs)
