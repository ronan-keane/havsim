"""Trains a tensorflow.keras model for car following behavior."""
import tensorflow as tf
import numpy as np
from havsim import helper


def make_dataset(meas, platooninfo, veh_list, dt=.1):
    """Makes dataset from meas and platooninfo.

    Args:
        meas: see havsim.helper.makeplatooninfo
        platooninfo: see havsim.helper.makeplatooninfo
        veh_list: list of vehicle IDs to put into dataset
        dt: timestep
    Returns:
        ds: (reads as dataset) dictionary of vehicles, values are a dictionary with keys
            'IC' - (initial conditions) list of starting position/speed for vehicle
            'times' - list of starting time, last time with observed leader
            'posmem' - (1d,) numpy array of observed positions for vehicle, 0 index corresponds to times[0]
            'speedmem' - (1d,) numpy array of observed speeds for vehicle
            'lead posmem' - (1d,) numpy array of positions for leaders, 0 index corresponds to times[0].
                length is subtracted from the lead position.
            'lead speedmem' - (1d,) numpy array of speeds for leaders.
        maxheadway: max headway observed in training set
        maxspeed: max velocity observed in training set
        minacc: minimum acceleration observed in training set
        maxacc: maximum acceleration observed in training set

    """
    # get normalization for inputs, and get input data
    ds = {}
    maxheadway, maxspeed = 0, 0
    minacc, maxacc = 1e4, -1e4
    for veh in veh_list:




        headway = compute_headway2(veh, meas, platooninfo, h=dt)
        t0, t1, t2, t3 = platooninfo[veh][:4]
        lead = platooninfo[veh][4][0]
        lt0, lt1, lt2, lt3 = platooninfo[lead][:4]
        leadspeed = meas[lead][t1-lt0:t2+1-lt0, 3]
        vehpos = meas[veh][t1-t0:t2+1-t0, 2]
        vehspd = meas[veh][t1-t0:t2+1-t0, 3]
        leadpos = headway + vehpos
        IC = [meas[veh][t1-t0, 2], meas[veh][t1-t0, 3]]

        vehacc = [(vehpos[i+2] - 2*vehpos[i+1] + vehpos[i])/(dt**2) for i in range(len(vehpos)-2)]
        minacc, maxacc = min(minacc, min(vehacc)), max(maxacc, max(vehacc))
        maxheadway = max(max(headway), maxheadway)
        maxspeed = max(max(leadspeed), maxspeed)
        ds[veh] = {'IC': IC, 'times': [t1, t2], 'posmem': vehpos,
                         'lead posmem': leadpos, 'lead speedmem': leadspeed}

    return ds, maxheadway, maxspeed, minacc, maxacc


class RNNCFModel(tf.keras.Model):
    """Simple RNN based CF model."""

    def __init__(self, maxhd, maxv, mina, maxa, dt=.1):
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
        self.lstm_cell = tf.keras.layers.LSTMCell(20)
        self.dense1 = tf.keras.layers.Dense(1)

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa
        self.dt = dt

    def call(self, inputs):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, cur_state, hidden_states.
                lead_inputs - tensor with shape (nveh, nt, 2), giving the leader position and speed at
                    each timestep.
                cur_state -  tensor with shape (nveh, 2) giving the vehicle position and speed at the
                    starting timestep.
                hidden_states - list of the two hidden states, each hidden state is a tensor with shape
                    of (nveh, lstm_units). Initialized as all zeros for the first timestep.

        Returns:
            outputs: tensor of vehicle trajectories, shape of (number of vehicles, number of timesteps)
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
            x, hidden_states = self.lstm_cell(cur_inputs, hidden_states)
            x = self.dense1(x)  # output of the model is current acceleration for the batch

            # update vehicle states
            x = tf.squeeze(x, axis=1)
            cur_acc = (self.maxa-self.mina)*x + self.mina
            cur_speed = cur_speed + self.dt*cur_acc
            cur_pos = cur_pos + self.dt*cur_acc
            outputs.append(cur_pos)

        outputs = tf.stack(outputs, 1)
        return outputs, cur_speed, hidden_states


def make_batch(vehs, vehs_counter, ds, nt=5, lstm_units=20):
    """Create batch of data to send to model.

    Args:
        vehs: list of vehicles in current batch
        vehs_counter: dictionary where keys are indexes, values are tuples of (current time index,
            max time index)
        ds: dataset, from make_dataset
        nt: number of timesteps in batch
        lstm_units: number of LSTM units in model

    Returns:
        lead_inputs: nested python list with shape (nveh, nt, 2), giving the leader position and speed at
            each timestep. Padded with zeros. nveh = len(vehs).
        true_traj: nested python list with shape (nveh, nt) giving the true vehicle position at each time.
            Padded with zeros
        loss_weights: nested python list with shape (nveh, nt) with either 1 or 0, used to weight each sample
            of the loss function. If 0, it means the input at the corresponding index doesn't contribute
            to the loss.
    """
    lead_inputs = []
    true_traj = []
    loss_weights = []
    for count, veh in enumerate(vehs):
        t, tmax = vehs_counter[count]
        leadpos, leadspeed = ds[veh]['lead posmem'], ds[veh]['lead speedmem']
        posmem = ds[veh]['posmem']
        curlead = []
        curtraj = []
        curweights = []
        for i in range(nt):
            if t+i < tmax:
                curlead.append([leadpos[t+i], leadspeed[t+i]])
                curtraj.append(posmem[t+i])
                curweights.append(1)
            else:
                curlead.append([0, 0])
                curtraj.append(0)
                curweights.append(0)
        lead_inputs.append(curlead)
        true_traj.append(curtraj)
        loss_weights.append(curweights)

    return [tf.convert_to_tensor(lead_inputs, dtype='float32'),
            tf.convert_to_tensor(true_traj, dtype='float32'),
            tf.convert_to_tensor(loss_weights, dtype='float32')]


def masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns MSE over the entire batch, element-wise weighted with mask_weights."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_mean(temp)


def train_step(x, y_true, sample_weight, model, loss_fn, optimizer):
    """Updates parameters for a single batch of examples.

    Args:
        x: input to model
        y_true: target for loss function
        sample_weight: weight for loss function
        model: tf.keras.Model
        loss_fn: function takes in y_true, y_pred, sample_weight, and returns the loss
        optimizer: tf.keras.optimizer
    Returns:
        y_pred: output from model
        cur_speeds: output from model
        hidden_state: hidden_state for model
        loss:
    """
    with tf.GradientTape() as tape:
        # would using model.predict_on_batch instead of model.call be faster to evaluate?
        # the ..._on_batch methods use the model.distribute_strategy - see tf.keras source code
        y_pred, cur_speeds, hidden_state = model(x)
        loss = loss_fn(y_true, y_pred, sample_weight)  # would l2 reg. be applied to this? add model.losses?
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_pred, cur_speeds, hidden_state, loss


def training_loop(model, loss, optimizer, ds, nbatches=10000, nveh=32, nt=10, lstm_units=20):
    """Trains model by repeatedly calling train_step.

    Args:
        model: tf.keras.Model instance
        loss: tf.keras.losses or custom loss function
        optimizer: tf.keras.optimzers instance
        ds: dataset from make_dataset
        nbatches: number of batches to run
        nveh: number of vehicles in each batch
        nt: number of timesteps per vehicle in each batch
        lstm_units: number of lstm_units in model
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
    hidden_states = [tf.zeros((nveh, lstm_units)),  tf.zeros((nveh, lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nt, lstm_units)

    for i in range(nbatches):
        veh_states, cur_speeds, hidden_states, loss_value = \
            train_step([lead_inputs, cur_state, hidden_states], true_traj, loss_weights, model,
                       loss, optimizer)
        if i % 10 == 0:
            print('loss for '+str(i)+'th batch is '+str(loss_value))

        # check if any vehicles in batch have had their entire trajectory simulated
        cur_state = tf.stack([veh_states[:, -1], cur_speeds], axis=1)
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
            hidden_state_updates = [[0 for j in range(lstm_units)] for k in need_new_vehs]
            hidden_state_updates = tf.convert_to_tensor(hidden_state_updates, dtype='float32')
            inds_to_update = tf.convert_to_tensor([[j] for j in need_new_vehs], dtype='int32')

            cur_state = tf.tensor_scatter_nd_update(cur_state, inds_to_update, cur_state_updates)
            h, c = hidden_states
            h = tf.tensor_scatter_nd_update(h, inds_to_update, hidden_state_updates)
            c = tf.tensor_scatter_nd_update(c, inds_to_update, hidden_state_updates)
            hidden_states = [h, c]

        lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nt, lstm_units)


def generate_trajectories(model, vehs, ds, lstm_units=20, loss=None):
    """Generate a batch of trajectories.

    Args:
        model: tf.keras.Model
        vehs: list of vehicle IDs
        ds: dataset from make_dataset
        lstm_units: number of lstm_units in model
        loss: if not None, we will call loss function and return the loss
    Returns:
        y_pred: tensor of vehicle trajectories, shape of (number of vehicles, number of timesteps)
        cur_speeds: tensor of current vehicle speeds, shape of (number of vehicles, 1)
    """
    # put all vehicles into a single batch, with the number of timesteps equal to the longest trajectory
    nveh = len(vehs)
    vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(vehs)}
    nt = max([i[1] for i in vehs_counter.values()])
    cur_state = [ds[veh]['IC'] for veh in vehs]
    hidden_states = [tf.zeros((nveh, lstm_units)),  tf.zeros((nveh, lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nt, lstm_units)

    y_pred, cur_speeds, hidden_state = model([lead_inputs, cur_state, hidden_states])
    if loss is not None:
        out_loss = loss(true_traj, y_pred, loss_weights)
        return y_pred, cur_speeds, out_loss
    else:
        return y_pred, cur_speeds