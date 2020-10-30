# imports and load data
import pickle
import numpy as np
import tensorflow as tf
import math
from havsim import helper

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

#%% modify deep_learning code to incorporate relaxation
"""
only minor changes to make_dataset, RNNCFModel.call, make_batch to support an extra input - the relaxation amounts
"""

def make_dataset(meas, platooninfo, veh_list, rp=15, dt=.1):
    """Makes dataset from meas and platooninfo.

    Args:
        meas: see havsim.helper.makeplatooninfo
        platooninfo: see havsim.helper.makeplatooninfo
        veh_list: list of vehicle IDs to put into dataset
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
            'lc times' - (1d,) numpy array with relaxation amounts at each time
        normalization amounts: tuple of
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
        # get data
        t0, t1, t2, t3 = platooninfo[veh][:4]
        leadpos, leadspeed = helper.get_lead_data(veh, meas, platooninfo, dt=dt)
        # relaxation added
        relax = helper.get_fixed_relaxation(veh, meas, platooninfo, rp, dt=dt)
        leadpos = leadpos + relax
        # new input indicates lane change
        leadinfo = helper.makeleadinfo([veh], platooninfo, meas)
        rinfo = helper.makerinfo([veh], platooninfo, meas, leadinfo, mergertype=None)
        lc_input = np.zeros((t2+1-t1,))
        for lc in rinfo[0]:
            time, relax_amount = lc
            lc_input[time-t1] = relax_amount

        vehpos = meas[veh][t1-t0:, 2]
        vehspd = meas[veh][t1-t0:, 3]
        IC = [meas[veh][t1-t0, 2], meas[veh][t1-t0, 3]]
        headway = leadpos - vehpos[:t2+1-t1]

        # normalization + add item to datset
        vehacc = [(vehpos[i+2] - 2*vehpos[i+1] + vehpos[i])/(dt**2) for i in range(len(vehpos)-2)]
        minacc, maxacc = min(minacc, min(vehacc)), max(maxacc, max(vehacc))
        maxheadway = max(max(headway), maxheadway)
        maxspeed = max(max(leadspeed), maxspeed)
        ds[veh] = {'IC': IC, 'times': [t1, min(int(t2+1), t3)], 'posmem': vehpos, 'speedmem': vehspd,
                         'lead posmem': leadpos, 'lead speedmem': leadspeed, 'lc times': lc_input}

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
            cur_lead_pos, cur_lead_speed, cur_lc = tf.unstack(cur_lead_input, axis=1)
            curhd = cur_lead_pos-cur_pos
            curhd = curhd/self.maxhd
            cur_lc = cur_lc/self.maxhd
            cur_lead_speed = cur_lead_speed/self.maxv
            norm_veh_speed = cur_speed/self.maxv
            cur_inputs = tf.stack([curhd, norm_veh_speed, cur_lead_speed, cur_lc], axis=1)

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
        leadpos, leadspeed, lctimes = ds[veh]['lead posmem'], ds[veh]['lead speedmem'], ds[veh]['lc times']
        if rp is not None:
            meas, platooninfo, dt = relax_args
            relax = helper.get_fixed_relaxation(veh, meas, platooninfo, rp, dt=dt)
            leadpos = leadpos + relax
        posmem = ds[veh]['posmem']
        curlead = []
        curtraj = []
        curweights = []
        for i in range(nt):
            if t+i < tmax:
                curlead.append([leadpos[t+i], leadspeed[t+i], lctimes[t+i]])
                curtraj.append(posmem[t+i+1])
                curweights.append(1)
            else:
                curlead.append([0, 0, 0])
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


def weighted_masked_MSE_loss(y_true, y_pred, mask_weights):
    """Returns masked_MSE over the entire batch, but we don't include 0 weight losses in the average."""
    temp = tf.math.multiply(tf.square(y_true-y_pred), mask_weights)
    return tf.reduce_sum(temp)/tf.reduce_sum(mask_weights)


@tf.function
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
        y_pred, cur_speeds, hidden_state = model(x, training=True)
        loss = loss_fn(y_true, y_pred, sample_weight) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return y_pred, cur_speeds, hidden_state, loss


def training_loop(model, loss, optimizer, ds, nbatches=10000, nveh=32, nt=10, m=100, n=20,
                  early_stopping_loss=None):
    """Trains model by repeatedly calling train_step.

    Args:
        model: tf.keras.Model instance
        loss: tf.keras.losses or custom loss function
        optimizer: tf.keras.optimzers instance
        ds: dataset from make_dataset
        nbatches: number of batches to run
        nveh: number of vehicles in each batch
        nt: number of timesteps per vehicle in each batch
        m: number of batches per print out. If using early stopping, the early_stopping_loss is evaluated
            every m batches.
        n: if using early stopping, number of batches that the testing loss can increase before stopping.
        early_stopping_loss: if None, we return the loss from train_step every m batches. If not None, it is
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
    lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nt)
    prev_loss = math.inf
    early_stop_counter = 0

    for i in range(nbatches):
        # call train_step
        veh_states, cur_speeds, hidden_states, loss_value = \
            train_step([lead_inputs, cur_state, hidden_states], true_traj, loss_weights, model,
                       loss, optimizer)

        # print out and early stopping
        if i % m == 0:
            if early_stopping_loss is not None:
                loss_value = early_stopping_loss(model)
                if loss_value > prev_loss:
                    early_stop_counter += 1
                    if early_stop_counter >= n:
                        print('loss for '+str(i)+'th batch is '+str(loss_value))
                        model.load_weights('prev_weights')  # folder must exist
                        break
                else:
                    model.save_weights('prev_weights')
                    prev_loss = loss_value
                    early_stop_counter = 0
            print('loss for '+str(i)+'th batch is '+str(loss_value))

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

        lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nt)


def generate_trajectories(model, vehs, ds, loss=None, kwargs={}):
    """Generate a batch of trajectories.

    Args:
        model: tf.keras.Model
        vehs: list of vehicle IDs
        ds: dataset from make_dataset
        loss: if not None, we will call loss function and return the loss
        kwargs: dictionary of keyword arguments to pass to make_batch
    Returns:
        y_pred: tensor of vehicle trajectories, shape of (number of vehicles, number of timesteps)
        cur_speeds: tensor of current vehicle speeds, shape of (number of vehicles, 1)
    """
    # put all vehicles into a single batch, with the number of timesteps equal to the longest trajectory
    nveh = len(vehs)
    vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(vehs)}
    nt = max([i[1] for i in vehs_counter.values()])
    cur_state = [ds[veh]['IC'] for veh in vehs]
    hidden_states = [tf.zeros((nveh, model.lstm_units)),  tf.zeros((nveh, model.lstm_units))]
    cur_state = tf.convert_to_tensor(cur_state, dtype='float32')
    hidden_states = tf.convert_to_tensor(hidden_states, dtype='float32')
    lead_inputs, true_traj, loss_weights = make_batch(vehs, vehs_counter, ds, nt, **kwargs)

    y_pred, cur_speeds, hidden_state = model([lead_inputs, cur_state, hidden_states])
    if loss is not None:
        out_loss = loss(true_traj, y_pred, loss_weights)
        return y_pred, cur_speeds, out_loss
    else:
        return y_pred, cur_speeds


#%%

veh_list = []
for i in meas.keys():
    if len(platooninfo[i][4])>1:  # >0
        veh_list.append(i)
np.random.shuffle(veh_list)
train_veh = veh_list[:-150]  # 300
test_veh = veh_list[-150:]

training, norm = make_dataset(meas, platooninfo, train_veh, rp=12)
maxhd, maxv, mina, maxa = norm
testing, unused = make_dataset(meas, platooninfo, test_veh, rp=12)

model = RNNCFModel(maxhd, maxv, 0, 1, lstm_units=60)
loss = masked_MSE_loss
opt = tf.keras.optimizers.Adam(learning_rate = .0005)

#%% train and save results
training_loop(model, loss, opt, training, nbatches = 10000, nveh = 32, nt = 50)
training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 100)
training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 200)
training_loop(model, loss, opt, training, nbatches = 1000, nveh = 32, nt = 300)
training_loop(model, loss, opt, training, nbatches = 1500, nveh = 32, nt = 500)

def early_stopping_loss(model):
        return generate_trajectories(model, list(testing.keys()), testing,
                                     loss=weighted_masked_MSE_loss)[-1]
training_loop(model, loss, opt, training, nbatches=2000, nveh=32, nt=500, m=100, n=5,
                                early_stopping_loss=early_stopping_loss)



#%%
test = generate_trajectories(model, list(testing.keys()), testing, loss=weighted_masked_MSE_loss)
test2 = generate_trajectories(model, list(training.keys()), training, loss=weighted_masked_MSE_loss)

print(' testing loss was '+str(test[-1]))
print(' training loss was '+str(test2[-1]))