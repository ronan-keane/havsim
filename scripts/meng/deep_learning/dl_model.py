"""Similar to deep_learning.py except the model uses rnn layer instead of cell"""
import tensorflow as tf
import numpy as np
from havsim import helper
import math
from time import time

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
            'times' - list of two int times. First is the first time with an observed leader. Second is the
                last time with an observed leader +1. The number of times we call the model is equal to
                times[1] - times[0], which is the length of the lead measurements.
            'posmem' - (1d,) numpy array of observed positions for vehicle, 0 index corresponds to times[0].
                Typically this has a longer length than the lead posmem/speedmem.
            'speedmem' - (1d,) numpy array of observed speeds for vehicle
            'lead posmem' - (1d,) numpy array of positions for leaders, 0 index corresponds to times[0].
                length is subtracted from the lead position.
            'lead speedmem' - (1d,) numpy array of speeds for leaders.
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
                         'lead posmem': leadpos, 'lead speedmem': leadspeed}

    return ds, (maxheadway, maxspeed, minacc, maxacc)

class RNNCFModel(tf.keras.Model):
    """Simple RNN based CF model."""

    def __init__(self, maxhd, maxv, mina, maxa, lstm_units=20, dt=.1, past=25, params=None):
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
        self.conv1 = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same',activation='relu', 
                                            use_bias=True, kernel_regularizer=None,bias_regularizer=None,
                                            input_shape=(None, 3))
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='valid',activation='relu', 
                                            use_bias=True, kernel_regularizer=None,bias_regularizer=None)
        self.maxPool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')
        self.maxPool2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.dropout = tf.keras.layers.Dropout(params['dropout'])
        self.lstm1 = tf.keras.layers.LSTM(lstm_units, dropout=params['dropout'], return_sequences=True,
                                            kernel_regularizer=tf.keras.regularizers.l2(l=params['regularizer']),
                                            recurrent_regularizer=tf.keras.regularizers.l2(l=params['regularizer']))
        self.lstm2 = tf.keras.layers.LSTM(lstm_units, dropout=params['dropout'],
                                            kernel_regularizer=tf.keras.regularizers.l2(l=params['regularizer']),
                                            recurrent_regularizer=tf.keras.regularizers.l2(l=params['regularizer']))
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(10, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=params['regularizer']))

        # normalization constants
        self.maxhd = maxhd
        self.maxv = maxv
        self.mina = mina
        self.maxa = maxa

        # other constants
        self.dt = dt
        self.lstm_units = lstm_units
        self.num_hidden_layers = 1
        self.past = past


    def call(self, inputs, training=False):
        """Updates states for a batch of vehicles.

        Args:
            inputs: list of lead_inputs, cur_state, hidden_states.
                lead_inputs - tensor with shape (nveh, nt, 2), giving the leader position and speed at
                    each timestep.
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
        lead_states, cur_states, mask, nt  = inputs #shapes (nveh, nt+past-1, 2), (nveh,past,2), (nveh, nt+past-1), (1,)
        lead_pos, lead_speeds = tf.unstack(lead_states, axis=2) #shape(nveh, past-1+nt)
        cur_pos, cur_speeds = tf.unstack(cur_states, axis=2) #shape(nveh, past)
        past = self.past #history len

        pos_preds = tf.TensorArray(tf.float32, size=nt)
        speed_preds = tf.TensorArray(tf.float32, size=nt)
        for t in tf.range(nt):
            cur_lead_pos = lead_pos[:, t:t+past]
            cur_lead_speeds = lead_speeds[:, t:t+past]
            cur_hds = cur_lead_pos - cur_pos
            cur_mask = mask[:, t:t+past]

            x = tf.stack([cur_hds/self.maxhd, cur_speeds/self.maxv, cur_lead_speeds/self.maxv], axis=2)
            # self.lstm1.reset_dropout_mask()
            # self.lstm2.reset_dropout_mask()
            # x = self.lstm1(x, training=training, mask=cur_mask)
            # x = self.lstm2(x, training=training, mask=cur_mask)
            x = self.conv1(x)
            x = self.maxPool1(x)
            x = self.conv2(x)
            x = self.maxPool2(x)
            x = self.flatten(x)
            x = self.dense2(x)
            x = self.dense1(x)
            # print('model output shape', x.shape)  #should be batch_size * 1?
            pred_accs = (self.maxa-self.mina)*x + self.mina
            prev_pos = cur_pos[:, -1:]  #shape (nveh, 1)
            prev_speeds = cur_speeds[:, -1:]  #shape (nveh, 1)
            pred_pos =  prev_pos + self.dt*prev_speeds  #shape (nveh, 1)
            pred_speeds = prev_speeds + self.dt*pred_accs  #shape (nveh, 1)
            pos_preds = pos_preds.write(t, pred_pos)
            speed_preds = speed_preds.write(t, pred_speeds)
            cur_pos = tf.concat([cur_pos, pred_pos], axis=1) #shape(nveh, past+1)
            cur_pos = cur_pos[:, -past:] #shape (nveh, past)
            cur_speeds = tf.concat([cur_speeds, pred_speeds], axis=1) #shape(nveh, past+1)
            cur_speeds = cur_speeds[:, -past:]

        pos_preds = tf.transpose(pos_preds.stack(), [1, 0, 2]) #shape (nveh, nt, 1)
        speed_preds = tf.transpose(speed_preds.stack(), [1, 0, 2]) #shape (nveh, nt, 1)
        output = tf.concat([pos_preds, speed_preds], axis=2) #shape (nveh, nt, 2)
        return output

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
                curlead.append([leadpos[t+i], leadspeed[t+i]])
                curtraj.append(posmem[t+i+1])
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
        hidden_state: hidden_states for model
        loss:
    """
    print('TRACING')
    with tf.GradientTape() as tape:
        # would using model.predict_on_batch instead of model.call be faster to evaluate?
        # the ..._on_batch methods use the model.distribute_strategy - see tf.keras source code
        pred_state = model(x, training=True)
        pred_pos, _ = tf.unstack(pred_state, axis=2)
        loss = loss_fn(y_true, pred_pos, sample_weight) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred_state, loss

# @profile
def training_loop(model, loss, optimizer, ds, epochs=100, nbatches=10000, nveh=32, nt=25, m=100, n=20,
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
    vehs_list = tf.data.Dataset.from_tensor_slices(vehlist)
    vehs_list = vehs_list.shuffle(len(vehlist), reshuffle_each_iteration=True)
    veh_batches = vehs_list.batch(nveh)
    tmax = max([ds[veh]['times'][1]-ds[veh]['times'][0] for veh in vehlist])

    for epoch in range(epochs):
        epoch_loss = 0
        for i, veh_batch in enumerate(veh_batches.as_numpy_iterator()):
            batch_start = time()
            vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(veh_batch)}
            cur_states = [ds[veh]['IC'] for veh in veh_batch]
            cur_states = tf.convert_to_tensor(cur_states, dtype='float32')  #shape (nveh, 2)
            cur_states = tf.expand_dims(cur_states, axis=1) #shape (nveh, 1, 2)
            lead_states, true_traj, loss_weights = make_batch(veh_batch, vehs_counter, ds, nt) #lead_states shape (nveh,nt,2)
            paddings = tf.constant([[0,0], [model.past-1,0], [0,0]]) #pad along the first axis
            cur_states = tf.pad(cur_states, paddings)  #shape (nveh, past, 2)
            #lead states will have nt+past-1 steps ??
            lead_states = tf.pad(lead_states, paddings) #shape (nveh, nt+past-1, 2)
            ignore = tf.fill([len(veh_batch), model.past-1], False)
            include = (loss_weights == 1)
            mask = tf.concat([ignore, include], 1) #shape (nveh, nt+past-1)
            batch_loss = 0
            for t in range(math.ceil(tmax/nt)): #loop over the longest trajectory
                #pred_sates should have shape (nveh, nt, 2)
                t1 = time()
                pred_states, loss_value = \
                    train_step([lead_states, cur_states, mask, tf.constant(nt)], true_traj, loss_weights, model,loss, optimizer)
                batch_loss += loss_value
                t2 = time()
                if t == 0 and i == 0:
                    print('{0:.0f}s'.format(t2-t1))
                for count, veh in enumerate(veh_batch):
                    vehs_counter[count][0] += nt
                #add pred_pos and speed to cur_state
                cur_states = tf.concat([cur_states, pred_states], axis=1)
                cur_states = cur_states[:, -model.past:, :]  #shape(nveh, past, 2)
                
                lead_inputs, true_traj, loss_weights = make_batch(veh_batch, vehs_counter, ds, nt)
                lead_states = tf.concat([lead_states, lead_inputs], axis=1)
                lead_states = lead_states[:, -(nt+model.past-1):, :]  #shape(nveh, nt+past, 2)
                include = (loss_weights == 1)
                mask = tf.concat([mask, include], 1)
                mask = mask[:, -(nt+model.past-1):]

            batch_loss /= (t+1)
            epoch_loss += batch_loss
            batch_end = time()
            print('Loss for current batch: {0:.2f}. Took {1:.0f}s '.format(batch_loss.numpy(), batch_end-batch_start))
        print('EPOCH {0} LOSS: {1:.2f}'.format(epoch, epoch_loss/(i+1)))
    # print(train_step.pretty_printed_concrete_signatures())


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
    total_vehs = len(vehs)
    vehs_list = tf.data.Dataset.from_tensor_slices(vehs)
    veh_batches = vehs_list.batch(128)
    tmax = max([ds[veh]['times'][1]-ds[veh]['times'][0] for veh in vehs])

    out_loss = 0
    cur_pos, cur_speeds = [], []
    for b, veh_batch in enumerate(veh_batches.as_numpy_iterator()):
        vehs_counter = {count: [0, ds[veh]['times'][1]-ds[veh]['times'][0]] for count, veh in enumerate(veh_batch)}
        cur_states = [ds[veh]['IC'] for veh in veh_batch]
        cur_states = tf.convert_to_tensor(cur_states, dtype='float32')  #shape (nveh, 2)
        cur_states = tf.expand_dims(cur_states, axis=1) #shape (nveh, 1, 2)
        lead_states, true_traj, loss_weights = make_batch(veh_batch, vehs_counter, ds, tmax) #lead_states shape (nveh,nt,2)

        paddings = tf.constant([[0,0], [model.past-1,0], [0,0]]) #pad along the first axis
        cur_states = tf.pad(cur_states, paddings)  #shape (nveh, past, 2)
        lead_states = tf.pad(lead_states, paddings) #shape (nveh, nt+past-1, 2)
        ignore = tf.fill([len(veh_batch), model.past-1], False)
        include = (loss_weights == 1)
        mask = tf.concat([ignore, include], 1) #shape (nveh, nt+past-1)

        pred_states = model([lead_states, cur_states, mask, tf.constant(tmax)]) 
        pred_pos, pred_speeds = tf.unstack(pred_states, axis=2)
        cur_pos.append(pred_pos)
        cur_speeds.append(pred_speeds)

        if loss is not None:
            cur_loss = loss(true_traj, pred_pos, loss_weights)
            wt_cur_loss = cur_loss * (len(veh_batch)/total_vehs)
            out_loss += wt_cur_loss

    if loss is None:
       return tf.concat(cur_pos, 0), tf.concat(cur_speeds,0) 
    else:
        return tf.concat(cur_pos, 0), tf.concat(cur_speeds, 0), out_loss
