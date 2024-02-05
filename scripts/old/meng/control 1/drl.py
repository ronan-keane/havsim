
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
import time


#disable gpu 
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
try: 
  # Disable first GPU 
  tf.config.set_visible_devices(physical_devices[1:], 'GPU') 
  logical_devices = tf.config.list_logical_devices('GPU') 
  # Logical device was not created for first GPU 
  assert len(logical_devices) == len(physical_devices) - 1 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 


import havsim
from havsim.simulation.simulationold2 import update2nd_cir, eq_circular, simulate_cir, simulate_step, update_cir
from havsim.plotting import plotformat, platoonplot
from havsim.simulation.models import  IDM_b3, IDM_b3_eql


#to start we will just use a quantized action space since continuous actions is more complicated
#%%
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class PolicyModel(tf.keras.Model):
  def __init__(self, num_actions, num_hiddenlayers = 3, num_neurons = 32, activationlayer = kl.LeakyReLU()):
    super().__init__('mlp_policy')
    self.num_hiddenlayers=num_hiddenlayers
    self.activationlayer = activationlayer
    
    self.hidden1 = kl.Dense(num_neurons) #hidden layer for actions (policy)
    self.hidden11 = kl.Dense(num_neurons)
    if self.num_hiddenlayers > 2:
        self.hidden111 = kl.Dense(num_neurons)
    if self.num_hiddenlayers > 3:
        self.hidden1111 = kl.Dense(num_neurons)

    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()
    
  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
    hidden_logs = self.activationlayer(hidden_logs)
    hidden_logs = self.hidden11(hidden_logs)
    hidden_logs = self.activationlayer(hidden_logs)
    if self.num_hiddenlayers > 2:
        hidden_logs = self.hidden111(hidden_logs)
        hidden_logs = self.activationlayer(hidden_logs)
    if self.num_hiddenlayers > 3:
        hidden_logs = self.hidden1111(hidden_logs)
        hidden_logs = self.activationlayer(hidden_logs)
    return self.logits(hidden_logs)

  def action(self, obs):
    logits = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return tf.squeeze(action, axis=-1)

class PolicyModel2(tf.keras.Model):
  def __init__(self, num_actions, num_hiddenlayers = 2, num_neurons = 32, activationlayer = kl.LeakyReLU()):
    super().__init__('mlp_policy')
    self.activationlayer = activationlayer
    self.hidden1 = kl.Dense(num_neurons, kernel_regularizer = tf.keras.regularizers.l2(l=.1)) #hidden layer for actions (policy)
    self.norm1 = kl.BatchNormalization()
    self.hidden11 = kl.Dense(num_neurons, kernel_regularizer = tf.keras.regularizers.l2(l=.1))
    self.norm11 = kl.BatchNormalization()
    
#    self.hidden111 = kl.Dense(num_neurons, kernel_regularizer = tf.keras.regularizers.l2(l=.1))
#    self.norm111 = kl.BatchNormalization()
#    self.hidden1111 = kl.Dense(num_neurons, kernel_regularizer = tf.keras.regularizers.l2(l=.1))
#    self.norm1111 = kl.BatchNormalization()

    # Logits are unnormalized log probabilities
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()
    
  def call(self, inputs, training = True, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
    hidden_logs = self.activationlayer(hidden_logs)
    hidden_logs = self.norm1(hidden_logs, training = training)
    hidden_logs = self.hidden11(hidden_logs)
    hidden_logs = self.activationlayer(hidden_logs)
    hidden_logs = self.norm11(hidden_logs, training = training)
    
#    hidden_logs = self.hidden111(hidden_logs)
#    hidden_logs = self.activationlayer(hidden_logs)
#    hidden_logs = self.norm111(hidden_logs, training = training)
#    hidden_logs = self.hidden1111(hidden_logs)
#    hidden_logs = self.activationlayer(hidden_logs)
#    hidden_logs = self.norm1111(hidden_logs, training = training)
    return self.logits(hidden_logs)

  def action(self, obs):
    logits = self.call(obs)
    action = self.dist(logits)
    return tf.squeeze(action, axis=-1)


class PolicyModel3(tf.keras.Model):
  def __init__(self, num_actions, num_hiddenlayers = 2, num_neurons = 32, activationlayer = kl.LeakyReLU()):
    super().__init__('mlp_policy')
    self.hidden1 = kl.Dense(560, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=.16)) #hidden layer for actions (policy)
    self.norm1 = kl.BatchNormalization()
    self.hidden11 = kl.Dense(270, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=.16))
    self.norm11 = kl.BatchNormalization()
    self.hidden111 = kl.Dense(num_actions*10, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=.16))
    self.norm111 = kl.BatchNormalization()
    # Logits are unnormalized log probabilities
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()
    
  def call(self, inputs, training = False, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
#    hidden_logs = self.norm1(hidden_logs, training = training)
    hidden_logs = self.hidden11(hidden_logs)
#    hidden_logs = self.norm11(hidden_logs, training = training)
    hidden_logs = self.hidden111(hidden_logs)
#    hidden_logs = self.norm111(hidden_logs, training = training)

    return self.logits(hidden_logs)

  def action(self, obs):
    logits = self.call(obs)
    action = self.dist(logits)
    return tf.squeeze(action, axis=-1)

class ValueModel(tf.keras.Model):
  def __init__(self, num_hiddenlayers = 3, num_neurons=64, activationlayer = kl.ELU()):
    super().__init__('mlp_policy')
    self.num_hiddenlayers=num_hiddenlayers
    self.activationlayer = activationlayer
    
    self.hidden2 = kl.Dense(num_neurons) #hidden layer for state-value
    self.hidden22 = kl.Dense(num_neurons)
    if self.num_hiddenlayers > 2:
        self.hidden222 = kl.Dense(num_neurons)
    if self.num_hiddenlayers > 3:
       self.hidden2222 = kl.Dense(num_neurons)
       
    self.val = kl.Dense(1, name = 'value') 

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_vals = self.hidden2(x)
    hidden_vals = self.activationlayer(hidden_vals)
    hidden_vals = self.hidden22(hidden_vals)
    hidden_vals = self.activationlayer(hidden_vals)
    if self.num_hiddenlayers > 2:
        hidden_vals = self.hidden222(hidden_vals)
        hidden_vals = self.activationlayer(hidden_vals)
    if self.num_hiddenlayers > 3:
        hidden_vals = self.hidden2222(hidden_vals)
        hidden_vals = self.activationlayer(hidden_vals)
    return self.val(hidden_vals)

  def value(self, obs):
    value = self.predict_on_batch(obs)
    return tf.squeeze(value, axis=-1)

class ValueModel2(tf.keras.Model):
  def __init__(self, num_hiddenlayers = 3, num_neurons=64, activationlayer = kl.ReLU()):
    super().__init__('mlp_policy')
    self.activationlayer = activationlayer
    self.hidden2 = kl.Dense(num_neurons, kernel_regularizer = tf.keras.regularizers.l2(l=.1)) #hidden layer for state-value
    self.norm2 = kl.BatchNormalization()
    self.hidden22 = kl.Dense(num_neurons, kernel_regularizer = tf.keras.regularizers.l2(l=.1))
    self.norm22 = kl.BatchNormalization()
    
       
    self.val = kl.Dense(1, name = 'value') 

  def call(self, inputs, training = True, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_vals = self.hidden2(x)
    hidden_vals = self.activationlayer(hidden_vals)
    hidden_vals = self.norm2(hidden_vals, training = training)
    hidden_vals = self.hidden22(hidden_vals)
    hidden_vals = self.activationlayer(hidden_vals)
    hidden_vals = self.norm22(hidden_vals, training = training)
    return self.val(hidden_vals)

  def value(self, obs):
    value = self.call(obs)
    return tf.squeeze(value, axis=-1)

class ValueModel3(tf.keras.Model):
  def __init__(self, num_hiddenlayers = 3, num_neurons=64, activationlayer = kl.ReLU()):
    super().__init__('mlp_policy')
    self.activationlayer = activationlayer
    self.hidden2 = kl.Dense(560, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=.16)) #hidden layer for state-value
    self.norm2 = kl.BatchNormalization()
    self.hidden22 = kl.Dense(52, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=.16))
    self.norm22 = kl.BatchNormalization()
    self.hidden222 = kl.Dense(5, activation='tanh', kernel_regularizer = tf.keras.regularizers.l2(l=.16))
    self.norm222 = kl.BatchNormalization()
    
       
    self.val = kl.Dense(1, name = 'value') 

  def call(self, inputs, training = False, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_vals = self.hidden2(x)
#    hidden_vals = self.norm2(hidden_vals, training = training)
    hidden_vals = self.hidden22(hidden_vals)
#    hidden_vals = self.norm22(hidden_vals, training = training)
    hidden_vals = self.hidden222(hidden_vals)
#    hidden_vals = self.norm222(hidden_vals, training = training)
    return self.val(hidden_vals)

  def value(self, obs):
    value = self.call(obs)
    return tf.squeeze(value, axis=-1)

class ValueModelReinforce(tf.keras.Model): #?What is this for? 
  def __init__(self):
    super().__init__('mlp_policy')
    self.hidden = kl.Dense(1)
    self.threshold = kl.ThresholdedReLU(theta=math.inf)
  def call(self, inputs, **kwargs):
    return self.threshold(self.hidden(inputs))
  def value(self, obs):
    value = self.predict_on_batch(obs)
    return tf.squeeze(value, axis=-1)

class ValueModelLinearBaseline(tf.keras.Model):
  def __init__(self):
    super().__init__('mlp_policy')
    self.hidden = kl.Dense(1, activation=None)
  def call(self, inputs, **kwargs):
    return self.hidden(inputs)
  def value(self, obs):
    value = self.predict_on_batch(obs)
    return tf.squeeze(value, axis=-1)
    
class ACagent:
    def __init__(self,policymodel, valuemodel, data_sz = 256, batch_sz=80,  lr = 0.000085, entropy_const = 1e-6, epochs = 20):
        #self.model = model
        self.policymodel = policymodel
        self.valuemodel = valuemodel
        
        
        self.policymodel.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr), 
                loss = [self._logits_loss])
        self.valuemodel.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr),
                loss = [self._value_loss])
        
        
        self.gamma = 1 #discounting
        self.data_sz = data_sz
        self.batch_sz = batch_sz #batch size
        self.epochs = epochs
        self.entropy_const = entropy_const #constant for entropy maximization term in logit loss function
        self.logit2logprob = kls.SparseCategoricalCrossentropy(from_logits=True) #tensorflow built in converts logits to log probability 
        
    
    def action_value(self, obs):
        return self.policymodel.action(obs), self.valuemodel.value(obs)
        
    def reset(self, env):
        state = env.reset()
        self.counter = 0
        return state
    
    def test(self,env,nruns = 4):
        #nruns = 4 - number of episodes simulated 
    
        #returns - list of total (undiscounted) rewards for each episode, list of nubmer of timesteps in each episode
        
        curstate = self.reset(env)
        run = 0
        rewards = []
        rewardslist = []
        eplenlist = []
        while (run < nruns):
            while True:
                action, value = self.action_value(curstate) #if using batch normalization may want to pass training = False to the model.call
                curstate, reward, done = env.step(action)
                rewards.append(reward)
                self.counter += 1
                if done:
                    eplenlist.append(self.counter)
                    rewardslist.append(sum(rewards))
                    if (run + 1 < nruns):
                        curstate = self.reset(env)
                        rewards = []
                    break
            run += 1
        return rewardslist, eplenlist
    
    def train(self, env, updates=250, by_eps = False, numeps = 1, nTDsteps = 5, simlen = 1500):   
        #env - environment
        #updates - number of times we will call model.fit. This is the number of iterations of the outer loop. 
           #before the first update, the environment is reset. after that, the environment is only reset if done = True is returned 
        #by_eps = False - if True, we generate entire episodes at a time. 
             #If False, we generate self.data_sz steps at a time
        #numeps = 1 - if by_eps = True, numeps is the number of entire episodes generated
        #nTDsteps = 5 - number of steps used for temporal difference errors (also known as advantages)
            #if nTDsteps = -1, then the maximum number of steps possible is used 
        #simlen = 1500 - if by_eps = True, the arrays are all initialized with numeps * simlen size, 
            #so you must provide an upper bound for the number of steps in a single episode
        
        #returns - ep_rewards, total (undiscounted) rewards for all complete episodes 
        
        #initialize
        curstate = self.reset(env)
        leftover = 0 #leftover has sum of undiscounted rewards from an unfinished episode in previous batch 
        
        #memory
        data_sz = env.simlen * numeps if by_eps else self.data_sz
        if nTDsteps < 0:
            nTDsteps = data_sz
        statemem = np.empty((data_sz,env.state_dim))
        rewards = np.empty((data_sz))
        values = np.empty((data_sz))
        actions = np.empty(data_sz)
        dones = np.empty((data_sz))
        
        #output
        ep_rewards = []
        ep_lens = []
        
        action,value = self.action_value(curstate)
        for i in tqdm(range(updates)):# for i in tqdm(range(updates)):
            
            batchlen = 0 #batchlen keeps track of how many steps are in inner loop. batchlen = bstep + 1
            #(self.counter keeps track of how many steps since start of most recent episode)
            epsdone = 0 #keeps track of number of episodes simulated
            curindex = 0 #keeps track of index for start of current episode
            #(or if episode is continueing from previous batch, curindex = 0)
            
            firstdone = -1
            gammafactor = self.counter
            
            for bstep in range(data_sz):
                statemem[bstep] = curstate
                nextstate, reward, done = env.step(action, False)
                nextaction, nextvalue = self.action_value(nextstate)
                self.counter += 1
                
                rewards[bstep] = reward
                values[bstep] = value
                dones[bstep] = done
                actions[bstep] = action
                batchlen += 1
                
                action, value, curstate = nextaction, nextvalue, nextstate    
                if done: #reset simulation 
                    ep_rewards.append(sum(rewards[curindex:batchlen])+leftover)
                    ep_lens.append(self.counter)
                    curindex = batchlen
                    leftover = 0
                    curstate = self.reset(env)
                    action,value = self.action_value(curstate)
                    
                    epsdone += 1
                    if by_eps and epsdone == numeps:
                        break
                    if firstdone == -1:
                        firstdone = batchlen
            leftover += sum(rewards[curindex:batchlen]) #if an episode goes over several batches, keep track of cumulative rewards

            gamma_adjust = np.ones(batchlen)
            adj_idx = firstdone  if (firstdone!= -1) else batchlen #update all gammas if no dones in batch
            gamma_adjust[:adj_idx] = self.gamma**gammafactor
            
            TDerrors = self._TDerrors(rewards[:batchlen], values[:batchlen], dones[:batchlen], nextvalue, gamma_adjust, nTDsteps)
            TDacc = np.reshape(np.append(TDerrors, actions[:batchlen]), (batchlen,2), order = 'F')
        
            self.policymodel.fit(statemem[:batchlen,:], TDacc, batch_size = self.batch_sz, epochs = self.epochs, verbose = 0)
            self.valuemodel.fit(statemem[:batchlen,:], TDerrors, batch_size = self.batch_sz, epochs = self.epochs, verbose = 0)
            

        return ep_rewards, ep_lens
            
    def _TDerrors(self, rewards, values, dones, nextvalue, gamma_adjust, nstep, normalize = False):
        returns = np.append(np.zeros_like(rewards), nextvalue)
        stepreturns = np.zeros_like(rewards)
        
        for t in reversed(range(rewards.shape[0])):
            #cumulative rewards
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1 - dones[t])
            
            #adjustment for nstep
            if ((t + nstep  < len(returns)-1) and (1 not in dones[t:t+nstep])):
                stepreturns[t] = returns[t] \
                            - self.gamma**nstep*returns[t+nstep]  \
                            + self.gamma**nstep*values[t+nstep]
            else:
                stepreturns[t] = returns[t]
                
        returns = np.multiply(stepreturns, gamma_adjust)
        if normalize: 
            temp = returns - values
            return (temp - np.mean(temp))/(np.std(temp)+1e-6)
        else:
            return returns - values

    def _value_loss(self, target, value):        
        return -target*value
    
    def _logits_loss(self,target, logits):
        TDerrors, actions = tf.split(target, 2, axis = -1) 
        logprob = self.logit2logprob(actions, logits, sample_weight = TDerrors) #really the log probability is negative of this.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs,probs)

        return logprob - self.entropy_const*entropy_loss   
        
def NNhelper(out, curstate, *args, **kwargs):
    #this is hacky but basically we just want the action from NN to end up
    #in a[i][1]
    return [curstate[1],out]

def myplot(sim, auxinfo, roadinfo, platoon= []):
    #note to self: platoon keyword is messed up becauase plotformat is hacky - when vehicles wrap around they get put in new keys
    meas, platooninfo = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1)
    platoonplot(meas,None,platooninfo,platoon=platoon, lane=None, colorcode= True, speed_limit = [0,25])
    plt.ylim(0,roadinfo[0])

class circ_singleav: #example of single AV environment
    #basically we just wrap the function simulate_step
    #avid = id of AV
    #simulates on a circular road
    
    def __init__(self, rewardfn, dt=.25,statemem=3, simlen = 1500):
        
        p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
        initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, 
                                                   IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
        sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
        del sim 
        vlist = {i: curstate[i][1] for i in curstate.keys()}
        avid = min(vlist, key=vlist.get)
        
        #for simulation backend
        self.initstate = curstate
        self.auxinfo = auxinfo
        self.auxinfo[avid][6] = NNhelper
        self.roadinfo = roadinfo
        self.updatefun = update_cir
        self.dt = dt
        self.rewardfn = rewardfn
        
        #stuff with memory of states
        self.sim = []
        self.mem = statemem
        self.simlen = simlen #maximum number of steps before done 
        #stuff for building states (len self.mem+1 tuple of past states)
        self.paststates = [] #holds sequence of states
        self.statecnt = 0
        self.state_dim = (self.mem)*5

        #normalization for state
        self.hd_m = 1/(50 - 0)
        self.hd_c = 0
        self.spd_m = 1/(15)
        self.spd_c = 0
        
        #indices for av/leader/follower 
        self.avid = avid
        self.avlead = self.auxinfo[self.avid][1]
        self.avfol = [k for k,v in self.auxinfo.items() if v[1] == self.avid][0] 
        
    def reset(self):
        #resets simulation to initial state 
        self.curstate = self.initstate
        del self.sim
        self.sim = {i:[self.curstate[i]] for i in self.initstate.keys()}
        self.vavg = {i:self.initstate[i][1]  for i in self.initstate.keys()}
   
        self.paststates = []
        self.counter = 1 #coutner starts at 1 because it counts the number of states and there is an initial state
        return self.get_state(self.curstate)

    def get_state(self, curstate):
        #curstate is the state for the simulation backend, get_state converts curstate to input for NN models
        state = [(curstate[self.avid][1]+self.spd_c)*self.spd_m, 
                      (curstate[self.avlead][1]+self.spd_c)*self.spd_m, 
                      (curstate[self.avid][2]+self.hd_c)*self.hd_m, 
                      (curstate[self.avfol][1]+self.spd_c)*self.spd_m, 
                      (curstate[self.avfol][2]+self.hd_c)*self.hd_m ]
        
        self.paststates.extend(state)
        if self.counter < self.mem:
            avstate = self.paststates + state*int(self.mem -self.counter)
        else:
            avstate = self.paststates[-self.state_dim:]
        
        return np.asarray([avstate])
    
    def get_acceleration(self,action,curstate):
        #action from NN gives a scalar, we convert it to the quantized acceleration
        acc = float(action)*.25 - 1
        
        #could add in constraint but this kinda messes up the policy model as it has no way to see
        #that the environment changed its action. 
#        nextspeed = curstate[self.avid][1] + self.dt*acc
#        if nextspeed < 0:
#            acc = -curstate[self.avid][1]/self.dt

        return acc
    
    def step(self, action, save_state = True, baseline = False): 
        #does a single step of the environment
        #action - action for the AV to follow (currently, integer for the quantized acceleration)
        #save_state - if True, we save the curstate to sim attribute
        #baseline - True when using the simulate_baseline method
        
        if baseline:
            acc = action
        else:
            acc = self.get_acceleration(action,self.curstate)
        self.auxinfo[self.avid][5] = acc
        nextstate, _ = simulate_step(self.curstate, self.auxinfo,self.roadinfo,self.updatefun,self.dt)
        
        #update environment state 
        self.curstate = nextstate 
        self.counter += 1
        if save_state:
            self.savestate()
        
        #get reward, update average velocity
        reward, vavg = self.rewardfn(nextstate,self.vavg) #rewards have a vavg which is a running average of velocity
        self.vavg = vavg
        
        #
        allheadways = [ nextstate[i][2] for i in nextstate.keys() ]
        shouldterminate = np.any(np.array(allheadways) <= 0)
        
        nextstate = nextstate if baseline else self.get_state(nextstate)
        
        if shouldterminate or self.counter == self.simlen:
            return nextstate, reward, True

        return nextstate, reward, False

    def simulate_baseline(self, CFmodel, p): 
        #CFmodel - function for car model
        #p - its parameters
        
        #returns - total (undiscounted) rewards for a single episode using CFmodel with parameters p 
        #as a baseline for solving the control problem
        self.reset()
        avlead = self.auxinfo[self.avid][1]
        rewards = []
        while True:
            action = CFmodel(p, self.curstate[self.avid],self.curstate[avlead], dt = self.dt)
            nextstate, reward, done = self.step(action[1], baseline = True)
            rewards.append(reward)
            #update state, update cumulative reward
            self.curstate = nextstate
            
            if done:
                break
        return sum(rewards)
    
    def savestate(self):
        #saves simulation's current state (as opposed to the agent's current state) to memory, for plotting/analysis purposes
        for j in self.curstate.keys():
            self.sim[j].append(self.curstate[j])
            
    def plot(self): 
        #spacetime plot
        myplot(self.sim, self.auxinfo, self.roadinfo)
        
    def trajplot(self, vehid = None):
        #plot of position, speed, headway time series
        if vehid == None: 
            vehid = self.avid
        avtraj = np.asarray(self.sim[vehid])
        plt.figure() #plots, in order, position, speed, and headway time series.
        plt.subplot(1,3,1)
        plt.plot(avtraj[:,0])
        plt.ylabel('position')
        plt.subplot(1,3,2)
        plt.plot(avtraj[:,1])
        plt.ylabel('speed')
        plt.subplot(1,3,3)
        plt.plot(avtraj[:,2])
        plt.ylabel('headway')
        
class gym_env:
    def __init__(self, env, simlen = 500):
        self.env = env
        self.initstate = self.env.reset()
        self.state_dim = self.initstate.shape[0]
        self.simlen = simlen
        
    def reset(self):
        self.curstate = self.env.reset()
        return self.get_state(self.curstate)
    
    def get_state(self, curstate):
        return curstate[None, :]
#        return curstate
    
    def step(self, action, *_, **__):
        nextstate, reward, done, _ = self.env.step(action.numpy())
        self.curstate = nextstate
        return self.get_state(nextstate), reward, done
    
#    def savestate(self):
#        pass