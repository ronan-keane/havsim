
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import matplotlib.pyplot as plt

import havsim
from havsim.simulation.simulation import *
from havsim.simulation.models import *
from havsim.plotting import plotformat, platoonplot

import copy
import math
#to start we will just use a quantized action space since continuous actions is more complicated
#%%
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
    #NN for actions outputs numbers over each action, we interpret these as the unnormalized
    #log probabilities for each action, and categorical from tf samples the action from probabilities
    #squeeze is just making the output to be 1d
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    #here is basically my understanding of how keras api works in tf 2.1
#    https://www.tensorflow.org/api_docs/python/tf/keras/Model#class_model_2
  #basically you are supposed to define the network architecture in __init__
#and the forward pass is defined in call()
  #main high level api is compile, evaluate, and fit;
  #respectively these do -specify optimizer and training procedure
  #-evaluate metrics on the testing dataset, -trains weights
  #then the main lower-level api is predict_on_batch, test_on_batch, and train_on_batch
  #which respectively: -does a forward pass through network, i.e. gives output given input
  #-does a forward pass and computes loss, given input and target value
  #-does a forward pass, computes loss and gradient of loss, and does an update of weights

  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    # Note: no tf.get_variable(), just simple Keras API!
    self.hidden1 = kl.Dense(64, activation='relu') #hidden layer for actions (policy)
    self.hidden2 = kl.Dense(64, activation='relu') #hidden layer for state-value
    self.value = kl.Dense(1, name = 'value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    # Inputs is a numpy array, convert to a tensor.
    x = tf.convert_to_tensor(inputs)
    # Separate hidden layers from the same input tensor.
    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    # Executes `call()` under the hood.
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)

    return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=-1)

class ACagent:
    def __init__(self,model):
        self.model = model

        self.gamma = .99 #discounting learning_rate = 3e-8
        self.model.compile(
                optimizer = tf.keras.optimizers.RMSprop(lr = 7e-3),
                loss = [self._logits_loss, self._value_loss])
        #I set learning rate small because rewards are pretty big, can try changing
        self.logitloss = kls.SparseCategoricalCrossentropy(from_logits=True)

    def get_action_value(self, curstate):
        return self.model.action_value(curstate)


    def test(self, env, timesteps): #Note that this is pretty much the same as simulate_baseline in the environment = circ_singleav
        env.reset()
        for i in range(timesteps):
            action,value = self.get_action_value(env.curstate)
            reward, done = env.step(action)
            #update state, update cumulative reward
            env.totloss += reward
            #save current state to memory (so we can plot everything)

            if done:
                print("break after {} timesteps with reward {}".format(i, env.totloss))
                break
    def train(self, env, updates=200):
        env.reset()
        # action,value = self.get_action_value(env.curstate)
        I = 1
        for i in range(updates):
            action,value = self.get_action_value(env.curstate)
            #first, get transition and reward
            reward, done = env.step(action)

            #get state value function of transition
            nextaction, nextvalue = self.get_action_value(env.curstate)
            TDerror = (reward + nextvalue - value) #temporal difference error

            self.model.train_on_batch(env.curstate, [tf.stack([I*TDerror[0],tf.cast(action,tf.float32)]), I*TDerror[0]])
            I = I * self.gamma

            # if done:
            #     break
    
    def _value_loss(self, target, value):
        #loss = -\delta * v(s, w) ==> gradient step looks like \delta* \nabla v(s,w)
        return -target*value

    def _logits_loss(self,target, logits):
        #remember, logits are unnormalized log probabilities, so we need to normalize
        #also, logits are a tensor over action space, but we only care about action we choose
#        logits = tf.math.exp(logits)
#        logits = logits / tf.math.reduce_sum(logits)
        getaction = tf.cast(target[1],tf.int32)
        logprob = self.logitloss(getaction, logits) #really the log probability is negative of this.

        return target[0]*logprob

def NNhelper(out, curstate, *args, **kwargs):
    #this is hacky but basically we just want the action from NN to end up
    #in a[i][1]
    return [curstate[1],out]

def myplot(sim, auxinfo, roadinfo, platoon= []):
    #note to self: platoon keyword is messed up becauase plotformat is hacky - when vehicles wrap around they get put in new keys
    meas, platooninfo = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1)
    platoonplot(meas,None,platooninfo,platoon=platoon, lane=1, colorcode= True, speed_limit = [0,25])
    plt.ylim(0,roadinfo[0])
    # plt.show()

class mdp_env:
    def __init__(self):
        self.rewardtable = {0:-1, 1:1, 2:3, 3:4}
        self.curstate = tf.convert_to_tensor([[.123, .456, .789]])
        self.p = 0.1

    def reset(self):
        self.totloss = 0

    def rewardfn(self,action):
        return self.rewardtable[action.numpy()]

    def step(self, action):
        return self.rewardfn(action), np.random.random() <= self.p

#%% initialize agent (we expect the agent to be awful before training)
model = Model(num_actions = 4)
agent = ACagent(model)
testenv = mdp_env()
#%%
agent.test(testenv,200) #200 timesteps
print('total reward before training is '+str(testenv.totloss)+' starting from initial with 200 timesteps')
#%%
    #MWE of training
for i in range(10):
    for j in range(5):
        agent.train(testenv)
    agent.test(testenv,200)
    print('after episode '+str(i + 1)+' total reward is '+str(testenv.totloss)+' starting from initial with 200 timesteps')