
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
import gym
#to start we will just use a quantized action space since continuous actions is more complicated
#%%
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
  def __init__(self, num_actions):
    super().__init__('mlp_policy')
    self.hidden1 = kl.Dense(128, activation='relu') #hidden layer for actions (policy)
    self.hidden2 = kl.Dense(128, activation='relu') #hidden layer for state-value
    self.value = kl.Dense(1, name = 'value')
    # Logits are unnormalized log probabilities.
    self.logits = kl.Dense(num_actions, name = 'policy_logits')
    self.dist = ProbabilityDistribution()

  def call(self, inputs, **kwargs):
    x = tf.convert_to_tensor(inputs)
    hidden_logs = self.hidden1(x)
    hidden_vals = self.hidden2(x)
    return self.logits(hidden_logs), self.value(hidden_vals)

  def action_value(self, obs):
    logits, value = self.predict_on_batch(obs)
    action = self.dist.predict_on_batch(logits)
    return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=-1)

class ACagent:
    def __init__(self,model, batch_sz=64):
        self.model = model

        self.gamma = .99 #discounting learning_rate = 3e-8
        self.model.compile(
                optimizer = tf.keras.optimizers.RMSprop(learning_rate = 7e-3), #optimizer = tf.keras.optimizers.RMSprop(learning_rate = 3e-7)
                #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-7,)
                loss = [self._logits_loss, self._value_loss])
        #I set learning rate small because rewards are pretty big, can try changing
        self.logitloss = kls.SparseCategoricalCrossentropy(from_logits=True)
        #stuff for building states (len self.mem+1 tuple of past states)
        self.paststates = [] #holds sequence of states
        self.statecnt = 0
        self.batch_sz = batch_sz

        self.mem = 4
        #keep track of how many steps in simulation we have taken 
        self.counter = 0
        #keep track of discounting 
        self.I = 1
        #goal for how long we want the simulation to be ideally (with no early termination)
        self.simlen = 1500

    def get_action_value(self, curstate):
        return self.model.action_value(curstate)

    def reset(self, env):
        env.reset()
        self.paststates = []
        self.statecnt = 0
        self.counter = 0
        self.I = 1
        
    
    def test(self, env, timesteps, nruns = 4): #Note that this is pretty much the same as simulate_baseline in the environment = circ_singleav
        self.reset(env)

        run = 0
        losses = []
        while (run < nruns):
            for i in range(timesteps):
                action,value = self.get_action_value(env.curstate[None, :])
                env.curstate, reward, done = env.step(np.array(action))
                #update state, update cumulative reward
                env.totloss += reward

                if done:
                    losses.append(env.totloss)
                    self.reset(env)
                    
                    break
            run += 1

        env.totloss = np.sum(losses) / nruns
        
    def train(self, env, updates=250):
        self.reset(env)
        
        # action,value,acc, avstate = self.get_action_value(env.curstate,avid,avlead)
        statemem = np.empty((self.batch_sz,env.curstate.shape[0]))
        
        rewards = np.empty((self.batch_sz))
        values = np.empty((self.batch_sz))
        actions = np.empty(self.batch_sz)
        dones = np.empty((self.batch_sz))
        
        ep_rewards = []
        
        action,value = self.get_action_value(env.curstate[None, :])
        for i in range(updates):
            for bstep in range(self.batch_sz):
                statemem[bstep] = env.curstate
                nextstate, reward, done = env.step(np.array(action))
    #                env.curstate, reward, done, _ 
                nextaction, nextvalue = self.get_action_value(nextstate[None,:])
                env.totloss += reward
                
                
                
#                if done: 
#                    TDerror = reward - value
#                else:
#                    TDerror = (reward + self.gamma*nextvalue - value) #temporal difference error
                
                self.counter += 1

                rewards[bstep] = reward
                values[bstep] = value
                dones[bstep] = done
                actions[bstep] = action
                
                
#                out1[bstep] = tf.stack([TDerror[0],tf.cast(action,tf.float32)])
#                out2[bstep] = TDerror[0]
    
                if done or self.counter >=self.simlen: #reset simulation 
                    ep_rewards.append(env.totloss)
                    self.reset(env)
                    action,value = self.get_action_value(env.curstate[None, :])
                action, value = nextaction, nextvalue
                
            TDerrors = self._TDerrors(rewards, values, dones, nextvalue)
            TDacc = tf.stack([TDerrors, tf.cast(actions, tf.float32)], axis = 1)
            
            self.model.train_on_batch(statemem, [TDacc,TDerrors])
#            self.model.train_on_batch(env.curstate[None,:], [tf.stack([self.I*TDerror[0],tf.cast(action,tf.float32)]), self.I*TDerror[0]])
            
            
            
        return ep_rewards
            
    def _TDerrors(self, rewards, values, dones, nextvalue):
        returns = np.append(np.zeros_like(rewards), nextvalue, axis = -1)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma*returns[t+1]*(1 - dones[t])
        returns = returns[:-1]
        return returns - values

    def _value_loss(self, target, value):
        #loss = -\delta * v(s, w) ==> gradient step looks like \delta* \nabla v(s,w)
        return -target*value
    
#    def _value_loss(self, target, value): 
#        return .5*target**2
        
#    def _value_loss(self, target, value):
#        return .5 * kls.mean_squared_error(target, value)
    
#    def _value_loss(self, target, value): 
#        return .5*tf.math.square(target - value)

    def _logits_loss(self,target, logits):
        #remember, logits are unnormalized log probabilities, so we need to normalize
        #also, logits are a tensor over action space, but we only care about action we choose
#        logits = tf.math.exp(logits)
#        logits = logits / tf.math.reduce_sum(logits)
        TDerrors, actions = tf.split(target, 2, axis = -1) 
#        actions = tf.expand_dims(tf.cast(target[:,1],tf.int32),1)
        logprob = self.logitloss(actions, logits, sample_weight = TDerrors) #really the log probability is negative of this.
        
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs,probs)

        return logprob - 1e-4*entropy_loss   

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

class cartpole_env:
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        self.curstate = self.env.reset()
        self.totloss = 0
        
    def step(self, action):
        nextstate, reward, done, _ = self.env.step(action)
        self.curstate = nextstate
        return nextstate, reward, done

#%% initialize agent (we expect the agent to be awful before training)
env = gym.make('CartPole-v0')
model = Model(num_actions=env.action_space.n)
agent = ACagent(model)
testenv = cartpole_env(env)
#%%
agent.test(testenv,200) #200 timesteps
print('total reward before training is '+str(testenv.totloss)+' starting from initial with 200 timesteps')
#%%
    #MWE of training
rewards = agent.train(testenv,updates=250)
plt.plot(rewards)
plt.ylabel('rewards')
plt.xlabel('episode')
agent.test(testenv,200)
print('total reward is '+str(testenv.totloss))