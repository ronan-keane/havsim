
"""
@author: rlk268@cornell.edu
"""
import numpy as np 
import tensorflow as tf 
import tensorflow.keras.layers as kl
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
        
    def get_action_value(self, curstate, avid, avlead):
        avstate = tf.convert_to_tensor([[curstate[avid][1], curstate[avlead][1], curstate[avid][2]]]) #state = current speed, leader speed, headway
        action, value = self.model.action_value(avstate) #forward pass of NN gets action and value function
        #action from NN gives a scalar, we convert it to the quantized acceleration 
        acc = tf.cast(action,tf.float32)*.1-1.5 #30 integer actions -> between -1.5 and 1.4 in increments of .1 
        return action, value, acc 
        
        
    def test(self, env, timesteps): #Note that this is pretty much the same as simulate_baseline in the environment = circ_singleav
        env.reset()
        avid = env.avid
        avlead = env.auxinfo[avid][1]
        for i in range(timesteps): 
            unused, unused, acc = self.get_action_value(env.curstate,avid,avlead)
            nextstate, reward = env.step(acc)
            #update state, update cumulative reward
            env.curstate = nextstate
            env.totloss += reward
            #save current state to memory (so we can plot everything)
            for j in nextstate.keys():
                env.sim[j].append(nextstate[j])
                
    def train(self, env, updates=200)
                
    def _value_loss(self, ):
    
def NNhelper(out, curstate, *args, **kwargs):
    #this is hacky but basically we just want the action from NN to end up 
    #in a[i][1] 
    return [curstate[1],out]

def myplot(sim, auxinfo, roadinfo, platoon= []):
    #note to self: platoon keyword is messed up becauase plotformat is hacky - when vehicles wrap around they get put in new keys 
    meas, platooninfo = plotformat(sim,auxinfo,roadinfo, starttimeind = 0, endtimeind = math.inf, density = 1) 
    platoonplot(meas,None,platooninfo,platoon=platoon, lane=1, colorcode= True, speed_limit = [0,25]) 
    plt.ylim(0,roadinfo[0])

class circ_singleav: #example of single AV environment 
    #basically we just wrap the function simulate_step
    #avid = id of AV
    #simulates on a circular road
    
    def __init__(self, initstate,auxinfo,roadinfo,avid,rewardfn,updatefun=update_cir,dt=.25):
        self.initstate = initstate
        self.auxinfo = auxinfo
        self.auxinfo[avid][6] = NNhelper
        self.roadinfo = roadinfo
        self.avid = avid
        self.updatefun = updatefun
        self.dt = dt
        self.rewardfn = rewardfn

    def reset(self): #reset to beginning of simulation 
        self.curstate = self.initstate
        self.sim = {i:[self.curstate[i]] for i in self.initstate.keys()} 
        self.vavg = {i:initstate[i][1]  for i in initstate.keys()}
        self.totloss = 0
        
    def step(self, action): #basically just a wrapper for simulate step to get the next timestep
        #simulate_step does all the updating; first line is just a hack which can be cleaned later
        self.auxinfo[self.avid][5] = action 
        nextstate, unused = simulate_step(self.curstate, self.auxinfo,self.roadinfo,self.updatefun,self.dt)
        
        #get reward, update average velocity
        reward, vavg = self.rewardfn(nextstate,self.vavg)
        self.vavg = vavg
        return nextstate, reward
    
    def simulate_baseline(self, CFmodel, p, timesteps): #can insert a CF model and parameters (e.g. put in human model or parametrized control model)
        #for debugging purposes to verify that timestepping is done correctly 
        #if using deep RL the code to simulate/test is the same except action is chosen from NN 
        self.reset()
        avlead = self.auxinfo[avid][1]
        for i in range(timesteps): 
            action = CFmodel(p, self.curstate[avid],self.curstate[avlead], dt = self.dt)
            nextstate, reward = self.step(action[1])
            #update state, update cumulative reward
            self.curstate = nextstate
            self.totloss += reward
            #save current state to memory (so we can plot everything)
            for j in nextstate.keys():
                self.sim[j].append(nextstate[j])
        

#%%
                #specify simulation 
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers 
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road 
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)

#create simulation environment 
testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward,dt = .25)
#%% sanity check
#test baseline with human AV and with control as a simple check for bugs 
testenv.simulate_baseline(IDM_b3,p,1500) #human model 
print('loss for all human scenario is'+str(testenv.totloss))
myplot(testenv.sim,auxinfo,roadinfo)

testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2], 1500) #control model
print('loss for one AV with parametrized control is'+str(testenv.totloss))
myplot(testenv.sim,auxinfo,roadinfo)
        
    
    #%% initialize agent (we expect the agent to be awful before training)
model = Model(num_actions = 30)
agent = ACagent(model)
#%% 
agent.test(testenv,200) #200 timesteps
myplot(testenv.sim,auxinfo,roadinfo) #plot of all vehicles 
avtraj = np.asarray(testenv.sim[testenv.avid])
plt.figure() #plots, in order, position, speed, and headway time series. 
plt.subplot(1,3,1)
plt.plot(avtraj[:,0])
plt.subplot(1,3,2)
plt.plot(avtraj[:,1])
plt.subplot(1,3,3)
plt.plot(avtraj[:,2])

#you can see that in the initial random strategy, the speed is basically just doing a random walk around 0, 
#because the accelerations are just uniform in [-1.5,1.4]
#so pretty soon the follower vehicle is going to 'collide' and at that point 
#the reward is just going to be dominated by the collision term 

    
    
    
    
    
    