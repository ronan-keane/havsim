
"""
@author: rlk268@cornell.edu
"""
import matplotlib.pyplot as plt
from havsim.simulation.models import IDM_b3, FS, drl_reward88, drl_reward8, drl_reward9,  drl_reward, drl_reward5
import numpy as np 
import gym

from drl import PolicyModel, ValueModel, ACagent, circ_singleav, gym_env, PolicyModel2, ValueModel2, PolicyModel3, ValueModel3

#%% initialize agent

##for circular -
testenv = circ_singleav( drl_reward88,dt = .25, simlen = 1500)
policymodel = PolicyModel3(9)
valuemodel = ValueModel3()
agent = ACagent(policymodel, valuemodel)
#baselines for circular
totreward = testenv.simulate_baseline(IDM_b3,[33.33, 1.2, 2, 1.1, 1.5]) #human model
print('loss for all human scenario is '+str(totreward)+
      ' starting from initial with '+str(testenv.simlen)+' maximum timesteps')
totreward = testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2]) #control model
print('loss for one AV with parametrized control is '+str(totreward)+
      ' starting from initial with '+str(testenv.simlen)+' maximum timesteps')

##for gym - 
#mcenv = gym.make('CartPole-v1')
#testenv = gym_env(mcenv)
#policymodel = PolicyModel3(mcenv.action_space.n)
#valuemodel = ValueModel3()
#agent = ACagent(policymodel, valuemodel, batch_sz = 128)

#%% training

#code to do testing
print('Before training')
rewards, eplens = agent.test(testenv,nruns = 5)
print('simulated '+str(len(rewards))+' episodes. average (std dev) reward is '+str(np.mean(rewards))+' (' 
          +str(np.std(rewards))+') episode length is '+str(np.mean(eplens))+' ('+str(np.std(eplens))+')')

#code to do training 
allrewards = []
alleplens = []
for i in range(1500):
#    rewards, eplens = agent.train(testenv, 100, nTDsteps = 5)
    rewards, eplens = agent.train(testenv, updates=1, by_eps = True, numeps = 20, nTDsteps = -1)
    allrewards.extend(rewards)
    alleplens.extend(eplens)
    print('Epoch '+str(i+1)+'. '+str(len(allrewards))+' episodes simulated so far')
    print('simulated '+str(len(rewards))+' episodes. average (std dev) reward is '+str(np.mean(rewards))+' (' 
          +str(np.std(rewards))+') episode length is '+str(np.mean(eplens))+' ('+str(np.std(eplens))+')')
    
#    if i % 100 == 0:
#        testenv.plot()
#        testenv.trajplot()
    

