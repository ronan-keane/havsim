#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#for profiling/speed test purposes. won't run in this subfolder

from drl import *
#from toysimulation import debugenv
import time
from havsim.simulation.models import drl_reward88, drl_reward8, drl_reward9, IDM_b3, IDM_b3_eql, FS
from havsim.simulation.simulationold2 import update2nd_cir, eq_circular, simulate_cir, update_cir
#tf.compat.v1.disable_eager_execution()

#%%
#specify simulation
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 0, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)
testingtime = 1500
#create simulation environment
testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25)
'''#%% sanity check
##test baseline with human AV and with control as a simple check for bugs
testenv.simulate_baseline(IDM_b3,p,testingtime) #human model
print('loss for all human scenario is '+str(testenv.totloss)+' starting from initial with '+str(testingtime)+' timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
#
testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2], testingtime) #control model
print('loss for one AV with parametrized control is '+str(testenv.totloss)+' starting from initial with '+str(testingtime)+' timesteps')
#    myplot(testenv.sim,auxinfo,roadinfo)
#%% initialize agent
#model = Model(num_actions = 3)
policymodel = PolicyModel(num_actions = 3)
valuemodel = ValueModel()
agent = ACagent(policymodel, valuemodel)
#%%
agent.test(testenv,testingtime, nruns = 1) #200 timesteps
print('before training total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')
#%%
#    MWE of training
allrewards = []
for i in range(10):
    rewards = agent.train(testenv, 100)
#    plt.plot(rewards)
#    plt.ylabel('rewards')
#    plt.xlabel('episode')
    allrewards.extend(rewards)
    agent.test(testenv,testingtime,nruns=1)
    print('total reward is '+str(testenv.totloss)+' over '+str(len(testenv.sim[testenv.avid]))+' timesteps')
#%%
avtraj = np.asarray(testenv.sim[testenv.avid])
plt.figure() #plots, in order, position, speed, and headway time series.
plt.subplot(1,3,1)
plt.plot(avtraj[:,0])
plt.subplot(1,3,2)
plt.plot(avtraj[:,1])
plt.subplot(1,3,3)
plt.plot(avtraj[:,2])
#%%
#mcenv = gym.make('MountainCar-v0')
#mcagent = ACagent(PolicyModel(num_actions=mcenv.action_space.n), ValueModel())
#mctestenv = gym_env(mcenv)
#allmcrewards = []
#for i in range(4):
#    rewards = mcagent.train(mctestenv, 100)
##    plt.plot(rewards)
##    plt.ylabel('rewards')
##    plt.xlabel('episode')
#    allmcrewards.extend(rewards)
#    mcagent.test(mctestenv,1000,nruns=1)
#    print('total reward is '+str(mctestenv.totloss)+' over '+str(mcagent.counter)+' timesteps')
'''#%% Performance optimization

#For some set batch size (e.g. 64) time how long it takes to do that many steps in the simulate baseline method for the circular environment.
#testingtime = 64
#
#times=[]
#for _ in range(5):
#    start = time.time()
#    testenv.simulate_baseline(FS,[2,.4,.4,3,3,7,15,2], testingtime)
#    end = time.time()
#    times.append(end-start)
#print("Average over 5 runs is {:.4f}".format(np.mean(times)))   #0.0115
##


###For the same batch size, time how long it takes to do that many steps in training for cart pole.
#env = gym.make('CartPole-v0')
#agent = ACagent(PolicyModel(num_actions=env.action_space.n), ValueModel(num_hiddenlayers = 4))
#testenv = gym_env(env)
#
#times2=[]
#out2 = []
#for _ in range(5):
#    start = time.time()
#    agent.train(testenv, updates=1)
#    end = time.time()
#    times2.append(end-start)
#    out2.append(agent.timecounter)
#print("Average time over 5 runs is {:.4f}".format(np.mean(times2)))  #5.8304 eager
#print('average time to run environment step method in training is '+str(np.mean(out2)))
#testenv.reset()
#start = time.time()
#for i in range(64):
#    out = testenv.step(tf.convert_to_tensor(0), 0, 0, False)
#print('time to run environment step method outside training is '+str(time.time()-start))
#
##Using the same neural network for the agent, and same batch size, time how long it takes to do the training for the circular environment.
testenv = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25)
agent = ACagent(PolicyModel(num_actions=3), ValueModel())

times3=[]
out3 = []
for _ in range(5):
    start = time.time()
    agent.train(testenv, updates=1)
    end = time.time()
    times3.append(end-start)
    out3.append(agent.timecounter)
print("Average time over 5 runs is {:.4f}".format(np.mean(times3)))  #25.1353 eager
print('average time to run environment step method is '+str(np.mean(out3)))
testenv.reset()
start = time.time()
for i in range(64):
    out = testenv.step(tf.convert_to_tensor(0), 0, 0, False)
print('time to run environment step method outside training is '+str(time.time()-start))


#testenv = debugenv() #from toysimulation(1).py
#agent = ACagent(PolicyModel(num_actions= 3), ValueModel())
#times4=[]
#out4 = []
#for _ in range(5):
#    start = time.time()
#    agent.train(testenv, updates=1)
#    end = time.time()
#    times4.append(end-start)
#    out4.append(agent.timecounter)
#print("Average time over 5 runs is {:.4f}".format(np.mean(times4)))  #25.1353 eager
#print('average time to run environment step method is '+str(np.mean(out4)))
#testenv.reset()
#start = time.time()
#for i in range(64):
#    out = testenv.step(tf.convert_to_tensor(0), 0, 0, False)
#print('time to run environment step method outside training is is '+str(time.time()-start))
# 
