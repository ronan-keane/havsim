# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:27:16 2019
created as example for aglos assignment 1 
@author: rlk268
"""
#%% 
from ..havsim.simulation.simulationold import * 
from ..havsim.simulation.models import * 
import matplotlib.pyplot as plt
import numpy as np 


#example of doing simulation and plotting the result

dt = .1 
length = 5 
simlen = 400 #number of timesteps, can change
N = 10 #number of vehicles, can change

#specify model 
model = OVM_lb2 #model used; this is a variant on the optimal velocity model with a linear optimal velocity function, bounded acceleration, and bounded speed.  
#it's not really necessary for you to understand this model but you should know a few things about the parameters
p = [20,.7,1.4,60] #parameters for the model, the parameters represent 
#p[0] - your closest desired distance to the leader is p[0] feet, so if your leader has 0 speed, you should come to a stop p[0] feet from them
#p[1] - this parameter controls how much extra space you need based on your leader's speed; i.e. if your leader had speed 10, you would want to be 
#10/p[1] + p[0] feet from them. 
#p[2] - this controls how strongly you accelerate. A higher number means you will adjust to your desired speed more quickly. Note however that you are also bounded 
#by a maximum acceleration in this model; if your desired acceleration is > 13.3 or <-20, you will have 13.3 or -20 acceleration respectively. 
#p[3] - the maximum speed you can reach

#create the velocity for the lead vehicle; this is the disturbance all the vehicles will be reacting to 
velocity = [60 for i in range(simlen)]
velocity[0:200] = [1/4*(.1*(i)-10)**2+35 for i in range(200)] #polynomial disturbance based on a quadratic
velocity = np.asarray(velocity)
#initialize the lead vehicle and collection of all vehicles
v1 = leadvehicle(0,0,length,velocity,dt)
universe = [v1]

#now add N vehicles to the simulation 
headway = eql(model,60,p,length) #this function finds the equilibrium headway for a given speed; we use this to get the initial conditions 
prev = 0 
for i in range(N):
    prev = prev - headway
    newveh = vehicle([prev,60],0,length,universe[-1],model,p,.1)
    universe.append(newveh)
    
simulate(universe,simlen-1)

plt.close('all')
plt.figure()
vehplot(universe,interval = 2) #plot the simulation results, interval = 2 means we only plot every other vehicle
plt.figure()
hd(universe[2],universe[3]) #plot the trajectory in phase plane (speed versus headway) for vehicle 3, which follows vehicle 2. 
#this plots the distance each vehicle loses due to the disturbance. 
plt.figure()
displacementlist = []
for j in range(len(universe)):
    testveh = universe[j]
    testdx = testveh.dx
    dt = testveh.dt
    displacement = [0]
    for i in range(len(testdx)):
        d = (testdx[i]-60)*dt
        displacement.append(displacement[-1]+d)
        
    displacementlist.append(displacement[-1])
plt.plot(displacementlist, 'k.')

#%%
import scipy.optimize as sc 
#example of solving optimization 
#requires headway, simlen, N, velocity to be defined already 
testobj = egobj(p,p,OVM_lb2,headway,.1,simlen,N,velocity,60,.1) #this is the objective value below the optimization 
args = (p,OVM_lb2,headway,.1,simlen,N,velocity,60,.1)
bounds = [(5,40),(.3,2),(1,5),(40,60)]
bfgs = sc.fmin_l_bfgs_b(egobj,p,None,args,1,bounds,maxfun=100) #do the optimization 
#%%
#look at the result
print('initial objective function value was '+str(testobj)+ ' after optimization the objective function value was '+str(bfgs[1]))
universe, obj = eguni(bfgs[0],*args) #take the optimized parameter values and run the simulation with them, note that this will be different each run 
print('on this particular run of the simulation we obtained an objective of '+str(obj))
plt.figure()
vehplot(universe, interval = 2)


#%%
#minimal working example for using NL-Opt. Note I don't know how to compile and fully install Nlopt so if you can just figure out how to install it that would be a great progress. 
import nlopt as nlopt 
from scipy.optimize import rosen, rosen_der
def testnlopt():
    counter = 0 #note that this is how you are supposed to return func evals with nlopt.It doesn't keep track and only returns the answer in terms of parameters
    countergrad = 0
    def nltest(x,grad):
        nonlocal counter
        nonlocal countergrad
        if len(grad) > 0:
            countergrad += 1
            grad[:] = rosen_der(x)
            return grad
        counter += 1
        return rosen(x)
    
    opt = nlopt.opt(nlopt.GN_AGS,2) #change name here for different algorithms; DIRECT_L, ISRES, and others will work using pip install. AGS, STOGO, many others
    #will not work unless you successfully compile the C code on your machine 
#     opt = nlopt.opt(nlopt.GN_AGS,2) #this will not work because AGS function requires you to compile the C code or something IDK 
    opt.set_min_objective(nltest)
    opt.set_lower_bounds([-1,-1])
    opt.set_upper_bounds([1,1])
    opt.set_maxeval(1000)
    #
    test = opt.optimize([.5,-.5])
    
    return test, counter, countergrad
#execute above 
test, counter, countergrad = testnlopt()
