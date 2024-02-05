#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:16:16 2019

@author: rlk268
"""

import jax
from jax import grad, jit 
from jax.ops import *
import jax.numpy as jnp
import numpy as onp
import time 

#%%  #test forward euler in python lists and numpy for baseline comparison
#the program is meant to have a similar structure to solving a system of 20 ODEs,
#with some objective function which depends on the solution to the ODEs
def testfun(x):
    return [x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]]

def testtime(p): #using python lists
    allx = []
    for i in range(20):
        x = [p]
        curx = x[0]
        newx = [None,None]
        for i in range(200):
            out = testfun(curx)
            newx[0] = curx[0]+out[0]
            newx[1] = curx[1]+out[1]
            x.append(newx)
            curx = newx
        allx.append(x)
    obj = 0
    for i in range(len(allx)):
        cur = onp.array(allx[i])
        obj = obj + onp.mean(cur[:,0] - onp.ones((201,)))
    return obj

def testfunnp(x):
    return onp.array([x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]])

def testtime_np(p): #using numpy array
    allx = []
    for i in range(20):
        x = onp.zeros((201,2))
        x[0,:] = onp.array([1,1])
        curx = x[0,:]
        for j in range(200):
            out = testfunnp(curx)
            newx = curx+out
            x[j+1,:] = newx
            curx = newx
        allx.append(x)
    obj = 0
    for i in range(len(allx)):
        cur = onp.array(allx[i])
        obj = obj + onp.mean(cur[:,0] - onp.ones((201,)))
    return obj

def timeit(fun, p, n = 100): #times fun(p) for n repeitions
    start = time.time()
    for i in range(n):
        obj = fun(p)
    end = time.time()
    print('time for '+str(n)+' runs is '+str(round(end-start,2))) 
    
p = [1.,1.]
timeit(testtime,p)
timeit(testtime_np,p)

#%% test jax equivalents 
def jittimeit(fun,p): #times how long it takes to call a jitted function for the first time
    start = time.time()
    myobj = jit(fun)
    temp = myobj(p)
    end = time.time()
    print('time for first jitted objective call is '+str(round(end-start,2)))
    start = time.time()
    mygrad = jit(grad(fun))
    temp = mygrad(p)
    end = time.time()
    print('time for first jitted gradient call is '+str(round(end-start,2)))
    return myobj, mygrad
    
#def testtime1(p): 
#    allx = []
#    for i in range(20):
#        x = [p]
#        curx = x[0]
#        newx = [None,None]
#        for i in range(200):
#            out = testfun(curx)
#            newx[0] = curx[0]+out[0]
#            newx[1] = curx[1]+out[1]
#            x.append(newx)
#            curx = newx
#        allx.append(x)
#    obj = 0
#    for i in range(len(allx)):
#        cur = jnp.array(allx[i])
#        obj = obj + jnp.mean(cur[:,0] - jnp.ones((201,)))
#    return obj
#
#myobj, mygrad = jittimeit(testtime1,p)
#timeit(myobj,p)
#timeit(mygrad,p)
    #%% this test we have the parameters influence model instead of influencing IC
def testfun2(x,p):
    return [x[1],x[0]**.5+x[1]**.5+p[0]*x[1]+p[1]*x[0]]
    
def testtime2(p): 
    allx = []
    for i in range(20):
        x = [[1.,1.]]
        curx = x[0]
        newx = [None,None]
        for i in range(200):
            out = testfun2(curx,p)
            newx[0] = curx[0]+out[0]
            newx[1] = curx[1]+out[1]
            x.append(newx)
            curx = newx
        allx.append(x)
    obj = 0
    for i in range(len(allx)):
        cur = jnp.array(allx[i])
        obj = obj + jnp.mean(cur[:,0] - jnp.ones((201,)))
    return obj

p = [.01,.02]
myobj, mygrad = jittimeit(testtime2,p)
timeit(myobj,p)
timeit(mygrad,p)
    #%% jax numpy
    #warning - this crashes my computer - memory usage will reach 100%, at which point it will start writing to disk, and then it will become very slow,
    #will never finish, forcing a hard restart. Have not tested on google collab. 
#def testfunjnp(x):
#    return jnp.array([x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]])
#
#def testtime_np1(p): 
#    allx = []
#    for i in range(20):
#        x = jnp.zeros((201,2))
##        x[0,:] = jnp.array([1,1])
#        x = index_update(x, index[0,:],jnp.array([p[0],p[1]]))
#        curx = x[0,:]
#        for i in range(200):
#            out = testfunjnp(curx)
#            newx = curx+out
##            x[i+1,:] = newx
#            x = index_update(x,index[i+1,:],newx)
#            curx = newx
#        allx.append(x)
#    obj = 0
#    for i in range(len(allx)):
#        cur = jnp.array(allx[i])
#        obj = obj + jnp.mean(cur[:,0] - jnp.ones((201,)))
#    return obj
#
#myobj, mygrad = jittimeit(testtime_np1, p )

