#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:23:30 2019

@author: rlk268

test the speed of jax for taking gradient of functions which have similar construction to simulation 
"""
#imports 
import jax
from jax import grad, jit 
from jax.ops import *
import jax.numpy as jnp
import numpy as onp

#%%

def fin_dif_wrapper(p,args, *eargs, eps = 1e-4, **kwargs):   
    #returns the gradient for function with call signature obj = objfun(p, *args)
    #note you should pass in 'objfun' as the last entry in the tuple for args
    #so objfun = args[-1]
    #uses first order forward difference with step size eps to compute the gradient 
    out = np.zeros((len(p),))
    objfun = args[-1]
    #modified
    args = args[:-1]
    obj = objfun(p,*args)
    for i in range(len(out)):
        curp = p.copy()
        curp[i] += eps
        out[i] = objfun(curp,*args)
    return (out-obj)/eps


def test(p):
    out = []
    for i in p:
        out.append(i**2)
    out = jnp.array(out)
    return jnp.mean(out)*3

p = [1.,2.,3.]

testobj =test(p)

gradtest = grad(test)
testgrad = gradtest(p)


#surprisingly...it works? So default lists are supported by jax it looks like
#%%
#test a program that has the same structure as euler integration scheme
import time 

def testfun(x):
    return [x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]]

#def testnp(x):
#    return np.array([x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]])

def testtime1(p, *args): 
    allx = []
    obj = 0
    for i in range(20):
        x = [[p[0],p[1]]]
        curx = x[0]
        newx = [None,None]
        for i in range(200):
            out = testfun(curx)
            newx[0] = curx[0]+out[0]
            newx[1] = curx[1]+out[1]
            x.append(newx)
            curx = newx
        allx.append(x)
    for i in range(len(allx)):
        cur = jnp.array(allx[i])
        obj = obj + jnp.mean(cur[:,0] - jnp.ones((201,)))
    return obj

p= [1.,1.]
testtimejit = jit(testtime1)
start = time.time()
testobj = testtimejit(p)
end = time.time()
objtime = end-start


gradtest = jit(grad(testtime1))
start = time.time()
testgrad = gradtest(p)
end = time.time()
gradtime = end-start

testfin = fin_dif_wrapper(p,(0,testtime1))
testfin = jnp.array(testfin)
print('obj is '+str(testobj)+' calculated in '+str(objtime))
print('jax grad is '+str(testgrad)+' calculated in '+str(gradtime))
print('normed residual is '+str(jnp.array([testgrad[0]-testfin[0],testgrad[1]-testfin[1]]/jnp.linalg.norm(testfin))))

#%% test speed of jitted functions 
#why is the objective so much slower here???
#maybe an issue that can be raised on the jax issues 
start = time.time()
for i in range(100):
    out = testtimejit(p)
end = time.time()
objtime = end-start

start = time.time()
for i in range(100):
    out = gradtest(p)
end = time.time()
gradtime = end-start


#%%
#test a program in jax when it is using in place assignments in numpy

def testnp(x):
    return jnp.array([x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]])

def testtime31(p,*args): 
    allx = []
    for i in range(20):
        x = jnp.zeros((201,2))
#        x[0,:] = jnp.array([1,1])
        x = index_update(x, index[0,:],jnp.array([p[0],p[1]]))
        curx = x[0,:]
        for i in range(200):
            out = testnp(curx)
            newx = curx+out
#            x[i+1,:] = newx
            x = index_update(x,index[i+1,:],newx)
            curx = newx
        allx.append(x)
        
    if True: 
        obj = 0
        for i in range(len(allx)):
            cur = jnp.array(allx[i])
            obj = obj + jnp.mean(cur[:,0] - jnp.ones((201,)))
    return obj
p = [1.,1.]
testtime31jit = jit(testtime31)
start = time.time()
#for i in range(100):
testobj = testtime31jit(p)
end = time.time()
print('obj time is '+str(end-start))

gradtest = jit(grad(testtime31))
start = time.time()
#for i in range(100):
testgrad = gradtest(p)
end = time.time()
print('grad time is '+str(end-start))

testfin = fin_dif_wrapper(p,(0,testtime31))