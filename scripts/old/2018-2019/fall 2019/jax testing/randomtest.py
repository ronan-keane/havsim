
"""
@author: rlk268@cornell.edu

test if it makes a difference how we do the integration for simulation
tested 4 different ways: 
    vectorized python list ~.3s
    sequential python list ~.3s
    vectorized numpy array ~1.8s
    sequential numpy array ~3.6s
    
python list is using regular list structure for integrating, 
numpy array is using numpy array. 
vectorized is doing all vehicles at the same time, sequential is doing vehicles one at a time. 

\\TO DO\\
would like to test this in jax as well so we can figure out what the best way to integrate stuff in jax is. 

There are also some tests in autodiffjax which are similar to this 
"""

import time
import numpy as np 

def testfun(x):
    return [x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]]

def testnp(x):
    return np.array([x[1],x[0]**.5+x[1]**.5+.01*x[1]+.02*x[0]])

def testtime1(): 
    allx = []
    for i in range(20):
        x = [[1,1]]
        curx = x[0]
        newx = [None,None]
        for i in range(200):
            out = testfun(curx)
            newx[0] = curx[0]+out[0]
            newx[1] = curx[1]+out[1]
            x.append(newx)
            curx = newx
        allx.append(x)
    if True: 
        obj = 0
        for i in range(len(allx)):
            cur = np.array(allx[i])
            obj = obj + np.mean(cur[:,0] - np.ones((201,)))
    
        
    return allx
            
start = time.time()
for i in range(100):
    testtime1()
end=  time.time()
print('in sequential code the time was '+str(end-start))
#%%
def testtime2():
    allx = []
    curx = [[1,1] for i in range(20)]
    allx.append(curx)
    newx = [[None, None] for i in range(20)]
    for i in range(200):
        for j in range(20):
            out = testfun(curx[j])
            newx[j][0] = curx[j][0]+out[0]
            newx[j][1] = curx[j][1] + out[1]
        allx.append(newx)
        curx = newx
    return allx

start = time.time()
for j in range(100):
    testtime2()
end = time.time()
print('in vectorized code the time was '+str(end-start))

def testtime11(): 
    allx = []
    for i in range(20):
        x = np.array([1,1])
        curx = x[:]
        for i in range(200):
            out = testnp(curx)
            newx = curx+out
            x = np.append(x,newx,axis=0)
            curx = newx
        allx.append(x)
    return allx
            
start = time.time()
for i in range(100):
    testtime11()
end=  time.time()
print('in sequential code the time was '+str(end-start))

def testtime21():
    curx = [[1,1] for i in range(20)]
    curx = np.asarray(curx)
    newx = curx.copy()
    allx = curx.copy()
    for i in range(200):
        for j in range(20):
            newx[j,:] = testnp(curx[j,:])
        newx = newx + curx
        allx = np.append(allx,newx,axis=0)
        curx = newx
    return allx

start = time.time()
for j in range(100):
    testtime21()
end = time.time()
print('in vectorized code the time was '+str(end-start))

#%%
#test using in place index assignment in numpy 
def testtime31(): 
    allx = []
    for i in range(20):
        x = np.zeros((201,2))
        x[0,:] = np.array([1,1])
        curx = x[0,:]
        for i in range(200):
            out = testnp(curx)
            newx = curx+out
            x[i+1,:] = newx
            curx = newx
        allx.append(x)
        
    if True: 
        obj = 0
        for i in range(len(allx)):
            cur = np.array(allx[i])
            obj = obj + np.mean(cur[:,0] - np.ones((201,)))
    return allx

start = time.time()
for j in range(100):
    testtime31()
end = time.time()
print('in place numpy code the time was '+str(end-start))