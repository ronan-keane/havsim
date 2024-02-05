# -*- coding: utf-8 -*-

"""

Created on Sun Sep 22 20:10:10 2019



@author: rlk268

"""

from jax import grad #we want to use jax's grad function



import numpy as np #note using np and scipy functions shouldn't work; you have to use the jax versions

import jax.numpy as jnp

import jax

from jax.ops import *

from jax import jit
import pickle
from jax import lax 
import copy 


from jax.config import config

config.update("jax_enable_x64", True) #if you want float 64 in jax this is the command 

#%% examples 
def intdiv(x,n):
    #integer divide x by n 
    out = x / n
    out = jnp.floor(out)
    return out

def eg1(x, *args):
#    x[0] = x[0] // 2
#    x = index_update(x,0,intdiv(x[0],2))
    x[0] = x[0] // 2
    return jnp.tanh(x[0] ** 2)


def eg2(x, *args):

    return jnp.sum(jnp.square(x))



def eg3(p, *args):

    out = 0

    for i in range(len(p)):

        out = out + eg1([p[i]]) #note += is not allowed

    return out 



def eg4(p,testdata, *args):

    out = 0

    for i in range(len(p)):

        if testdata[i]<.5:

            out = out + eg2(jnp.array([p[i]]))

        else: 

            out = out + eg1([p[i]])

    return out 
 
#Code for a jax friendly linspace graciously provided by levskaya, contributor to jax package 
    #note that this functino uses both ordinary and jax numpy. Be careful. 
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, 
             axis=0):
  """Implementation of linspace differentiable w.r.t. start and stop args."""
  lax._check_user_dtype_supported(dtype, "linspace")
  dtype = np.float32 if dtype is None else dtype
  bounds_shape = list(lax.broadcast_shapes(np.shape(start), np.shape(stop)))
  axis = len(bounds_shape) if axis == -1 else axis
  bounds_shape.insert(axis, 1)
  iota_shape = [1,] * len(bounds_shape)
  iota_shape[axis] = num
  delta = (stop - start) / num
  if endpoint:
    delta *= num / (num - 1)
  out = (jnp.reshape(start, bounds_shape) + 
         jnp.reshape(lax.iota(dtype, num), iota_shape) * 
         jnp.reshape(delta, bounds_shape))
  if retstep:
    return jnp.array(out, dtype=dtype), delta
  else:
    return jnp.array(out, dtype=dtype)

def eg5(p,testdata,*args):

    out = jnp.zeros((1))

    n = len(p)

    testdata2 = jnp.zeros((n))

    halfway = n // 2

    #note you can potentially get some out of bounds error here p should be ~len 10 -20 something in that range. or just make testdata longer

    #testdata2[0:halfway] = testdata[10:10+halfway] #set data using slices...does this break things?

    testdata2 = index_update(testdata2, index[0:halfway], testdata[10:10+halfway])

    #testdata2[halfway:] = testdata[35:35+halfway]

    testdata2 = index_update(testdata2, index[halfway:], testdata[35:35+halfway])


    #out[0] = p[0] #in place setting of array

    out = index_update(out, index[0], p[0])
    

    out = jnp.append(out, testdata2) #concatenation of nd arrays

    #out[1:] = out[1:] + p #more slices and also array addition

    out = index_update(out, index[1:], out[1:]+p)

    return sum(out) #here we use default sum instead of np.sum


def finaleg(p,eg2, testdata2,testdata3, testdata4, testdata5, *args):

    #lead is basically an extra input we need for the simulation 

    #sim is where the simulation will go 

    #relax is another input to the simulation 

    #out is the objective which is based on the simulation output

    lead = jnp.zeros(testdata5)

    relax = jnp.zeros(testdata5)

    sim = jnp.zeros(testdata5)





    # lead = [0 for _ in range(testdata5)]

    # relax = [0 for _ in range(testdata5)]

    #first populate lead

    cur = 0

    for i in testdata3:

        curlen = i[2] - i[1]

        # lead[cur:cur+curlen] = testdata2[i[0]][i[1]:i[2]]

        lead = index_update(lead, index[cur:cur+curlen], testdata2[i[0]][i[1]:i[2]])

        cur = cur+curlen

    #now get relax

    num = testdata4[0] // p[2]
#    print(num)
#    temp = copy.deepcopy(num)

    end = testdata4[0] - num*p[2]

    temp = linspace(testdata4[0], end, int(num + 1))

    lenrelax = len(relax[5:])

    lentemp = len(temp)

    uselen = min(lenrelax,lentemp)

    # relax[testdata4[1]:testdata4[1]+uselen] = temp[0:uselen]

    relax = index_update(relax, index[testdata4[1]:testdata4[1]+uselen], temp[0:uselen])

        

    #now get the sim 

    #sim[0] = lead[0]

    sim = index_update(sim, index[0], lead[0])

    for i in range(testdata5-1):

        #sim[i+1] = sim[i] + p[0]*eg2([lead[i],p[1],relax[i]])

        sim = index_update(sim, index[i+1], sim[i] + p[0]*eg2(jnp.array([lead[i],p[1],relax[i]])))

        

    #return output

    out = lead - sim

    return sum(out)


def finaleg_jax(p, eg2, testdata2, testdata3, testdata4, testdata5, *args):
    # lead is basically an extra input we need for the simulation

    # sim is where the simulation will go

    # relax is another input to the simulation

    # out is the objective which is based on the simulation output

    lead = jnp.zeros(testdata5)

    relax = jnp.zeros(testdata5)

    sim = jnp.zeros(testdata5)

    # lead = [0 for _ in range(testdata5)]

    # relax = [0 for _ in range(testdata5)]

    # first populate lead

    cur = 0

    for i in testdata3:
        curlen = i[2] - i[1]

        # lead[cur:cur+curlen] = testdata2[i[0]][i[1]:i[2]]

        lead = index_update(lead, index[cur:cur + curlen], testdata2[i[0]][i[1]:i[2]])

        cur = cur + curlen

    # now get relax

    num = testdata4[0] // p[2] #old 
    """
    originally thought integer division was doing bad things. Actually I think that part is fine, the problem is linspace can't handle the gradient in this
    case because the inputs to linspace are depending on the parameters. Interesting it didn't throw an error, but this is a good illustration of why 
    its super important to write the code in regular python and differentiate using finite differneces/adjoint method first before
    then converting to auto diff. Because otherwise its very easy to miss things which go wrong!
    """
#    #test manually doing integer division using numpy.floor
#    num = testdata4[0] / p[2]
#    num = jnp.floor(num)
#    #results of test - doesn't work still
    
    #test only doing division
#    num = testdata4[0]/p[2]
    #throws typeerror ``abstract value passed to 'int' '' 

    end = testdata4[0] - num * p[2]

    temp = jnp.linspace(testdata4[0], end.primal, int(num + 1))

    lenrelax = len(relax[5:])

    lentemp = len(temp)

    uselen = min(lenrelax, lentemp)

    # relax[testdata4[1]:testdata4[1]+uselen] = temp[0:uselen]

    relax = index_update(relax, index[testdata4[1]:testdata4[1] + uselen], temp[0:uselen])

    # now get the sim

    # sim[0] = lead[0]

    sim = index_update(sim, index[0], lead[0])

    for i in range(testdata5 - 1):
        # sim[i+1] = sim[i] + p[0]*eg2([lead[i],p[1],relax[i]])

        #modified
        # sim = index_update(sim, index[i + 1], sim[i] + p[0].primal.pval[1] * eg2(np.array([lead[i], p[1].primal.pval[1], relax[i]])))
        sim = index_update(sim, index[i+1], sim[i] + p[0]*eg2(jnp.array([lead[i],p[1],relax[i]])))
    # return output

    out = lead - sim

    return sum(out)

def interp1ds(X,Y,times):
    #given time series data X, Y (such that each (X[i], Y[i]) tuple is an observation), 
    #and the array times, interpolates the data onto times. 
    
    #X, Y, and times all need to be sorted in terms of increasing time. X and times need to have a constant time discretization
    #runtime is O(n+m) where X is len(n) and times is len(m)
    
    #uses 1d interpolation. This is similar to the functionality of scipy.interp1d. Name is interp1ds because it does linear interpolation in 1d (interp1d) on sorted data (s)
    
    #e.g. X = [1,2,3,4,5]
    #Y = [2,3,4,5,6]
    #times = [3.25,4,4.75]
    
    #out = [4.25,5,5.75]
    
    if times[0] < X[0] or times[-1] > X[-1]:
        print('Error: requested times are outside measurements')
        return None

    Xdt = X[1] - X[0]
    timesdt = times[1]-times[0]
    change = timesdt/Xdt
    
    m = binaryint(X,times[0])
    out = np.zeros(len(times))
    curind = m + (times[0]-X[m])/Xdt
    
    leftover = curind % 1
    out[0] = Y[m] + leftover*(Y[m+1]-Y[m])
    
    for i in range(len(times)-1):
        curind = curind + change #update index
        
        leftover = curind % 1 #leftover needed for interpolation
        ind = int(curind // 1) #cast to int because it is casted to float automatically 
        
        out[i+1] = Y[ind] + leftover*(Y[ind+1]-Y[ind])
    
    return out
    

def binaryint(X,time): 
    #finds index m such that the interval X[m], X[m+1] contains time. 
    #X = array 
    #time = float
    lo = 0 
    hi = len(X)-2
    m = (lo + hi) //  2
    while (hi - lo) > 1: 
        if time < X[m]:
            hi = m
        else: 
            lo = m
        m = (lo + hi) // 2
    return lo 

def eg7(p,X,Y,times):
    #uses interp1ds which uses binaryint
    out = interp1ds(X,Y,times)
    out = p[0]*sum(out)+p[1]
    return out



#%% you can test your gradient is correct like this 

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

with open('/home/rlk268/Downloads/hav-sim-master(1)/hav-sim-master/autodiffeg.pkl','rb') as f:
    x1,x2,p,pfinal, testdata,testdata2,testdata3,testdata4,testdata5,X,Y,times,p7 = pickle.load(f)
    
    
#modify pfinal for test 
#pfinal[2] = 5
    
#testobj1
x1[0] = 2.4
#woops I accidently broke obj3 and 4, it's not important don't worry about it 

#get all objectives
obj1 = eg1(x1)
obj2 = eg2(x2)
obj3 = eg3(p)
obj4 = eg4(p,testdata)
obj5 = eg5(p,testdata)
obj6 = finaleg(pfinal,eg2,testdata2,testdata3,testdata4,testdata5)
obj7 = eg7(p7,X,Y,times)

#get all gradients using finite differences
fgrad1 = fin_dif_wrapper(x1,(0,eg1))
fgrad2 = fin_dif_wrapper(x2,(0,eg2))
fgrad3 = fin_dif_wrapper(p,(0,eg3))
fgrad4 = fin_dif_wrapper(p,(testdata,eg4))
fgrad5 = fin_dif_wrapper(p,(testdata,eg5))
fgrad6 = fin_dif_wrapper(pfinal,(eg2,testdata2,testdata3,testdata4,testdata5,finaleg))
fgrad7 = fin_dif_wrapper(p7,(X,Y,times,eg7))

print('obj1 = '+str(obj1))
print('obj2 = '+str(obj2))
print('obj3 = '+str(obj3))
print('obj4 = '+str(obj4))
print('obj5 = '+str(obj5))
print('obj6 = '+str(obj6))
print('obj7 = '+str(obj7))

def getDiff(grad1, grad2):
    print("jax_grad:", grad1)
    print("fgrad:", grad2)
    print("differences are:")
    print(np.linalg.norm(grad1 - grad2) / np.linalg.norm(grad2))
    print(np.divide(grad1 - grad2, grad2))
    print()

x1 = list(map(float, x1))
jaxgrad1 = grad(eg1)(x1)[0]
print("eg1")
getDiff(jaxgrad1, fgrad1)

jaxgrad2 = grad(eg2)(x2)
print("eg2")
getDiff(jaxgrad2, fgrad2)

jaxgrad3 = grad(eg3)(p)
print("eg3")
getDiff(jaxgrad3, fgrad3)

jaxgrad4 = grad(eg4)(p, testdata)
print("eg4")
getDiff(jaxgrad4, fgrad4)

jaxgrad5 = grad(eg5)(p, testdata)
print("eg5")
getDiff(jaxgrad5, fgrad5)


pfinal = list(map(float, pfinal))



# testdata4 = list(map(float,testdata4))


jaxgrad6 = grad(finaleg_jax)(pfinal, eg2,testdata2,testdata3,testdata4,testdata5)
print("finaleg")
getDiff(jaxgrad6, fgrad6)

p7 = list(map(float, p7))
jaxgrad7 = grad(eg7)(p7, X,Y,times)
print("eg7")
getDiff(jaxgrad7,fgrad7)


"""
\\ TO DO \\
get gradient of all examples using jax

test the gradient is accurate by doing 
np.linalg.norm(jaxgrad1-fgrad1)/np.linalg.norm(fgrad1)
and 
np.divide(jaxgrad1-fgrad1,fgrad1)
"""

#%%
#test only finaleg6
##this version is working!!!!! :)
obj6 = finaleg(pfinal,eg2,testdata2,testdata3,testdata4,testdata5)
print(obj6)
fgrad6 = fin_dif_wrapper(pfinal,(eg2,testdata2,testdata3,testdata4,testdata5,finaleg))
jaxgrad6 = grad(finaleg)(pfinal, eg2,testdata2,testdata3,testdata4,testdata5)
print("finaleg")
getDiff(jaxgrad6, fgrad6)


#%%
#extra tests by ronan - moved to fall 2019 folder for random tests 

