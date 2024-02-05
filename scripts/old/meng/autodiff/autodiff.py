# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:10:10 2019

@author: rlk268
"""
#from jax import grad #we want to use jax's grad function

#import numpy as np #note using np and scipy functions shouldn't work; you have to use the jax versions
import numpy as np
import pickle
import scipy.interpolate as sci
#first examples
def eg2(x, *args):
    return np.sum(np.square(x))

def eg1(x, *args):
    x[0] = x[0] // 2
    return np.tanh(x[0]**2)


"""
TO DO 
take the derivative of eg1 and eg2 using jax. The answer should be 2*x*(1/np.cosh(x)**2) for eg1, and sum(2*x) for eg2. 
check eg1 using x as a scalar. check eg2 using x as an ndarray 
"""

#example with a for loop which calls another function 
def eg3(p, *args):
    out = 0
    for i in range(len(p)):
        out += eg1([p[i]]) #note += is not allowed
    return out 
    
#example with a for loop and if else statements which depend on an outside datasource 
def eg4(p,testdata, *args):
    out = 0
    for i in range(len(p)):
        if testdata[i]<.5:
            out = out + eg2([p[i]])
        else: 
            out = out + eg1([p[i]])
    return out 

"""
take the derivative of eg3 and eg4 using jax. Send me your code (for eg1 - eg4) in an email when you complete this.
"""
#%%
#now we will test out how to get things like in place modification of arrays, and using slices etc. work. 

def eg5(p,testdata,*args):
    out = np.zeros((1))
    n = len(p)
    testdata2 = np.zeros((n))
    halfway = n // 2
    #note you can potentially get some out of bounds error here p should be ~len 10 -20 something in that range. or just make testdata longer
    testdata2[0:halfway] = testdata[10:10+halfway] #set data using slices...does this break things? 
    testdata2[halfway:] = testdata[35:35+halfway]
    
    out[0] = p[0] #in place setting of array
    
    out = np.append(out,testdata2) #concatenation of nd arrays
    out[1:] = out[1:] + p #more slices and also array addition
    
    return sum(out) #here we use default sum instead of np.sum
"""
#now we will try one last example, which is the most complicated one yet and is fairly close the function I ultimately want you to apply jax to. 
#if you can figure this out then the rest of the assignment should basically just be plug and chug, applying everything you've learned to the actual function I'm interested in.
"""



def finaleg(p,eg2, testdata2,testdata3, testdata4, testdata5, *args):
    #lead is basically an extra input we need for the simulation 
    #sim is where the simulation will go 
    #relax is another input to the simulation 
    #out is the objective which is based on the simulation output
    lead = np.zeros((testdata5))
    relax = np.zeros(testdata5)
    sim = np.zeros((testdata5))
    #first populate lead
    cur = 0
    for i in testdata3:
        curlen = i[2] - i[1]
        lead[cur:cur+curlen] = testdata2[i[0]][i[1]:i[2]]
        cur = cur+curlen
    #now get relax
    num = testdata4[0] // p[2]
    end = testdata4[0] - num*p[2]
    temp = np.linspace(testdata4[0],end,int(num+1))
    lenrelax = len(relax[5:])
    lentemp = len(temp)
    uselen = min(lenrelax,lentemp)
    relax[testdata4[1]:testdata4[1]+uselen] = temp[0:uselen]
        
    #now get the sim 
    sim[0] = lead[0]
    for i in range(testdata5-1):
        sim[i+1] = sim[i] + p[0]*eg2([lead[i],p[1],relax[i]])
        
    #return output
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

def eg7(p,X,Y,times,*args):
    #uses interp1ds which uses binaryint
    out = interp1ds(X,Y,times)
    out = p[0]*sum(out)+p[1]
    return out
#%% you can test your gradient is correct like this 

def fin_dif_wrapper(p,args, *eargs, eps = 1e-8, **kwargs):   
    #returns the gradient for function with call signature obj = objfun(p, *args)
    #note you should pass in 'objfun' as the last entry in the tuple for args
    #so objfun = args[-1]
    #uses first order forward difference with step size eps to compute the gradient 
    out = np.zeros((len(p),))
    objfun = args[-1]
    obj = objfun(p,*args)
    for i in range(len(out)):
        curp = p.copy()
        curp[i] += eps
        out[i] = objfun(curp,*args)
    return (out-obj)/eps

with open('/home/rlk268/Downloads/hav-sim-master(1)/hav-sim-master/autodiffeg.pkl','rb') as f:
    x1,x2,p,pfinal, testdata,testdata2,testdata3,testdata4,testdata5,X,Y,times,p7 = pickle.load(f)
    
    
#testobj1
x1[0] = 2.4


#get all objectives
obj1 = eg1(x1)
obj2 = eg2(x2)
obj3 = eg3(p)
obj4 = eg4(p,testdata)
obj5 = eg5(p,testdata)
obj6 = finaleg(pfinal,eg2,testdata2,testdata3,testdata4,testdata5)
obj7 = eg7(p7,X,Y,times)
print('obj1 = '+str(obj1))
print('obj2 = '+str(obj2))
print('obj3 = '+str(obj3))
print('obj4 = '+str(obj4))
print('obj5 = '+str(obj5))
print('obj6 = '+str(obj6))
print('obj7 = '+str(obj7))


#get all gradients using finite differences
fgrad1 = fin_dif_wrapper(x1,(0,eg1))
fgrad2 = fin_dif_wrapper(x2,(0,eg2))
fgrad3 = fin_dif_wrapper(p,(0,eg3))
fgrad4 = fin_dif_wrapper(p,(testdata,eg4))
fgrad5 = fin_dif_wrapper(p,(testdata,eg5))
fgrad6 = fin_dif_wrapper(pfinal,(eg2,testdata2,testdata3,testdata4,testdata5,finaleg))
fgrad7 = fin_dif_wrapper(p7,(X,Y,times,eg7))
print('grad = '+str(fgrad1))
print('grad = '+str(fgrad2))
print('grad = '+str(fgrad3))
print('grad = '+str(fgrad4))
print('grad = '+str(fgrad5))
print('grad = '+str(fgrad6))
print('grad = '+str(fgrad7))

"""
\\ TO DO \\
get gradient of all examples using jax

test the gradient is accurate by doing 
np.linalg.norm(jaxgrad1-fgrad1)/np.linalg.norm(fgrad1)
and 
np.divide(jaxgrad1-fgrad1,fgrad1)
"""

#%% ultimate end goal of the assignment 
#from calibration import platoonobjfn_obj, platoonobjfn_der, platoonobjfn_fder, makeleadfolinfo_r3,OVM, OVMadj, OVMadjsys, r_constant, euler, euleradj, shifted_end #all functinos from main file needed
#import time 
#import copy
#import pickle 
#import numpy as np
##load data you will need 
#with open('dataautodiff.pkl','rb') as f:
#    meas,platooninfo = pickle.load(f)
##define inputs needed 
#sim = copy.deepcopy(meas)
#pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5.01]
#args   = (True,6)
#curplatoon = [[],581,611]
#n = int(len(curplatoon)-1)
#leadinfo,folinfo,rinfo = makeleadfolinfo_r3(curplatoon,platooninfo,meas)
#p2 = np.tile(pguess,n)
##run the objective and time it
#start = time.time()
#obj = platoonobjfn_obj(p2,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#end = time.time()
#objtime = end-start
##get the gradient using adjoint method and time it
#start = time.time()
#der = platoonobjfn_der(p2,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#end = time.time()
#dertime = end-start
##get the gradient using finite differences
#start = time.time()
#fder = platoonobjfn_fder(p2,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#end = time.time()
#fdertime = end-start
##compare accurcy of finite difference gradient with adjoint method gradient 
#acc = np.linalg.norm(der-fder)/np.linalg.norm(fder)
#acc2 = np.divide(der-fder,fder)
#print('accuracy in norm is '+str(acc))
#print('relative error in each parameter is '+str(acc2))


#%% used this to create data for you 

#testmeas = {}
#testmeas[581] = meas[581].copy()
#testmeas[611] = meas[611].copy()
#testmeas[582] = meas[582].copy()
#testmeas[573] = meas[573].copy()
#platooninfo2 = {}
#platooninfo2[581] = platooninfo[581].copy()
#platooninfo2[611] = platooninfo[611].copy()
#platooninfo2[582] = platooninfo[582].copy()
#platooninfo2[573] = platooninfo[573].copy()
#
#with open('dataautodiff.pkl','wb') as f:
#    pickle.dump([testmeas,platooninfo2],f)