
"""
@author: rlk268@cornell.edu
"""
import numpy as np
import scipy.stats as ss
import math
import matplotlib.pyplot as plt

def f(x, p):
    return p[0]*x + p[1]

def fdx(x, p):
    return 1+p[0]

def fdp(x, p):
    return np.array([x, 1, 0, 0,])

def g(p, r):
    # return np.random.normal(p[2], p[3])
    return ss.norm.ppf(r, loc= p[2], scale=p[3])

def pofg(p, x):
    return ss.norm.pdf(x, loc=p[2], scale=p[3])

def gdx(p, r):
    return 0

def pi(x, p, r):
    return ss.norm.ppf(r, loc= p[0]*x+p[1]+p[2], scale=p[3])

def pofpi(p, xx, x):
    return ss.norm.pdf(xx, loc= p[0]*x+p[1]+p[2], scale=p[3])

def pidp(p, xx, x):
    mu = p[0]*x + p[1] + p[2]
    sigma = p[3]
    temp = -(xx-mu)**2/(2*sigma**2)
    temp = math.e**temp
    temp1 = temp*(xx-mu)/((2*math.pi)**.5*sigma**3)
    temp2 = -temp/((2*math.pi)**.5*sigma**2) + temp*(xx-mu)**2/((2*math.pi)**.5*sigma**4)
    return np.array([temp1*x, temp1, temp1, temp2])

def gdp(p, x):
    mu = p[2]
    sigma = p[3]
    temp = -(x-mu)**2/(2*sigma**2)
    temp = math.e**temp
    temp1 = temp*(x-mu)/((2*math.pi)**.5*sigma**3)
    temp2 = -temp/((2*math.pi)**.5*sigma**2) + temp*(x-mu)**2/((2*math.pi)**.5*sigma**4)
    return np.array([0, 0, temp1, temp2])

def loss(y, yhat):
    return np.mean(np.square(y - yhat))

def lossdx(y, true):
    return 1/11*2*(y-true)


def obj(p, r, yhat):
    x = 1
    y = np.zeros((11,))
    y[0] = x
    for i in range(10):
        a = f(x, p) + g(p, r[i])
        x = x + a
        y[i] = x
    return loss(y, yhat)

def findiff(p, r, yhat):
    f1 = obj(p, r, yhat)
    pcopy = p.copy()
    grad = np.zeros((4,))
    for i in range(4):
        p[i] = p[i] + 1e-6
        grad[i] = (obj(p, r, yhat) - f1)/1e-6
        p[i] = pcopy[i]
    return f1, grad

def obj_and_grad(p, r, yhat):
    x = 1
    y = np.zeros((11,))
    y[0] = x
    glist = [0]
    for i in range(10):
        temp = g(p, r[i])
        glist.append(temp)
        a = f(x, p) + g(p, r[i])
        x = x + a
        y[i+1] = x

    lam = np.zeros((11,))
    lam[10] = -lossdx(y[10], yhat[10])
    for i in range(10):
        ind = 9-i
        lam[ind] = -lossdx(y[ind], yhat[ind]) + lam[ind+1]*fdx(y[ind], p)

    yy = np.square(y-yhat)/11
    v = np.zeros((10,))
    cur = 0
    for i in range(10):
        cur += yy[10-i]
        v[9-i] = cur
    grad = np.zeros((4,))
    for i in range(10):
        curg = pofg(p, glist[i+1])
        grad += gdp(p, glist[i+1])/(curg)*v[i] - lam[i]*fdp(y[i],p)
        # grad += gdp(p, glist[i+1])*v[i] - lam[i]*fdp(y[i],p)
        # grad += - lam[i]*fdp(y[i],p)

    return loss(y, yhat), grad


def policy_grad(p, r, yhat):
    x = 1
    y = np.zeros((11,))
    y[0] = x
    glist = [0]
    for i in range(10):
        a = pi(x, p, r[i])
        glist.append(a)
        x = x + a
        y[i+1] = x

    yy = np.square(y-yhat)/11
    v = np.zeros((10,))
    cur = 0
    for i in range(10):
        cur += yy[10-i]
        v[9-i] = cur

    grad = np.zeros((4,))
    for i in range(10):
        curg = pofpi(p, glist[i+1], y[i])
        grad += pidp(p, glist[i+1], y[i])/(curg+1e-6)*v[i]

    return loss(y, yhat), grad



#%%
yhat = np.array([1., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0.,])
p = np.array([-1., -1., 1., 1.])
pp = np.array([-1., -1., 1., 1.])
# p = np.array([-.5, .5, -.2, 1.5])
r = np.random.rand(10,)
lr = 5e-4
lr2 = 5e-4

objlist = []
for i in range(100):
    r = np.random.rand(10,)
    curobj = obj(p, r, yhat)
    objlist.append(curobj)
print('initial obj is '+str(np.mean(objlist)))

objlist = []
objlist2 = []
for i in range(500):
    r = np.random.rand(10,)
    # curobj, curgrad = findiff(p, r, yhat)
    curobj, curgrad = obj_and_grad(p, r, yhat)
    for j in range(9):
        r = np.random.rand(10,)
        curobj, curgrad2 = obj_and_grad(p, r, yhat)
        curgrad += curgrad2
    curobj2, curpgrad = policy_grad(pp, r, yhat)
    for j in range(9):
        r = np.random.rand(10,)
        curobj, curpgrad2 = obj_and_grad(p, r, yhat)
        curpgrad += curpgrad2
    p = p - lr*np.nan_to_num(curgrad)
    pp = pp - lr2*np.nan_to_num(curpgrad)
    if p[3] < 5e-2:
        p[3] = 5e-2
    if pp[3] < 5e-2:
        pp[3] = 5e-2
    objlist.append(curobj)
    objlist2.append(curobj2)

plt.plot(objlist)
plt.figure()
plt.plot(objlist2)

temp = []
for i in range(100):
    r = np.random.rand(10,)
    curobj = obj(pp, r, yhat)
    temp.append(curobj)
print('finalobj is '+str(np.mean(temp)))

#%%
gradlist = []
gradlist2 = []
for i in range(3000):
    r = np.random.rand(10,)
    # curobj, curgrad = findiff(p, r, yhat)
    curobj, curgrad = obj_and_grad(p, r, yhat)
    curobj2, curpgrad = policy_grad(p, r, yhat)
    gradlist.append(curgrad)
    gradlist2.append(curpgrad)
gradlist = np.array(gradlist)
gradlist2 = np.array(gradlist2)