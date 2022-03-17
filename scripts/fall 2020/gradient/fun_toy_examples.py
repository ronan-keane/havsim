
"""
@author: rlk268@cornell.edu
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import numpy as np
import math
#%%
# example 1 => For a discrete system, all timesteps have same regime = good, we are lipschitz continuous.
# some timesteps change regime = somewhat bad. many timesteps change regime = very bad.
timesteps = 20
prange = (.4,1.6)

def fun(x, p):
    return p if x <= 0 else -1

def solution(p, timesteps=timesteps):
    x = 0
    out = [x]
    for i in range(timesteps):
        x = x + fun(x, p)
        out.append(x)
    return out

def objective(out):
    return sum(out)

def obj_and_grad(p):
    obj = objective(solution(p))
    return obj, (objective(solution(p+1e-8))-obj)/1e-8

#%%
# show system
fig = plt.figure(figsize=(10,10))
ax = plt.subplot()
plt.subplots_adjust(bottom = .2)
x = list(range(timesteps))
out = solution(.99)
artist, = ax.plot(out)
ax.plot((0, timesteps), (0, 0))
artist2, = ax.plot(x, out[:-1], 'k.')
ax.set_ylim([-1, prange[1]+.1])
ax.set_ylabel('x(t)')
ax.set_xlabel('t')
ax.set_xticks(list(range(5,timesteps+1,5)))
ax.set_xlabel
axp = plt.axes([.15, 0.1, 0.65, 0.03])

def update(val):
    out = solution(val)
    artist.set_ydata(out)
    artist2.set_ydata(out[:-1])
    fig.canvas.draw_idle()

p_values = Slider(axp, 'p', prange[0], prange[1], valfmt='%.9f', valinit=.99)
p_values.on_changed(update)

#%%
# show system + objective + gradient
gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(gs[0,:])
ax2, ax3 = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
plt.subplots_adjust(bottom = .2)
x2 = np.linspace(prange[0], prange[1], 10000)
objlist = []
gradlist = []
for p in x2:
    obj, grad = obj_and_grad(p)
    objlist.append(obj)
    gradlist.append(grad)
obj, grad = obj_and_grad(.99)
x = list(range(timesteps))
out = solution(.99)
artist, = ax.plot(out)
ax.plot((0, timesteps), (0, 0))
artist2, = ax.plot(x, out[:-1], 'k.')
ax2.plot(x2, objlist, 'k.', markersize=2)
ax3.plot(x2, gradlist, 'k.', markersize=2)
artist3, = ax2.plot(.99, obj, 'r.')
artist4, = ax3.plot(.99, grad, 'r.')
ax2.set_ylabel('objective')
ax3.set_ylabel('gradient')
ax3.set_ylim([50, 150])
ax.set_ylim([-1, prange[1]+.1])
ax.set_ylabel('x(t)')
ax.set_xlabel('t')
ax.set_xticks(list(range(5,timesteps+1,5)))
ax.set_xlabel
axp = plt.axes([.15, 0.1, 0.65, 0.03])

def update(val):
    out = solution(val)
    artist.set_ydata(out)
    artist2.set_ydata(out[:-1])
    obj, grad = obj_and_grad(val)
    artist3.set_xdata(val)
    artist4.set_xdata(val)
    artist3.set_ydata(obj)
    artist4.set_ydata(grad)
    fig.canvas.draw_idle()

p_values = Slider(axp, 'p', prange[0], prange[1], valfmt='%.9f', valinit=.99)
p_values.on_changed(update)

#%%
# example 2 => no sensitivity = bad. example where continuous system has continuous obj/grad (for 0=<p<2) and discrete system has 0 gradient everywhere.
# continuous
prange = (.5, 2)
pinit = 1.99
def solution(p):
    if p > 2:
        x = np.linspace(0, 2*math.pi, 1000)
        y = 2*np.sin(x)
        return x, y
    point = math.asin(p/2)
    x1 = np.linspace(0, point, 1000)
    x2 = np.linspace(point, 2*math.pi, 1000)
    y1 = 2*np.sin(x1)
    y2 = p + 2*math.cos(point)*(x2 - point)
    x = np.append(x1, x2, axis=0)
    y = np.append(y1, y2, axis=0)
    return x, y

def obj_and_grad(p):
    if p <=2:
        star = math.asin(p/2)
        obj = 2 - 2*math.cos(star) +(2*math.pi-star)*(p-2*math.cos(star)*star)+math.cos(star)*(4*math.pi**2-star**2)
    else:
        obj = 0
    if p == 2:
        grad = math.nan
    elif p > 2:
        grad = 0
    else:
        ds = 1/2/(1-p**2/4)**.5
        grad = 2*math.sin(star)*ds + -ds*(p-2*math.cos(star)*star) + (2*math.pi-star)*(2*math.sin(star)*star*ds+1-2*math.cos(star)*ds) \
            -math.sin(star)*ds*(4*math.pi**2-star**2)+math.cos(star)*2*star*ds
    return obj, grad

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(gs[0,:])
ax2, ax3 = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
plt.subplots_adjust(bottom = .2)
x2 = np.linspace(prange[0], prange[1], 10000)
objlist = []
gradlist = []
for p in x2:
    obj, grad = obj_and_grad(p)
    objlist.append(obj)
    gradlist.append(grad)
obj, grad = obj_and_grad(pinit)
x, out = solution(pinit)
artist, = ax.plot(x, out)
artist2, = ax.plot((0, 2*math.pi), (pinit, pinit), 'C0--', linewidth=1, alpha=.2)
ax2.plot(x2, objlist)
ax3.plot(x2, gradlist)
artist3, = ax2.plot(pinit, obj, 'r.')
artist4, = ax3.plot(pinit, grad, 'r.')
ax2.set_ylabel('objective')
ax3.set_ylabel('gradient')
ax.set_ylabel('x(t)')
ax.set_xlabel('t')
ax.set_ylim([-2,10])
ax3.set_ylim([-200,10])
axp = plt.axes([.15, 0.1, 0.65, 0.03])

def update(val):
    x, y = solution(val)
    artist.set_xdata(x)
    artist.set_ydata(y)
    artist2.set_ydata((val, val))
    obj, grad = obj_and_grad(val)
    artist3.set_xdata(val)
    artist4.set_xdata(val)
    artist3.set_ydata(obj)
    artist4.set_ydata(grad)
    fig.canvas.draw_idle()

p_values = Slider(axp, 'p', prange[0], prange[1]+.2, valfmt='%.9f', valinit=pinit)
p_values.on_changed(update)

#%% example 2 for discrete
prange = (.5, 2)
pinit = 1.99
timesteps = 100
x = np.linspace(0,2*math.pi, timesteps+1)

def fun(x, p):
    if x < p:
        return -x
    else:
        return 0

def solution2(p, dt=2*math.pi/timesteps):
    x = [0, 2]
    out = [x[0]]
    for i in range(timesteps):
         dx = fun(x[0], p)
         x[0] = x[0]+dt*x[1]
         x[1] = x[1] + dt*dx
         out.append(x[0])
    return out

def objective2(out, dt=2*math.pi/timesteps):
    return sum(out)*dt

def obj_and_grad2(p):
    obj = objective2(solution2(p))
    return obj, (objective2(solution2(p+1e-8))-obj)/1e-8

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(gs[0,:])
ax2, ax3 = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
plt.subplots_adjust(bottom = .2)
x2 = np.linspace(prange[0], prange[1], 10000)
objlist = []
gradlist = []
for p in x2:
    obj, grad = obj_and_grad2(p)
    objlist.append(obj)
    gradlist.append(grad)
obj, grad = obj_and_grad2(pinit)
out = solution2(pinit)
artist1, = ax.plot(x, out)
artist5, = ax.plot((0, 2*math.pi), (pinit, pinit), 'C0--', alpha=.2)
artist22, = ax.plot(x, out, 'k.')
ax2.plot(x2, objlist, 'k.', markersize=2)
ax3.plot(x2, gradlist, 'k.', markersize=2)
artist33, = ax2.plot(pinit, obj, 'r.')
artist44, = ax3.plot(pinit, grad, 'r.')
ax.set_ylim([-2, 10])
ax2.set_ylabel('objective')
ax3.set_ylabel('gradient')
ax.set_ylabel('x(t)')
ax.set_xlabel('t')
ax.set_xlabel
axp2 = plt.axes([.15, 0.1, 0.65, 0.03])

def update(val):
    out = solution2(val)
    artist1.set_ydata(out)
    artist22.set_ydata(out)
    obj, grad = obj_and_grad2(val)
    artist33.set_xdata(val)
    artist44.set_xdata(val)
    artist33.set_ydata(obj)
    artist44.set_ydata(grad)
    artist5.set_ydata((val,val))
    fig.canvas.draw_idle()

p_values2 = Slider(axp2, 'p', prange[0], prange[1]+.2, valfmt='%.9f', valinit=pinit)
p_values2.on_changed(update)

#%%
# example 3 - example of state dependent discrete system where we are happy (objective is lipschitz continuous, quasiconvex)
prange = (0, 1)
pinit = .2
timesteps = 100
x = np.linspace(0,2, timesteps+1)

def fun(x, p):
    return -p*x if x > 2 else -p*x-2

def solution2(p, timesteps=timesteps, dt = 2/timesteps):
    x = 3
    out = [x]
    for i in range(timesteps):
        x = x + dt*fun(x, p)
        out.append(x)
    return out

def objective2(out):
    return sum(out)*2/timesteps

def obj_and_grad2(p):
    obj = objective2(solution2(p))
    return obj, (objective2(solution2(p+1e-8))-obj)/1e-8

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(gs[0,:])
ax2, ax3 = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
plt.subplots_adjust(bottom = .2)
x2 = np.linspace(prange[0], prange[1], 10000)
objlist = []
gradlist = []
for p in x2:
    obj, grad = obj_and_grad2(p)
    objlist.append(obj)
    gradlist.append(grad)
obj, grad = obj_and_grad2(pinit)
out = solution2(pinit)
artist1, = ax.plot(x, out)
artist5, = ax.plot((0, 2), (2, 2), 'C0--', alpha=.2)
artist22, = ax.plot(x, out, 'k.')
ax2.plot(x2, objlist, 'k.', markersize=2)
ax3.plot(x2, gradlist, 'k.', markersize=2)
artist33, = ax2.plot(pinit, obj, 'r.')
artist44, = ax3.plot(pinit, grad, 'r.')
ax2.set_ylabel('objective')
ax3.set_ylabel('gradient')
ax.set_ylabel('x(t)')
ax.set_xlabel('t')
ax.set_xlabel
ax.set_ylim([-2, 3.2])
axp2 = plt.axes([.15, 0.1, 0.65, 0.03])

def update(val):
    out = solution2(val)
    artist1.set_ydata(out)
    artist22.set_ydata(out)
    obj, grad = obj_and_grad2(val)
    artist33.set_xdata(val)
    artist44.set_xdata(val)
    artist33.set_ydata(obj)
    artist44.set_ydata(grad)
    fig.canvas.draw_idle()

p_values2 = Slider(axp2, 'p', prange[0], prange[1]+.2, valfmt='%.9f', valinit=pinit)
p_values2.on_changed(update)

#%% Real, not toy, example. Want actual models to be like example 3 - many models already seem to have that property
from havsim.old.opt import platoonobjfn_objder
from havsim.old.helper import makeleadfolinfo
from havsim.old.models import daganzo, daganzoadjsys, daganzoadj
import scipy.optimize as sc
import copy
import pickle
import havsim

# with open('/home/rlk268/havsim/data/recon-ngsim-old.pkl', 'rb') as f:
#     meas, platooninfo = pickle.load(f)
with open('C:/Users/rlk268/OneDrive - Cornell University/havsim/data/recon-ngsim-old.pkl', 'rb') as f:
    meas, platooninfo = pickle.load(f)

model, modeladjsys, modeladj = daganzo, daganzoadjsys, daganzoadj
pguess, mybounds = [1, 10, 30,5], [(.1,10),(0,100),(20,100),(.1,75)]
curveh = [995]
t_nstar, t_n, T_nm1, T_n = platooninfo[curveh[0]][:4]
end = T_nm1-t_nstar
leadinfo, folinfo, rinfo = makeleadfolinfo(curveh, platooninfo,meas)
args = (True,4)
sim = copy.deepcopy(meas)
bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(model, modeladjsys, modeladj, meas, sim, platooninfo, curveh, leadinfo, folinfo,rinfo,*args),0,mybounds,maxfun=200)
# plot the calibrated trajectory
sim[curveh[0]][:-1,3] = (sim[curveh[0]][1:,2] - sim[curveh[0]][:-1,2])/.1
havsim.plotting.plotvhd(meas, sim, platooninfo, curveh)

#%%
pinit = [2.08804351, 16.44383503, 32.14563722, 7.62970812]
p = pinit.copy()
def helper(p, rediff=True, process_regime=True):
    obj, grad, regimes = platoonobjfn_objder(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curveh, leadinfo, folinfo,rinfo, *args, return_regime=True)
    if rediff:
        sim[curveh[0]][:-1,3] = (sim[curveh[0]][1:,2] - sim[curveh[0]][:-1,2])/.1
    if process_regime:
        temp = regimes[0][0]
        relax = regimes[0][1]!=0
        regimes = [temp, relax]
    return obj, grad, regimes

gs = gridspec.GridSpec(3,2)
fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom = .18, top=.97)
ax = plt.subplot(gs[:2,:])
# ax2, ax3, ax4, ax5 = plt.subplot(gs[2,0]), plt.subplot(gs[2,1]), plt.subplot(gs[3,0]), plt.subplot(gs[3,1])
ax3, ax5 = plt.subplot(gs[2,0]), plt.subplot(gs[2,1])

# ax plots
initobj, initgrad, regimes = helper(pinit)
artist, = ax.plot(sim[curveh[0]][:end,3])
ax.plot(meas[curveh[0]][:end,3])
artist2 = ax.scatter(list(range(end)), (35,)*end, s=1, c=regimes[0][:end], cmap='viridis')
artist3 = ax.scatter(list(range(end)), (35.5,)*end, s=1, c=regimes[1][:end], cmap='viridis')
# for max speed
ax2p = np.linspace(25, 45, 10000)
# objlist, gradlist = [], []
# for p1 in ax2p:
#     p[2] = p1
#     obj, grad, unused = helper(p, False, False)
#     objlist.append(obj), gradlist.append(grad[2])
# p[2] = pinit[2]
# with open('gradlist1.pkl', 'wb') as f:
#     pickle.dump([objlist, gradlist], f)
with open('gradlist1.pkl', 'rb') as f:
    objlist, gradlist = pickle.load(f)
# ax2.plot(ax2p, objlist, 'k.', markersize=2)
ax3.plot(ax2p, gradlist, 'k.', markersize=2)
# ax2art, = ax2.plot(pinit[2], initobj, 'r.')
ax3art, = ax3.plot(pinit[2], initgrad[2], 'r.')
# for relax
ax4p = np.linspace(.1, 15, 10000)
# objlist, gradlist = [], []
# for p1 in ax4p:
#     p[3] = p1
#     obj, grad, unused = helper(p, False, False)
#     objlist.append(obj), gradlist.append(grad[3])
# # with open('gradlist2.pkl', 'wb') as f:
# #     pickle.dump([objlist, gradlist], f)
# p[3] = pinit[3]
with open('gradlist2.pkl', 'rb') as f:
    objlist, gradlist = pickle.load(f)
# ax4.plot(ax4p, objlist, 'k.', markersize=2)
ax5.plot(ax4p, gradlist, 'k.', markersize=2)
# ax4art, = ax4.plot(pinit[3], initobj, 'r.')
ax5art, = ax5.plot(pinit[3], initgrad[3], 'r.')
ax.set_ylabel('speed')
# ax2.set_ylabel('objective')
# ax2.set_xlabel('free flow speed')
ax3.set_ylabel('gradient')
# ax4.set_ylabel('objective')
# ax4.set_xlabel('relaxation time')
ax5.set_ylabel('gradient')

def update(val):
    p[2] = val
    obj, grad, regime = helper(p)
    artist.set_ydata(sim[curveh[0]][:end,3])
    artist2.set_array(np.array(regime[0][:end]))
    artist3.set_array(np.array(regime[1][:end]))

    # ax2art.set_xdata(val)
    ax3art.set_xdata(val)
    # ax2art.set_ydata(obj)
    ax3art.set_ydata(grad[2])
    # ax4art.set_ydata(obj)
    ax5art.set_ydata(grad[3])
    fig.canvas.draw_idle()

def update2(val):
    p[3] = val
    obj, grad, regime = helper(p)
    artist.set_ydata(sim[curveh[0]][:end,3])
    artist2.set_array(np.array(regime[0][:end]))
    artist3.set_array(np.array(regime[1][:end]))

    # ax4art.set_xdata(val)
    ax5art.set_xdata(val)
    # ax2art.set_ydata(obj)
    ax3art.set_ydata(grad[2])
    # ax4art.set_ydata(obj)
    ax5art.set_ydata(grad[3])
    fig.canvas.draw_idle()

axp2 = plt.axes([.15, 0.1, 0.65, 0.03])
axp = plt.axes([.15, 0.05, 0.65, 0.03])
p_values2 = Slider(axp, 'relaxation', .1, 15, valinit=pinit[3])
p_values2.on_changed(update2)
p_values = Slider(axp2, 'free flow', 25, 45, valinit=pinit[2])
p_values.on_changed(update)






