"""
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

lead = list(range(8))+list(np.arange(7.1,7.6,.1))+list(np.arange(7.6,15.6,1))
tar = np.array(list(range(-2,7)) + list(np.arange(6.1,6.6,.1)) + list(np.arange(6.6,13.6,1)))
timesteps = len(lead)

def forward(p, return_traj=False):
    out = [-2]
    out2 = []
    p1, p2 = p[0], p[1]
    cur = out[-1]
    for i in range(len(lead)-1):
        if (lead[i] - cur) <= p1:
            cur = cur + p2
            out2.append(1)
        else:
            cur = cur + 1
            out2.append(0)
        out.append(cur)
    if return_traj:
        return np.sum(np.square(np.array(out) - tar)), out, out2
    else:
        return np.sum(np.square(np.array(out) - tar))

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(gs[0,:])
ax2, ax3 = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
plt.subplots_adjust(bottom=.2)

pinit = [2.1, .1]
p = pinit.copy()
x1, x2 = np.linspace(0, 3, 5000), np.linspace(-1,1, 5000)
objlist1, objlist2 = [], []
for p1 in x1:
    p[0] = p1
    objlist1.append(forward(p))
p = pinit.copy()
for p2 in x2:
    p[1]= p2
    objlist2.append(forward(p))
#main plot
obj, traj, branches = forward(pinit, True)
ax.plot(lead, 'k.-')
ax.plot(tar, 'C0', alpha=.5)
artist_switch, = ax.plot(np.array(lead)-pinit[0], 'C2--', alpha=.2, linewidth=1)
artist_traj, = ax.plot(traj, 'C1.-')
ax.legend(['leader', 'target trajectory', 'switching condition', 'trajectory'])
artist_branch = ax.scatter(list(range(0,len(lead)-1)), [-3]*(len(lead)-1), s=4, c=branches, cmap='viridis')
#side plots
ax2.plot(x1, objlist1, 'k.', markersize=2)
ax3.plot(x2, objlist2, 'k.', markersize=2)
ax2.set_ylabel('objective')
ax2.set_xlabel('value of switching parameter')
ax3.set_ylabel('objective')
ax3.set_xlabel('value of branch parameter')
artist2, = ax2.plot(pinit[0], obj, 'r.')
artist3, = ax3.plot(pinit[1], obj, 'r.')

def update(val): #slider for parameter 0
    pinit[0] = val
    obj, traj, branches = forward(pinit, True)
    artist_branch.set_array(np.array(branches))
    artist_switch.set_ydata(np.array(lead)-val)
    artist_traj.set_ydata(traj)
    artist2.set_xdata(val)
    artist2.set_ydata(obj)
    artist3.set_ydata(obj)
    fig.canvas.draw_idle()

def update2(val): # slider for parameter 1
    pinit[1] = val
    obj, traj, branches = forward(pinit, True)
    artist_branch.set_array(np.array(branches))
    artist_traj.set_ydata(traj)
    artist2.set_ydata(obj)
    artist3.set_xdata(val)
    artist3.set_ydata(obj)
    fig.canvas.draw_idle()

axp = plt.axes([.15, .1, .65, .03]) # for slider
axp2 = plt.axes([.15, .05, .65, .03])
p_values = Slider(axp, 'switching', x1[0], x1[-1], valinit=pinit[0])
p_values.on_changed(update)
p_values2 = Slider(axp2, 'branch', x2[0], x2[-1], valinit=pinit[1])
p_values2.on_changed(update2)


#%%  solve problem by using stochastic model + SGD
pinit = [2.1, 0, np.log(.1)]  #p[1] and p[2] are taken to be positive
norm_dist = tfp.distributions.Normal(tf.convert_to_tensor(0, dtype=tf.float32), tf.convert_to_tensor(1, dtype=tf.float32))

def obj_and_grad(p):  # note - no variance reduction used
    out = [-2]
    mu, eps, p2 = p[0], p[1], p[2]
    cur = out[-1]
    zlist = []
    out2 = []
    for i in range(len(lead)-1):
        z = tf.random.normal((1,))
        zlist.append(z)
        if  np.exp(eps)*float(z)<= mu - (lead[i] - cur):
            cur = cur + np.exp(p2)
            out2.append(1)
        else:
            cur = cur + 1
            out2.append(0)
        out.append(cur)
    obj = np.sum(np.square(np.array(out) - tar))

    lam = 0
    ghat = np.array([0,0,0])
    mu, eps = tf.convert_to_tensor(mu, dtype=tf.float32), tf.convert_to_tensor(eps, dtype=tf.float32)
    for i in reversed(range(0,len(lead)-1)):
        dfdx = 2*(out[i+1] - tar[i+1])
        dhdx = 1
        if out2[i]==1:
            dhdp = np.array([0, 0, np.exp(p2)])
        else:
            dhdp = np.array([0, 0, 0])

        z = zlist[i]
        cur = tf.convert_to_tensor(out[i], dtype=tf.float32)
        cur_lead = tf.convert_to_tensor(lead[i], dtype=tf.float32)
        grad_log_probs = diff_prob(mu, eps, z, cur_lead, cur, tf.cast(out2[i], tf.bool))
        lam = lam + dfdx

        ghat = ghat + lam*dhdp + obj*np.array([grad_log_probs[0].numpy(), grad_log_probs[1].numpy(), 0])
        lam = lam*dhdx + obj*grad_log_probs[2].numpy()
    return obj, ghat, out


@tf.function
def diff_prob(mu, eps, z, lead, cur, branch):
    with tf.GradientTape() as g:
        g.watch([cur, mu, eps])
        x = (mu-(lead-cur))/tf.math.exp(eps)
        y = norm_dist.log_cdf(x)
        y = tf.cond(branch, lambda: y, lambda: tf.math.log(1-tf.math.exp(y)))
    return g.gradient(y, [mu, eps, cur])


def train(pinit, nsteps=1000, lr=1e-3, clipnorm=10):
    p = pinit.copy()
    objlist = []
    outlist = []
    plist = []
    for i in range(nsteps):
        obj, grad, out = obj_and_grad(p)
        # if np.linalg.norm(grad) > clipnorm:
        #     grad = grad/np.linalg.norm(grad)*clipnorm
        p = p-lr*grad
        objlist.append(obj)
        outlist.append(out)
        plist.append(p)
    return objlist, outlist, plist

trainobj, traintraj, plist = train(pinit)


#%% plot results of above

fig = plt.figure(figsize=(10,10))
ax = plt.subplot()
plt.subplots_adjust(bottom=.2)

ax.plot(lead, 'k.-')
ax.plot(tar, 'C0', alpha=.5)
leadarray = np.array(lead)
leadtime = list(range(len(lead)))
artist_mean, = ax.plot(leadtime, leadarray-pinit[0], 'C2--', alpha=.2, linewidth=1)
artist_stdev = ax.fill_between(leadtime, leadarray-pinit[0]-np.exp(pinit[1]), leadarray-pinit[0]+np.exp(pinit[1]),
                                alpha=.2, facecolor='C2')
artist_traj, = ax.plot(traintraj[0], 'C1.-')

def update(val): #slider plots iteration number
    val = int(val)
    curtraj = traintraj[val]
    curp = plist[val]
    artist_mean.set_ydata(leadarray-curp[0])
    ax.collections.clear()
    artist_stdev = ax.fill_between(leadtime, leadarray-curp[0]-np.exp(curp[1]), leadarray-curp[0]+np.exp(curp[1]),
                                alpha=.2, facecolor='C2')
    artist_traj.set_ydata(curtraj)
    fig.canvas.draw_idle()

axp = plt.axes([.15, .1, .65, .03])
p_values = Slider(axp, 'iteration', 0, 399, valinit=0)
p_values.on_changed(update)
