import matplotlib.pyplot as plt
import numpy as np

def IDM(s, v, vl, p):
    """Intelligent Driver Model with parameters p = [c1, c2, c3, c4, c5].
    Using Treiber, Kesting notation, p = [v_0, T, s_0, a, b] =
    [free flow speed, desired time gap, jam distance, acceleration, comfortable deceleration]
    """
    s_star = p[2] + p[1]*v + (v*(v-vl))/(2*(p[3]*p[4])**.5)
    return p[3]*(1 - (v/p[0])**4 - (s_star/s)**2)

def make_follower_trajectory(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p, gamma_p, gamma_fn, seed=None):
    """Generates following trajectory given the lead vehicle trajectory.

    Args:
        lead_pos: list of floats giving leader position at times 0, dt, 2dt, etc.
        lead_speed: list of floats giving leader speeds at times 0, dt, 2dt, etc.
        dt: timestep
        l: lead vehicle length
        veh_pos: initial position of following vehicle
        veh_speed: initial speed of following vehicle
        p: parameters for car following model (IDM)
    Returns:
        xn: list of positions of follower vehicle, with same shape as lead_pos
        xn_dot: list of speeds of follower vehicle
    """
    xn = [veh_pos]
    xn_dot = [veh_speed]
    xn_ddot = []

    t_ind = -1
    prev_acc = 0
    gamma = 0
    bar_gamma = (gamma/dt) // 1.
    beta = gamma/dt - bar_gamma
    next_t_ind = t_ind + bar_gamma + 1.

    for i in range(len(lead_pos)-1):
        if i == next_t_ind:
            l_pos, l_speed = lead_pos[i], lead_speed[i]
            s = l_pos - veh_pos - l
            new_acc = IDM(s, veh_speed, l_speed, p)
            veh_acc = prev_acc * beta + new_acc * (1 - beta)

            t_ind = next_t_ind
            prev_acc = new_acc
            if i == 0 and seed is not None:
                gamma = seed/gamma_scale(new_acc)
            else:
                gamma = gamma_fn(gamma_p)/gamma_scale(new_acc)
            bar_gamma = (gamma / dt) // 1.
            beta = gamma / dt - bar_gamma
            next_t_ind = t_ind + bar_gamma + 1.
        else:
            veh_acc = prev_acc
        veh_acc = max(min(veh_acc, 4), -6)
        veh_pos += dt*veh_speed
        veh_speed += dt*veh_acc
        veh_speed = max(veh_speed, 0)
        xn.append(veh_pos)
        xn_dot.append(veh_speed)
        xn_ddot.append(veh_acc)
    return xn, xn_dot, xn_ddot


def make_follower_trajectory_delay(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p, tau):
    """Generates following trajectory given the lead vehicle trajectory.

    Args:
        lead_pos: list of floats giving leader position at times 0, dt, 2dt, etc.
        lead_speed: list of floats giving leader speeds at times 0, dt, 2dt, etc.
        dt: timestep
        l: lead vehicle length
        veh_pos: initial position of following vehicle
        veh_speed: initial speed of following vehicle
        p: parameters for car following model (IDM)
        tau: the reaction time is tau * dt where tau is an integer
    Returns:
        xn: list of positions of follower vehicle, with same shape as lead_pos
        xn_dot: list of speeds of follower vehicle
    """

    xn = [veh_pos]
    xn_dot = [veh_speed]
    xn_ddot = []

    for i in range(len(lead_pos)-1):
        if i < tau:  # assume start in equilibrium
            s = lead_pos[0] - xn[0] - l
            veh_acc = IDM(s, xn_dot[0], lead_speed[0], p)
        else:
            s = lead_pos[i-tau] - xn[i-tau] - l
            veh_acc = IDM(s, xn_dot[i-tau], lead_speed[i - tau], p)
        veh_acc = max(min(veh_acc, 4), -6)
        veh_pos += dt*veh_speed
        veh_speed += dt*veh_acc
        veh_speed = max(veh_speed, 0)
        xn.append(veh_pos)
        xn_dot.append(veh_speed)
        xn_ddot.append(veh_acc)
    return xn, xn_dot, xn_ddot


def gamma_scale(acc):
    return max(1.33*abs(acc)**2-.33, 0)+1


def lognormal_pdf(x, mu, sigma):
    return 1 / ((2 * 3.1415926) ** .5 * sigma) / x * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))


def pareto_pdf(x, scale, shape):
    return (shape*scale**shape)/(x + scale)**(shape+1)


def frechet_pdf(x, scale, shape):
    x = x/scale
    return shape/scale*x**(-1-shape)*np.exp(-x**(-1*shape))


def weibull_pdf(x, scale, shape):
    x = x / scale
    return shape / scale * x ** (shape-1) * np.exp(-x ** shape)


def no_sample(*args):
    return 0.


RNG = np.random.default_rng()
def sample_lognormal(p):
    return np.exp(RNG.standard_normal()*p[1]+p[0])


def sample_pareto(p):
    return p[0]/(RNG.random()**(1/p[1])) - p[0]


def sample_weibull(p):
    return p[0]*(-np.log(RNG.random()))**(1/p[1])


def plot_pdf(pdf_param, pdf_fn=None, end_value=10):
    """pdf_param = iterable of parameter values, pdf_fn=function return pdf values, end_value=last x value in plot."""
    x = np.linspace(.01, end_value, 200)
    y = pdf_fn(x, *pdf_param)
    plt.figure()
    plt.plot(x, y)
    # plt.show()

p = [35, 1.3, 2, 1.1, 1.5]
lognormal_p = [np.log(.6), .7]
pareto_p = [1., 2.]

zs = np.array([-1.28, -.52, .52, 1.28])  # .1, .3, .7, .9 percentiles
print('percentiles are '+str(np.exp(zs*lognormal_p[1] + lognormal_p[0])))
print('mean is '+str(np.exp(lognormal_p[0]+lognormal_p[1]**2/2)))
plot_pdf(lognormal_p, lognormal_pdf, 20)
plot_pdf(pareto_p, pareto_pdf, 10)
seeds = [.2, .45, .8, 1.2, 2.]

lead_speed = [20.2 - i*.4 for i in range(14)]
lead_speed.extend([15]*51)
dt = .2
t = [dt*i for i in range(len(lead_speed))]
lead_pos = [0]
for i in lead_speed[:-1]:
    lead_pos.append(lead_pos[-1]+i*dt)
l = 5
eql_hd = ((p[2]+p[1]*lead_speed[0])**2/(1 - (lead_speed[0]/p[0])**4))**.5
veh_pos = lead_pos[0]-l-eql_hd
veh_speed = lead_speed[0]
xn, xn_dot, xn_ddot = make_follower_trajectory(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p, lognormal_p, sample_lognormal, seed=seeds[0])
xn1, xn_dot1, xn_ddot1 = make_follower_trajectory(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p, lognormal_p,
                                                   no_sample)
xn2, xn_dot2, xn_ddot2 = make_follower_trajectory_delay(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p, 3)

fig1 = plt.figure()
# plt.plot(t, lead_speed, 'k--')
plt.plot(t, xn_dot1, 'k', linestyle='dashed')
plt.plot(t, xn_dot2, 'C2', linestyle='dotted')
plt.plot(t, xn_dot, 'C1', alpha=.5)
plt.xlabel('time (s)')
plt.ylabel('speed (m/s)')
plt.legend(['normal car following', 'delayed car following', 'stochastic car following (ours)'])
fig2 = plt.figure()
plt.plot(t[:-1], xn_ddot1, 'k', linestyle='dashed')
plt.plot(t[:-1], xn_ddot2, 'C2', linestyle='dotted')
plt.plot(t[:-1], xn_ddot, 'C1', alpha=.5)
plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s/s)')
plt.legend(['normal car following', 'delayed car following', 'stochastic car following (ours)'])
fig3 = plt.figure()
plt.plot(np.array(lead_pos)-xn-l, alpha=.5)
for i in range(len(seeds)-1):
    xn, xn_dot, xn_ddot = make_follower_trajectory(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p, lognormal_p,
                                                   sample_lognormal, seed=seeds[i+1])
    fig1.axes[0].plot(t, xn_dot, 'C1', alpha=.5)
    fig2.axes[0].plot(t[:-1], xn_ddot, 'C1', alpha=.5)
    fig3.axes[0].plot(np.array(lead_pos)-xn-l, alpha=.5)
plt.show()