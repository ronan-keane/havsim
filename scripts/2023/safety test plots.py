import matplotlib.pyplot as plt
import numpy as np

def IDM(s, v, vl, p):
    """Intelligent Driver Model with parameters p = [c1, c2, c3, c4, c5].
    Using Treiber, Kesting notation, p = [v_0, T, s_0, a, b] =
    [free flow speed, desired time gap, jam distance, acceleration, comfortable deceleration]
    """
    s_star = p[2] + p[1]*v + (v*(v-vl))/(2*(p[3]*p[4])**.5)
    return p[3]*(1 - (v/p[0])**4 - (s_star/s)**2)

def make_follower_trajectory(lead_pos, lead_speed, dt, l, veh_pos, veh_speed, p):
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
    for i in range(len(lead_pos)-1):
        l_pos, l_speed = lead_pos[i], lead_speed[i]
        s = l_pos - veh_pos - l

        veh_acc = IDM(s, veh_speed, l_speed, p)
        veh_pos += dt*veh_speed
        veh_speed += dt*veh_acc
        xn.append(veh_pos)
        xn_dot.append(veh_speed)
    return xn, xn_dot


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


def plot_pdf(pdf_param, pdf_fn=None, end_value=10):
    """pdf_param = iterable of parameter values, pdf_fn=function return pdf values, end_value=last x value in plot."""
    x = np.linspace(.01, end_value, 200)
    y = pdf_fn(x, *pdf_param)
    plt.figure()
    plt.plot(x, y)
    plt.show()

p = [35, 1.3, 2, 1.1, 1.5]
lognormal_p = [np.log(1.85), .894]