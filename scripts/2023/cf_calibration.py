"""Simple car following only calibration."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sopt
import pickle
import havsim


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
        veh_acc = havsim.simulation.models.IDM(p, [s, veh_speed, l_speed])
        veh_pos += dt*veh_speed
        veh_speed += dt*veh_acc
        xn.append(veh_pos)
        xn_dot.append(veh_speed)
    return xn, xn_dot


def make_loss_fn(meas, platooninfo, veh):
    pass