"""Simple car following only calibration."""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sopt
import pickle
import havsim


def make_follower_trajectory(lead_pos, lead_speed, dt, length, veh_pos, veh_speed, p):
    """Generates following trajectory given the lead vehicle trajectory.

    Args:
        lead_pos: list of floats giving leader position at times 0, dt, 2dt, etc.
        lead_speed: list of floats giving leader speeds at times 0, dt, 2dt, etc.
        dt: timestep
        length: lead vehicle length
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
        s = l_pos - veh_pos - length
        veh_acc = havsim.simulation.models.IDM(p, [s, veh_speed, l_speed])
        veh_pos += dt*veh_speed
        veh_speed += dt*veh_acc
        xn.append(veh_pos)
        xn_dot.append(veh_speed)
    return xn, xn_dot


def get_data(meas, platooninfo, vehid):
    t_0, t_1, t_2 = platooninfo[vehid][:3]
    veh_pos_gt = meas[vehid][t_1 - t_0:t_2 + 1 - t_0, 2]
    veh_speed_gt = meas[vehid][t_1 - t_0:t_2 + 1 - t_0, 3]
    veh_pos = float(veh_pos_gt[0])
    veh_speed = float(meas[vehid][t_1 - t_0, 3])
    leadid = platooninfo[vehid][4][0]
    lead_t0 = int(meas[leadid][0][1])
    lead_pos = list(meas[leadid][t_1 - lead_t0:t_2 + 1 - lead_t0, 2])
    lead_speed = list(meas[leadid][t_1 - lead_t0:t_2 + 1 - lead_t0, 3])
    length = meas[leadid][0, 6]
    return lead_pos, lead_speed, length, veh_pos_gt, veh_speed_gt, veh_pos, veh_speed


def make_loss_fn(meas, platooninfo, vehid):
    lead_pos, lead_speed, length, veh_pos_gt, veh_speed_gt, veh_pos, veh_speed = get_data(meas, platooninfo, vehid)

    def loss(p):
        xn, xn_dot = make_follower_trajectory(lead_pos, lead_speed, .1, length, veh_pos, veh_speed, p)
        return np.sum(np.square(np.array(xn) - veh_pos_gt))
    return loss


if __name__ == '__main__':
    with open('C:\\Users\\tawit\\Documents\\Github\\havsim\\data\\recon-ngsim-old.pkl', 'rb') as f:
        meas, platooninfo = pickle.load(f)
    vehid = 1013
    p = [35*3.3, 1.3, 2*3.3, 1.1*3.3, 1.5*3.3]  # initial guess, note that units use feet, seconds
    bounds = [(20, 120), (.5, 2.8), (1, 8), (1, 25), (1, 25)]  # bounds that optimizer must stay within
    if len(platooninfo[vehid][4]) > 1:
        raise AssertionError('Vehicle '+str(vehid)+' has lane changes and will not be calibrated')

    loss = make_loss_fn(meas, platooninfo, vehid)
    res = sopt.minimize(loss, p, bounds=bounds, method='L-BFGS-B')
    calibrated_p = res['x']
    lead_pos, lead_speed, length, veh_pos_gt, veh_speed_gt, veh_pos, veh_speed = get_data(meas, platooninfo, vehid)
    xn, xn_dot = make_follower_trajectory(lead_pos, lead_speed, .1, length, veh_pos, veh_speed, p)
    plt.plot(xn_dot, 'k', alpha=.2)
    xn_cal, xn_dot_cal = make_follower_trajectory(lead_pos, lead_speed, .1, length, veh_pos, veh_speed, calibrated_p)
    plt.plot(veh_speed_gt, 'C0')
    plt.plot(xn_dot_cal, 'C1')
    plt.legend(['init guess', 'ground truth', 'calibrated'])
    plt.show()
