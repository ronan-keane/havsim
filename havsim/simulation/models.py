"""Houses all the different models for simulation."""

import math
import numpy as np
from havsim.simulation.road_networks import get_headway


def IDM(p, state):
    """Intelligent Driver Model (IDM), second order ODE.

    Note that if the headway is negative, model will begin accelerating; If velocity is negative,
    model will begin decelerating. Therefore you must take care to avoid any collisions or negative speeds
    in the simulation.

    Args:
        p: parameters - [max speed, comfortable time headway, jam spacing, comfortable acceleration,
                         comfortable deceleration]
        state: list of [headway, self velocity, leader velocity]

    Returns:
        acceleration
    """
    return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2]))
                                        / (2*(p[3]*p[4])**(1/2)))/(state[0]))**2)


def IDM_free(p, state):
    """Free flow model for IDM, state = self velocity, p = parameters. Returns acceleration."""
    return p[3]*(1-(state/p[0])**4)


def IDM_eql(p, v):
    """Equilibrium solution for IDM, v = velocity, p = parameters. Returns headway corresponding to eql."""
    s = ((p[2]+p[1]*v)**2/(1 - (v/p[0])**4))**.5
    return s


def OVM(p, state):
    """Optimal Velocity Model (OVM), second order ODE.

    Different forms of the optimal velocity function are possible - this implements the original
    formulation which uses tanh, with 4 parameters for the optimal velocity function.

    Args:
        p: parameters - p[0],p[1],p[2],p[4] are shape parameters for the optimal velocity function.
            The maximum speed is p[0]*(1 - tanh(-p[2])), jam spacing is p[4]/p[1]. p[1] controls the
            slope of the velocity/headway equilibrium curve. higher p[1] = higher maximum flow.
            There is no physical interpretation of p[1].
            p[3] is a sensitivity parameter which controls the strength of acceleration.
        state: list of [headway, self velocity, leader velocity] (leader velocity unused)

    Returns:
        acceleration
    """
    return p[3]*(p[0]*(math.tanh(p[1]*state[0]-p[2]-p[4])-math.tanh(-p[2]))-state[1])


def OVM_free(p, v):
    """Free flow model for OVM."""
    return p[3]*(p[0]*(1-math.tanh(-p[2])) - v)


def OVM_eql(p, s):
    """Equilibrium Solution for OVM, s = headway, p = parameters. Note that eql_type = 's'."""
    return p[0]*(math.tanh(p[1]*s-p[2]-p[4])-math.tanh(-p[2]))


def lc_havsim(veh, lc_actions, lc_followers, timeind):
    """Havsim lane changing model.

    Complete lane changing model, based on the mandatory/discretionary criteria given in (Traffic Flow Dynamics, 2013),
    combined with discretionary, tactical, cooperation, relaxation, and route models given by havsim.

    The different groups parameters are as follows (parameters are lists where each index has a given parameter).
    lc_parameters:
        0 - safety criteria (maximum deceleration allowed after LC, more negative = less strict),
        1 - safety criteria during mandatory lane changing
        2 - incentive criteria (>0, larger = more strict. smaller = discretionary changes more likely),
        3 - politeness (taking other vehicles into account, 0 = ignore other vehicles, ~.1-.2 = realistic),
        4 - bias on left side (can add a bias to make vehicles default to a certain side of the road),
        5 - bias on right side
        6 - probability of checking LC while in discretionary state per timestep
        7 - after meeting discretionary change incentive, number of timesteps that the probability of checking LC
            will be 100%
        8 - number of timesteps after a discretionary change when another discretionary change is not
            possible (not in original model)

    lc2 parameters:
        0 - deceleration applied during tactical/cooperative manuevering (more negative = stronger adjustments)
        1 - acceleration applied during tactical/cooperative manuevering (more positive = stronger adjustments)
        2 - acceleration threshold for strong cooperation (more negative = more cooperation)
        3 - minimum time headway needed for cooperation (gap must be at least p * spd where p is the parameter
            and spd is the vehicle speed).
        4 - cooperation default probability

    relax_parameters:
        0 - relaxation time length, time needed to adjust after a lane change
        1 - minimum time headway used for relaxation safeguard
        2 - minimum time to collision used for relaxation safeguard

    route_parameters: (see also simulation.update_lane_routes.make_cur_route)
        0 - reach 100% forced cooperation of follower when this distance from end of merge
        1 - 0 + 1 gives the minimum comfortable distance for completing a mandatory lane change

    Args:
        veh: ego vehicle
        lc_actions: dictionary where lane changing actions are stored, keys = vehicles, values = 'l' or 'r'
        lc_followers: For any Vehicle which request to change lanes, the new follower must be a key in lc_followers,
            value is a list of all vehicles which requested change. Used to prevent multiple vehicles from changing
            in front of same follower in the same timestep.
        timeind: time index
    Returns:
        lc_actions, lc_followers
    """
    if veh.chk_disc:
        if np.random.rand() > veh.lc_parameters[6]:
            return lc_actions, lc_followers

    p, l_lc, r_lc = veh.lc_parameters, veh.l_lc, veh.r_lc
    if l_lc == 'd':
        if r_lc is None:
            new_fol_hd, new_fol_a = veh.fol.get_cf(veh.lead, timeind)
            new_lcfol = veh.lfol
            new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_hd_acc_under_lc(new_lcfol, veh, timeind)
            incentive = new_veh_a - veh.acc + p[3]*(new_lcfol_a - new_lcfol.acc + new_fol_a - veh.fol.acc) + p[4]

        elif r_lc == 'd':
            new_fol_hd, new_fol_a = veh.fol.get_cf(veh.lead, timeind)
            new_lcfol = veh.lfol
            new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_hd_acc_under_lc(new_lcfol, veh, timeind)
            incentive = new_veh_a - veh.acc + p[3]*(new_lcfol_a - new_lcfol.acc + new_fol_a - veh.fol.acc) + p[4]

            new_lcfol2 = veh.rfol
            new_lcfol_hd2, new_veh_hd2, new_lcfol_a2, new_veh_a2 = new_hd_acc_under_lc(new_lcfol2, veh, timeind)
            incentive2 = new_veh_a2 - veh.acc + p[3]*(new_lcfol_a2 - new_lcfol2.acc + new_fol_a - veh.fol.acc) + p[5]
            if incentive2 > incentive:
                incentive = incentive2
                new_lcfol = new_lcfol2
                new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_lcfol_hd2, new_veh_hd2, new_lcfol_a2, new_veh_a2

        else:
            new_lcfol = veh.rfol
            new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_hd_acc_under_lc(new_lcfol, veh, timeind)

    elif l_lc is None:
        if r_lc == 'd':
            new_fol_hd, new_fol_a = veh.fol.get_cf(veh.lead, timeind)
            new_lcfol = veh.rfol
            new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_hd_acc_under_lc(new_lcfol, veh, timeind)
            incentive = new_veh_a - veh.acc + p[3]*(new_lcfol_a - new_lcfol.acc + new_fol_a - veh.fol.acc) + p[5]

        elif r_lc is None:
            return lc_actions, lc_followers

        else:
            new_lcfol = veh.rfol
            new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_hd_acc_under_lc(new_lcfol, veh, timeind)
    else:
        new_lcfol = veh.lfol
        new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a = new_hd_acc_under_lc(new_lcfol, veh, timeind)

    if veh.in_disc:  # discretionary update
        if veh.is_coop:  # if cooperating, make sure the incentive includes lc_acceleration
            if veh.lc_acc == 0:
                check_coop_and_apply(veh, veh.is_coop, veh.lc2_parameters, timeind)
            incentive -= veh.lc_acc

        if incentive > p[2]:
            if timeind < veh.disc_cooldown:
                return lc_actions, lc_followers
            fol_safe, veh_safe = new_lcfol_a > p[0], new_veh_a > p[0]
            if fol_safe and veh_safe:
                return complete_change(lc_actions, lc_followers, veh, new_lcfol)
            if veh.chk_disc:
                veh.chk_disc = False
                veh.disc_endtime = timeind + p[7]

            if veh.lc_acc == 0:
                veh.lc_acc = apply_tactical_discretionary(fol_safe, veh_safe, new_lcfol, veh)
            if not veh.has_coop:
                try_find_new_coop(new_lcfol, veh, new_lcfol_hd, timeind, 0)
            else:
                check_coop_and_apply(veh.has_coop, veh, veh.has_coop.lc2_parameters, timeind)

        else:
            if veh.has_coop:
                coop_veh = veh.has_coop
                coop_veh.chk_disc = timeind > coop_veh.disc_endtime if coop_veh.in_disc else False
                coop_veh.is_coop = veh.has_coop = None

        if not veh.chk_disc:
            if timeind > veh.disc_endtime:
                veh.chk_disc = True

    else:  # mandatory update
        fol_safe, veh_safe = new_lcfol_a, new_veh_a
        if fol_safe and veh_safe:
            return complete_change(lc_actions, lc_followers, veh, new_lcfol)

        if veh.lc_acc == 0:
            veh.lc_acc = apply_tactical(fol_safe, veh_safe, new_lcfol, veh)
        if not veh.has_coop:
            s1, s2 = veh.lc_urgency
            coop_correction = (veh.pos - s1)/(s2 - s1 + 1e-6)
            try_find_new_coop(new_lcfol, veh, new_lcfol_hd, timeind, coop_correction)
        else:
            check_coop_and_apply(veh.has_coop, veh, veh.has_coop.lc2_parameters, timeind)

    return lc_actions, lc_followers


def new_hd_acc_under_lc(lcfol, veh, timeind):
    new_lcfol_hd, new_lcfol_a = lcfol.get_cf(veh, timeind)
    new_veh_hd, new_veh_a = veh.get_cf(lcfol.lead, timeind)
    return new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a


def complete_change(lc_actions, lc_followers, veh, new_lcfol):
    side = 'l' if new_lcfol is veh.lfol else 'r'
    lc_actions[veh] = side
    if new_lcfol in lc_followers:
        lc_followers[new_lcfol].append(veh)
    else:
        lc_followers[new_lcfol] = [veh]
    return lc_actions, lc_followers


def apply_coop(veh, ego_veh, hd, p2):
    lc_acc = veh.cf_model(veh.cf_parameters, [hd, veh.speed, ego_veh.speed])
    if veh.acc > lc_acc > p2[2]:
        veh.lc_acc = -veh.acc + lc_acc
    else:
        veh.lc_acc = p2[0]


def check_coop_and_apply(veh, ego_veh, p2, timeind):
    """For active cooperation, check it is still valid and set the lc_acc from cooperation if so."""
    # veh = cooperating, ego_veh = trying to change, p2 = parameters for veh
    hd = get_headway(veh, ego_veh)
    min_hd = 2 + p2[3] * veh.speed
    if hd > min_hd:
        if veh == getattr(ego_veh, ego_veh.coop_side_fol):
            apply_coop(veh, ego_veh, hd, p2)
            return
        elif veh.lead is not None:
            lead = veh.lead
            lead_hd = get_headway(lead, ego_veh)
            if lead.speed > ego_veh.speed and lead_hd < min_hd:
                apply_coop(veh, ego_veh, hd, p2)
                return
    veh.chk_disc = timeind > veh.disc_endtime if veh.in_disc else False
    veh.is_coop = ego_veh.has_coop = None


def apply_tactical(fol_safe, veh_safe, new_lcfol, veh):
    if not fol_safe and veh_safe:
        if new_lcfol.speed + 6 > veh.speed > new_lcfol.speed - 2:
            return veh.lc2_parameters[1]
    elif fol_safe and not veh_safe:
        if veh.speed > new_lcfol.speed - 3:
            return veh.lc2_parameters[0]
    return 0


def apply_tactical_discretionary(fol_safe, veh_safe, new_lcfol, veh):
    if not fol_safe and veh_safe:
        if new_lcfol.speed + 6 > veh.speed > new_lcfol.speed - 2:
            return veh.lc2_parameters[1]
    return 0


def try_find_new_coop(new_lcfol, veh, new_lcfol_hd, timeind, coop_correction):
    """For a requested lane change, try to start a new cooperation."""
    # new_lcfol = lane change side follower of veh. veh = vehicle trying to change
    # coop_correction is the extra cooperation probability that comes from mandatory lane changes
    test_veh = new_lcfol
    hd = new_lcfol_hd
    min_hd = 2 + test_veh.lc2_parameters[3] * test_veh.speed
    if hd > min_hd:  # headway condition met
        maybe_add_new_coop(test_veh, veh, new_lcfol, hd, timeind, coop_correction)
    else:
        # find first vehicle which meets headway condition
        lead, prev_hd = test_veh, hd
        test_veh = test_veh.fol
        hd = get_headway(test_veh, veh)
        min_hd = 2 + test_veh.lc2_parameters[3] * test_veh.speed
        while hd < min_hd:  # note that anchor vehicles have min_hd = -inf, so while loop must terminate
            lead, prev_hd = test_veh, hd
            test_veh = test_veh.fol
            hd = get_headway(test_veh, veh)
            min_hd = 2 + test_veh.lc2_parameters[3] * test_veh.speed
        if lead.speed > veh.speed and prev_hd < min_hd:  # extra condition for cooperation if not new_lcfol
            maybe_add_new_coop(test_veh, veh, new_lcfol, hd, timeind, coop_correction)


def maybe_add_new_coop(test_veh, veh, new_lcfol, hd, timeind, coop_correction):
    """For a vehicle which meets the headway condition for cooperation, start new cooperation if possible."""
    if not test_veh.is_coop:  # not currently cooperating
        coop_p = test_veh.lc2_parameters[4] + coop_correction
        if np.random.rand() < coop_p:  # cooperation condition met
            test_veh.is_coop = veh
            veh.has_coop = test_veh
            veh.coop_side_fol = 'lfol' if new_lcfol == veh.lfol else 'rfol'
            test_veh.chk_disc = False
            test_veh.disc_endtime = timeind + test_veh.lc_parameters[7]
            apply_coop(test_veh, veh, hd, test_veh.lc2_parameters)


def default_parameters():
    """Suggested parameters for the IDM and havsim lane changing model."""
    cf_parameters = [35, 1.3, 2, 1.1, 1.5]
    lc_parameters = [-4, -8, .3, .15, 0, 0, .2, 10, 42]
    lc2_parameters = [-3, 2, -4, .5, .2]
    relax_parameters = [8.7, .1, 1.5]
    route_parameters = [300, 500]
    return {'cf_parameters': cf_parameters, 'lc_parameters': lc_parameters, 'lc2_parameters': lc2_parameters,
            'relax_parameters': relax_parameters, 'route_parameters': route_parameters}
