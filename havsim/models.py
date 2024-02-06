"""Houses all the different models for simulation."""

import havsim
import numpy as np


def IDM(p, state):
    """Intelligent Driver Model (IDM), second order ODE.

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
    return p[3]*(p[0]*(np.tanh(p[1]*state[0]-p[2]-p[4])-np.tanh(-p[2]))-state[1])


def OVM_free(p, v):
    """Free flow model for OVM."""
    return p[3]*(p[0]*(1-np.tanh(-p[2])) - v)


def OVM_eql(p, s):
    """Equilibrium Solution for OVM, s = headway, p = parameters. Note that eql_type = 's'."""
    return p[0]*(np.tanh(p[1]*s-p[2]-p[4])-np.tanh(-p[2]))


def lc_havsim(veh, lc_actions, lc_followers, timeind):
    """Havsim lane changing model.

    Complete lane changing model, based on the mandatory/discretionary criteria given in (Traffic Flow Dynamics, 2013),
    combined with discretionary, tactical, cooperation, relaxation, and route models given by havsim.

    The different groups parameters are as follows (parameters are lists where each index has a given parameter).
    lc_parameters:
        0 - safety criteria (maximum deceleration allowed after LC, more negative = less strict. Must be negative)
        1 - safety criteria during mandatory lane changing
        2 - incentive criteria (>0, larger = more strict. smaller = discretionary changes more likely),
        3 - politeness (taking other vehicles into account, 0 = ignore other vehicles, ~.1-.2 = realistic),
        4 - bias on left side (can add a bias to make vehicles default to a certain side of the road),
        5 - bias on right side
        6 - probability of checking LC while in discretionary state per timestep
        7 - after meeting discretionary change incentive, number of timesteps that the probability of checking LC
            will be 100%
        8 - number of timesteps after a discretionary change when another discretionary change is not
            possible

    lc2 parameters:
        0 - speed adjustment time (need 2.3*p[0] seconds to adjust to 90% of the speed difference)
        1 - comfortable acceleration for lane changes
        2 - comfortable speed gap. Decelerate if more than p[2] speed gap with new leader for lane changing
        3 - comfortable deceleration for cooperation during lane change
        4 - comfortable speed gap for cooperation during lane change. The cooperation vehicle will slow down until
            within p[4] speed of the changing vehicle.
        5 - default probability to cooperate. During mandatory changes, the probability is increased.

        If parameter 1 is zero, parameter 2 is large, then tactical model will have no effect. If parameters 3 and 4
        are large, then cooperation will have no effect. If parameter 5 is very negative, cooperation won't be applied.

    relax_parameters:
        0 - relaxation time length, time needed to adjust after a lane change. Used for positive relaxation (vehicle
            can accept shorter gaps after changing).
        1 - relaxation time length for negative relaxation (vehicle is sluggish to adjust to new faster lane)
        2 - minimum time headway used for relaxation safeguard
        3 - minimum time to collision used for relaxation safeguard

        If both parameters 0 and 1 are zero (or less than dt) then there is no relaxation.

    route_parameters: (see also simulation.update_lane_routes.make_cur_route)
        0 - reach 100% forced cooperation of follower when this distance from end of merge. If set to very negative,
            then we will always stay at whatever the default probability to cooperate is.
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
        if veh.npr.random() > veh.lc_parameters[6]:
            if veh.has_coop:  # remove coop
                coop_veh = veh.has_coop
                coop_veh.chk_disc = timeind > coop_veh.disc_endtime if coop_veh.in_disc else False
                coop_veh.is_coop = veh.has_coop = None
            return lc_actions, lc_followers

    # get relevant accelerations/headways
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
                ego_veh = veh.is_coop
                ego_safety = ego_veh.lc_parameters[0] if ego_veh.in_disc else ego_veh.lc_parameters[1]
                check_coop_and_apply(veh, ego_veh, veh.lc2_parameters, ego_safety, timeind)
            incentive -= veh.lc_acc

        if incentive > p[2]:
            if timeind < veh.disc_cooldown:
                return lc_actions, lc_followers
            safety = p[0]
            fol_safe, veh_safe = new_lcfol_a > safety, new_veh_a > safety
            if fol_safe and veh_safe:
                return complete_change(lc_actions, lc_followers, veh, new_lcfol)
            if veh.chk_disc:
                veh.chk_disc = False
            veh.disc_endtime = timeind + p[7]

            # apply tactical and cooperative
            if veh.lc_acc == 0:
                apply_tactical_discretionary(veh, veh_safe, fol_safe, veh.lc2_parameters)
            has_coop = veh.has_coop
            if not has_coop:
                try_find_new_coop(new_lcfol, veh, new_lcfol_a, safety, timeind, 0)
            elif has_coop.lc_acc == 0:
                check_coop_and_apply(has_coop, veh, has_coop.lc2_parameters, safety, timeind)

        else:
            if veh.has_coop:
                coop_veh = veh.has_coop
                coop_veh.chk_disc = timeind > coop_veh.disc_endtime if coop_veh.in_disc else False
                coop_veh.is_coop = veh.has_coop = None

        if not veh.chk_disc:
            if timeind > veh.disc_endtime:
                veh.chk_disc = True

    else:  # mandatory update
        safety = p[1]
        fol_safe, veh_safe = new_lcfol_a > safety, new_veh_a > safety
        if fol_safe and veh_safe:
            return complete_change(lc_actions, lc_followers, veh, new_lcfol)

        # apply tactical and cooperative
        if veh.lc_acc == 0:
            apply_tactical(veh, new_lcfol, veh_safe, fol_safe, veh.lc2_parameters)
        has_coop = veh.has_coop
        if not has_coop:
            s1, s2 = veh.lc_urgency
            coop_correction = (veh.pos - s1)/(s2 - s1 + 1)
            try_find_new_coop(new_lcfol, veh, new_lcfol_a, safety, timeind, coop_correction)
        elif has_coop.lc_acc == 0:
            check_coop_and_apply(has_coop, veh, has_coop.lc2_parameters, safety, timeind)

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


def apply_coop(veh, ego_veh, lc_acc, p2):
    # veh = cooperating, ego_veh = trying to change, p2 = parameters for veh
    min_speed = ego_veh.speed + p2[4]
    if veh.speed > min_speed:  # speed too fast -> always decelerate
        decel = max((min_speed - veh.speed)/p2[0], p2[3])
        if veh.acc > 0:
            veh.lc_acc = -veh.acc + decel
        else:
            veh.lc_acc = decel
    elif lc_acc > p2[3]:  # safe to follow ego -> use min of leader, ego
        if veh.acc > lc_acc:
            veh.lc_acc = -veh.acc + lc_acc


def check_coop_and_apply(veh, ego_veh, veh_p, ego_safety, timeind):
    """For active cooperation, check it is still valid and set the lc_acc from cooperation if so."""
    # veh = cooperating, ego_veh = trying to change, veh_p = parameters for veh
    unused, new_a = veh.get_cf(ego_veh, timeind)
    if new_a > ego_safety:
        if veh == getattr(ego_veh, ego_veh.coop_side_fol):
            apply_coop(veh, ego_veh, new_a, veh_p)
            return
        elif veh.lead is not None:
            lead = veh.lead
            unused, lead_a = lead.get_cf(ego_veh, timeind)
            if lead_a < ego_safety and lead.speed > ego_veh.speed:
                apply_coop(veh, ego_veh, new_a, veh_p)
                return
    veh.chk_disc = timeind > veh.disc_endtime if veh.in_disc else False
    veh.is_coop = ego_veh.has_coop = None


def apply_tactical(veh, new_lcfol, veh_safe, fol_safe, p2):
    if not veh_safe:
        lead = new_lcfol.lead
        if lead is not None:
            min_speed = lead.speed + p2[2]
            if veh.speed > min_speed:  # forced deceleration to match leader speed if necessary
                if veh.acc > 0:
                    veh.lc_acc = -veh.acc + (min_speed - veh.speed)/p2[0]
                else:
                    veh.lc_acc = (min_speed - veh.speed)/p2[0]
    elif not fol_safe:  # accelerate if possible
        veh.lc_acc = p2[1]


def apply_coop_old(veh, ego_veh, lc_acc, p2):
    """Alternative tactical/cooperation model that always uses constant deceleration.

    parameters:
        0 - comfortable deceleration for lane changes
        1 - comfortable acceleration for lane changes
        2 - comfortable speed gap for mandatory lane change. Only decelerate if more than p[2] speed than new leader.
            If parameters 1 is zero and parameter 2 is large, the tactical model has no effect.
        3 - comfortable deceleration for cooperation during lane change
        4 - comfortable speed gap for cooperation during lane change. The cooperation vehicle will slow down until
            within p[3] speed of the changing vehicle. Note that if parameters 3 and 4 are both set to very large
            values, then cooperating will have no effect.
        5 - default probability to cooperate. During mandatory changes, the probability is increased. If set to very
            negative value, then cooperation will never be applied.
    """
    if veh.speed > ego_veh.speed + p2[4]:  # speed too fast -> always decelerate
        if veh.acc > 0:
            veh.lc_acc = -veh.acc + p2[3]
        else:
            veh.lc_acc = p2[3]
    elif lc_acc > p2[3]:  # safe to follow ego -> use min of leader, ego
        if veh.acc > lc_acc:
            veh.lc_acc = -veh.acc + lc_acc


def apply_tactical_old(veh, new_lcfol, veh_safe, fol_safe, p2):
    if not veh_safe:
        lead = new_lcfol.lead
        if lead is not None:
            if veh.speed > lead.speed + p2[2]:  # forced deceleration to match leader speed if necessary
                if veh.acc > 0:
                    veh.lc_acc = -veh.acc + p2[0]
                else:
                    veh.lc_acc = p2[0]
    elif not fol_safe:  # accelerate if possible
        veh.lc_acc = p2[1]


def apply_tactical_discretionary(veh, veh_safe, fol_safe, p2):
    if not fol_safe and veh_safe:
        veh.lc_acc = p2[1]


def try_find_new_coop(new_lcfol, veh, new_lcfol_a, veh_safety, timeind, coop_correction):
    """For a requested lane change, try to start a new cooperation."""
    # new_lcfol = lane change side follower of veh. veh = vehicle trying to change
    # coop_correction is the extra cooperation probability that comes from mandatory lane changes
    if new_lcfol_a > veh_safety:
        maybe_add_new_coop(new_lcfol, veh, new_lcfol, new_lcfol_a, timeind, coop_correction)
        return
    lead, lead_a = new_lcfol, new_lcfol_a
    for i in range(5):
        test_veh = lead.fol
        unused, test_veh_a = test_veh.get_cf(veh, timeind)
        if test_veh_a > veh_safety:
            if lead.speed > veh.speed:
                maybe_add_new_coop(test_veh, veh, new_lcfol, test_veh_a, timeind, coop_correction)
            break
        lead, lead_a = test_veh, test_veh_a


def maybe_add_new_coop(test_veh, veh, new_lcfol, test_veh_a, timeind, coop_correction):
    """For a vehicle which meets the safety condition for cooperation, start new cooperation if possible."""
    if not test_veh.is_coop:  # not currently cooperating
        coop_p = test_veh.lc2_parameters[5] + coop_correction
        if veh.npr.random() < coop_p:  # cooperation condition met
            test_veh.is_coop = veh
            veh.has_coop = test_veh
            veh.coop_side_fol = 'lfol' if new_lcfol is veh.lfol else 'rfol'
            test_veh.chk_disc = False
            test_veh.disc_endtime = timeind + test_veh.lc_parameters[7]
            apply_coop(test_veh, veh, test_veh_a, test_veh.lc2_parameters)


def new_relaxation(veh, timeind, dt):
    """Generates relaxation for a vehicle after it experiences a lane change.

    This is called directly after a vehicle changes it lane, while it still has the old value for its
    headway, and its position has not yet been updated.
    See (https://arxiv.org/abs/1904.08395) for an explanation of the relaxation model.

    Args:
        veh: Vehicle to add relaxation to
        timeind: int giving the timestep of the simulation (0 indexed)
        dt: length of timestep
    Returns:
        None. Modifies veh in place.
    """
    if veh.lead is None:  # new lead is None -> reset relaxation
        if veh.in_relax:
            veh.in_relax = False
            veh.relax = veh.relax[:timeind-veh.relax_start]
            veh.relaxmem.append((veh.relax, veh.relax_start))
        return
    prevlead = get_prev_lead(veh, timeind)
    if prevlead is None:  # old lead is None -> reset relaxation
        if veh.in_relax:
            veh.in_relax = False
            veh.relax = veh.relax[:timeind-veh.relax_start]
            veh.relaxmem.append((veh.relax, veh.relax_start))
        return
        # olds = veh.get_eql(veh.speed)  # alternative definition of relaxation when old lead is None
        # oldv = veh.speed
    olds = veh.hd
    oldv = prevlead.speed
    news = havsim.get_headway(veh, veh.lead)
    newv = veh.lead.speed

    rp = veh.relax_parameters
    relaxamount_s = olds-news
    relaxamount_v = oldv-newv
    relax_helper_vhd(rp[0], rp[1], 60, relaxamount_s, relaxamount_v, veh, timeind, dt)


def get_prev_lead(veh, timeind):
    mem = veh.leadmem[-2]
    if mem[1] > timeind:
        for i in range(len(veh.leadmem)-2):
            mem = veh.leadmem[-3-i]
            if mem[1] < timeind + 1:
                break
        else:
            return None
        return mem[0]
    else:
        return mem[0]


def relax_helper_vhd(pos_r, neg_r, max_s, relaxamount_s, relaxamount_v, veh, timeind, dt):
    """Helper function for headway + speed relaxation.

    Args:
        pos_r: positive relaxation length in seconds (float)
        neg_r: negative relaxation length in seconds (float)
        max_s: safeguard amount for headway relaxation
        relaxamount_s: relaxation amount for headway (float)
        relaxamount_v: relaxation amount for speed (float)
        veh: Vehicle to apply relaxation to
        timeind: time index
        dt: length of timestep
    """
    # rp = parameter, relaxamount_s = headway relaxation, _v = velocity relaxation
    rp = neg_r if relaxamount_v < 0 else pos_r
    relaxlen = int(np.ceil(rp/dt)) - 1
    if relaxlen <= 0:
        return

    relaxamount_s = max(min(relaxamount_s, max_s), -max_s)
    if veh.in_relax:  # case where relaxation is added to existing relax
        if timeind > veh.relax_start + len(veh.relax) - 1:  # edge case where veh is not actually in_relax
            veh.relaxmem.append((veh.relax, veh.relax_start))
            apply_normal_relaxation(veh, rp, relaxamount_s, relaxamount_v, timeind, dt, relaxlen)
        else:
            # calculate updated relax amount taking into account the current relax
            old_relax_s = veh.relax[timeind - veh.relax_start][0]
            relaxamount_s = max(min(relaxamount_s + old_relax_s, max_s), -max_s)
            relaxamount_s = relaxamount_s - old_relax_s
            tempdt = -dt / rp * relaxamount_s
            tempdt2 = -dt / rp * relaxamount_v
            curr = [(relaxamount_s + tempdt * i, relaxamount_v + tempdt2 * i) for i in range(1, relaxlen + 1)]

            # find indices with overlapping relax, add relax together
            if veh.relax_end < timeind + relaxlen:
                overlap_end = veh.relax_end
                veh.relax_end = timeind + relaxlen
                need_extend = True
            else:
                overlap_end = timeind + relaxlen
                need_extend = False
            prevr_indoffset = timeind - veh.relax_start + 1
            prevr = veh.relax
            overlap_len = overlap_end - timeind
            for i in range(overlap_len):
                curtime = prevr_indoffset + i
                prevrelax, currelax = prevr[curtime], curr[i]
                prevr[curtime] = (prevrelax[0] + currelax[0], prevrelax[1] + currelax[1])
            if need_extend:
                prevr.extend(curr[overlap_len:])
    else:  # normal case (no existing relax)
        veh.in_relax = True
        apply_normal_relaxation(veh, rp, relaxamount_s, relaxamount_v, timeind, dt, relaxlen)


def apply_normal_relaxation(veh, rp, relaxamount_s, relaxamount_v, timeind, dt, relaxlen):
    tempdt = -dt / rp * relaxamount_s
    tempdt2 = -dt / rp * relaxamount_v
    curr = [(relaxamount_s + tempdt * i, relaxamount_v + tempdt2 * i) for i in range(1, relaxlen + 1)]

    veh.relax = curr
    veh.relax_start = timeind + 1
    veh.relax_end = timeind + relaxlen


def default_parameters(truck_prob=0., stochasticity=True):
    """Suggested parameters for the IDM and havsim lane changing model."""
    npr = np.random.default_rng()
    is_car = True if truck_prob == 0. else npr.random() > truck_prob
    if is_car:
        cf_parameters = [37.5, 1.12, 3, 1.7, 1.5]
        lc_parameters = [-10, -10, .3, .03, 0, 0, .1, 5, 100]
        lc2_parameters = [1, 2, 1, -1, 1, .5]
        relax_parameters = [11., 4.5, .6, 2.]
        route_parameters = [300, 500]
        length = 4
        accbounds = [-10, None]
    else:
        cf_parameters = [34, 1.3, 6, 1.1, 1.6]
        lc_parameters = [-10, -10, 1, .1, 0, 0, .1, 5, 100]
        lc2_parameters = [1, 2, 1, -.5, 1, .5]
        relax_parameters = [9., 4.5, .6, 2.]
        route_parameters = [500, 1000]
        length = 22
        accbounds = [-9.5, None]
    if stochasticity:
        s1 = 2*npr.normal()
        s1 = s1 if s1 > 0 else s1/4
        cf_parameters[0] += s1
        cf_parameters[1] += npr.random()*.1
        cf_parameters[3] = cf_parameters[3] * (.9 + npr.random()*.3)
        length = length * (.8 + npr.random()*.4) if is_car else length * (.85 + npr.random()*.2)
        s2 = npr.random()
        accbounds[0] += s2
        lc_parameters[0] += s2
        lc_parameters[1] += s2

    return {'cf_parameters': cf_parameters, 'lc_parameters': lc_parameters, 'lc2_parameters': lc2_parameters,
            'relax_parameters': relax_parameters, 'route_parameters': route_parameters, 'accbounds': accbounds,
            'length': length}


def stochastic_lc_havsim(veh, lc_actions, lc_followers, timeind):
    """lc_havsim model for StochasticVehicle."""
    if veh.chk_disc:
        if veh.npr.random() > veh.lc_parameters[6]:
            if veh.has_coop:  # remove coop
                coop_veh = veh.has_coop
                coop_veh.chk_disc = timeind > coop_veh.disc_endtime if coop_veh.in_disc else False
                coop_veh.is_coop = veh.has_coop = None
            return lc_actions, lc_followers

    # get relevant accelerations/headways
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

    xi = veh.sample_xi(timeind)
    if veh.in_disc:  # discretionary update
        if veh.is_coop:  # if cooperating, make sure the incentive includes lc_acceleration
            if veh.lc_acc == 0:
                ego_veh = veh.is_coop
                ego_safety = ego_veh.lc_parameters[0] if ego_veh.in_disc else ego_veh.lc_parameters[1]
                check_coop_and_apply(veh, ego_veh, veh.lc2_parameters, ego_safety, timeind)
            incentive -= veh.lc_acc

        incentive = incentive + 2 * p[3] * xi
        if incentive > p[2]:
            if timeind < veh.disc_cooldown:
                return lc_actions, lc_followers
            safety = p[0]
            stochastic_safety = safety - xi
            fol_safe, veh_safe = new_lcfol_a > stochastic_safety, new_veh_a > stochastic_safety
            if fol_safe and veh_safe:
                return complete_change(lc_actions, lc_followers, veh, new_lcfol)
            if veh.chk_disc:
                veh.chk_disc = False
            veh.disc_endtime = timeind + p[7]

            # apply tactical and cooperative
            if veh.lc_acc == 0:
                apply_tactical_discretionary(veh, veh_safe, fol_safe, veh.lc2_parameters)
            has_coop = veh.has_coop
            if not has_coop:
                try_find_new_coop(new_lcfol, veh, new_lcfol_a, safety, timeind, 0)
            elif has_coop.lc_acc == 0:
                check_coop_and_apply(has_coop, veh, has_coop.lc2_parameters, safety, timeind)

        else:
            if veh.has_coop:
                coop_veh = veh.has_coop
                coop_veh.chk_disc = timeind > coop_veh.disc_endtime if coop_veh.in_disc else False
                coop_veh.is_coop = veh.has_coop = None

        if not veh.chk_disc:
            if timeind > veh.disc_endtime:
                veh.chk_disc = True

    else:  # mandatory update
        safety = p[1]
        stochastic_safety = safety - xi
        fol_safe, veh_safe = new_lcfol_a > stochastic_safety, new_veh_a > stochastic_safety
        if fol_safe and veh_safe:
            return complete_change(lc_actions, lc_followers, veh, new_lcfol)

        # apply tactical and cooperative
        if veh.lc_acc == 0:
            apply_tactical(veh, new_lcfol, veh_safe, fol_safe, veh.lc2_parameters)
        has_coop = veh.has_coop
        if not has_coop:
            s1, s2 = veh.lc_urgency
            coop_correction = (veh.pos - s1)/(s2 - s1 + 1)
            try_find_new_coop(new_lcfol, veh, new_lcfol_a, safety, timeind, coop_correction)
        elif has_coop.lc_acc == 0:
            check_coop_and_apply(has_coop, veh, has_coop.lc2_parameters, safety, timeind)

    return lc_actions, lc_followers

