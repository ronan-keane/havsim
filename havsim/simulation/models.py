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


def havsim_mobil(veh, lc_actions, newlfolhd, newlhd, newrfolhd, newrhd, newfolhd, timeind, dt):
    """Minimizing total braking during lane change (MOBIL) lane changing decision model.

   The mobil is a dicretionary/incentive lane changing model, and we use the safety conditions
    proposed by Treiber, Kesting to accompany mobil (see their book (Traffic Flow Dynamics, 2013)).
    We added an original tactical/cooperative model which uses Vehicle's shift_eql method
    to modify the car following acceleration (see coop_tact_model). We also add a probability
    of checking the discretionary model, and a cooldown period after making a discretionary change, during
    which vehicles cannot make another discretionary change.
    We also use the concept of an 'activated' discretionary state, where after meeting their
    incentive criteria, vehicles will always check the LC/tactical/cooperation model (whereas in the
    normal discretionary state, vehicles only have a probability to check the LC model)

    parameters of IDM-
        0 - safety criteria (maximum deceleration allowed after LC, more negative = less strict),
        1 - safety criteria during mandatory lane changing
        2 - incentive criteria (>0, larger = more strict. smaller = discretionary changes more likely),
        3 - politeness (taking other vehicles into account, 0 = ignore other vehicles, ~.1-.2 = realistic),
        4 - bias on left side (can add a bias to make vehicles default to a certain side of the road),
        5 - bias on right side,
        6 - probability of checking LC while in discretionary state (not in original model. set to 1
            to always check discretionary. units are probability of checking per timestep)
        7 - number of timesteps in cooperative/tactical state after meeting incentive criteria for
            a discretionary change (not in original model)
        8 - number of timesteps after a discretionary change when another discretionary change is not
            possible (not in original model)

    naming convention - 'l/r' = left/right respectively, current lane if no l/r
    'new' indicates that it would be the configuration after potential change
    e.g. newlfolhd = new left follower headway - the headway of the
    left follower if the ego vehicle changed left

    Information on lane changing models -
    There are two types of lane changes, discretionary and mandatory. In a discretionary change, you change
    lanes in order to improve your driving situation, i.e. to change to a faster/more comfortable lane.
    In a mandatory change, you have to change lanes in order to follow your route (e.g. taking an exit or
    merging from on-ramp onto the main road).
    To make a discretionary change, you have to meet an incentive condition as well as a safety condition.
    The incentive condition encodes how beneficial the lane change is to the ego vehicle, and the safety
    condition is a binary decision which states whether or not the lane change will potentially cause a
    collision. To make a mandatory change, only the safety condition must be met.
    In the mobil model, the incentive and safety conditions are based off of the accelerations of
    the vehicles, meaning the car following model is used to determine the safety and incentive conditions.
    The incentive is the difference of ego vehicle's acceleration in the current lane with
    the acceleration in the new lane; the incentive must pass a threshold to trigger a discretionary
    lane change. The safety condition is based off the ego vehicle's acceleration in the new lane and
    the left follower's acceleration in the new lane; both must be above a threshold for the change to be
    considered safe.

    Args:
        veh: ego vehicle
        lc_actions: dictionary where lane changing actions are stored, keys = vehicles, values = 'l' or 'r'
        newlfolhd: new left follower headway
        newlhd: new vehicle headway for left change
        newrfolhd: new right follower headway
        newrhd: new vehicle headway for right change
        newfolhd: new follower headway
        timeind: time index
        dt: time step

    Returns:
        None. (modifies lc_actions in place)
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
                # todo other stuff based on what is needed
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

    if veh.in_disc:
        if timeind < veh.disc_cooldown:
            return lc_actions, lc_followers


    p = veh.lc_parameters
    lincentive = rincentive = -math.inf
    in_disc = veh.in_disc

    # if calculating incentives, need to calculate veha, fola, newfola
    if in_disc:
        fol = veh.fol
        veha = veh.acc
        fola = fol.acc
        newfola = fol.get_cf(newfolhd, fol.speed, veh.lead, fol.lane, timeind, dt, False)
    else:
        veha = fola = newfola = 0

    # calculate safeties and incentives for each side
    if veh.lside:
        # safeguard for negative/zero headway (needed for IDM, not necessarily needed for other models)
        use_newlfolhd = max(newlfolhd, 1e-6)
        newlhd = max(newlhd, 1e-6) if newlhd is not None else None
        newlfola, newla, lincentive = mobil_helper(p[3], p[4], in_disc, veh.lfol, veh.lfol.lead, veh,
                                                   veh.llane, use_newlfolhd, newlhd, veha, fola, newfola,
                                                   timeind, dt, userelax_cur, userelax_new)
    if veh.rside:
        # safeguard for negative/zero headway (needed for IDM, not necessarily needed for other models)
        use_newrfolhd = max(newrfolhd, 1e-6)
        newrhd = max(newrhd, 1e-6) if newrhd is not None else None
        newrfola, newra, rincentive = mobil_helper(p[3], p[5], in_disc, veh.rfol, veh.rfol.lead, veh,
                                                   veh.rlane, use_newrfolhd, newrhd, veha, fola, newfola,
                                                   timeind, dt, userelax_cur, userelax_new)

    # determine which side we want to potentially intiate LC for
    if rincentive > lincentive:
        side = 'r'
        incentive = rincentive

        newlcsidefolhd = newrfolhd
        vehsafe = newra
        lcsidefolsafe = newrfola
        lcsidefol = veh.rfol

    else:
        side = 'l'
        incentive = lincentive

        newlcsidefolhd = newlfolhd
        vehsafe = newla
        lcsidefolsafe = newlfola
        lcsidefol = veh.lfol

    # safe = veh.speed/veh.maxspeed
    # safe = safe*p[0] + (1-safe)*p[1]  # safety based off relative velocity

    # determine if LC can be completed, and if not, determine if we want to enter cooperative or
    # tactical states. update the internal lc state accordingly
    if in_disc:
        # version 1 - must continually meet incentive to stay in tactical/cooperative state
        # if incentive > p[2]:
        # version 2 - only need to meet incentive to trigger tactical/cooperative state
        if incentive > p[2] or veh.chk_lc:
            if vehsafe > safe and lcsidefolsafe > safe:
                lc_actions[veh] = side
            else:
                if veh.chk_lc == False:  # always check discretionary on same side for next p[6] timesteps
                    veh.chk_lc = True
                    veh.disc_endtime = timeind + p[7]
                    if side == 'r':
                        veh.lside = False
                    else:
                        veh.rside = False
                elif timeind > veh.disc_endtime:  # end always check discretionary state
                    veh.chk_lc = False
                    if side == 'r':
                        if veh.l_lc is not None:
                            veh.lside = True
                    elif veh.r_lc is not None:
                        veh.rside = True
                coop_tact_model(veh, newlcsidefolhd, lcsidefolsafe, vehsafe, safe, lcsidefol, in_disc,
                                use_coop=use_coop, use_tact=use_tact)
        # elif veh.chk_lc == True:  # incentive not met -> end always check discretionary state
        # # (redundant for version 2)
        #     veh.chk_lc = False

    else:  # mandatory state
        if vehsafe > safe and lcsidefolsafe > safe:
            lc_actions[veh] = side
        else:
            coop_tact_model(veh, newlcsidefolhd, lcsidefolsafe, vehsafe, safe, lcsidefol, in_disc,
                            use_coop=use_coop, use_tact=use_tact)
    return


def new_hd_acc_under_lc(lcfol, veh, timeind):
    new_lcfol_hd, new_lcfol_a = lcfol.get_cf(veh, timeind)
    new_veh_hd, new_veh_a = veh.get_cf(lcfol.lead, timeind)
    return new_lcfol_hd, new_veh_hd, new_lcfol_a, new_veh_a


def mobil_helper(polite, bias, in_disc, lfol, llead, veh, vehlane, newlfolhd, newlhd, veha, fola, newfola,
                 timeind, dt, userelax_cur, userelax_new):
    """Helper function for MOBIL computes safeties and incentives.

    Args:
        polite: politeness parameter in mobil
        bias: bias for lane change side in mobil
        in_disc: True if the considered lane change is discretionary, in which case we need to compute
            the incentive.
        lfol: the left (or right) follower
        llead: left leader (leader of lfol)
        veh: ego vehicle (for which the LC is considered)
        vehlane: lane veh is evaluating the change for
        newlfolhd: new headway for lfol
        newlhd: new headway for veh
        veha: current vehicle acceleration, used for incentive
        fola: current follower (veh.fol) acceleration, used for incentive
        newfola: new follower acceleration
        timeind: time index
        dt: time step
        userelax_cur: If True, apply relaxation in current acceleration. True recommended.
        userelax_new: If True, apply relaxation in new acceleration. False recommended.

    Returns:
        newlfola: new acceleration for left follower (used for safety condition)
        newla: new acceleration for ego vehicle (used for safety condition)
        lincentive: computed incentive (0 if mandatory change)
    """
    userelax = userelax_new and lfol.in_relax
    newlfola = lfol.get_cf(newlfolhd, lfol.speed, veh, lfol.lane, timeind, dt, userelax)

    userelax = userelax_new and veh.in_relax
    newla = veh.get_cf(newlhd, veh.speed, llead, vehlane, timeind, dt, userelax)

    if in_disc:
        if not userelax_cur and lfol.in_relax:
            lfola = lfol.get_cf(lfol.hd, lfol.speed, llead, lfol.lane, timeind, dt, False)
        else:
            lfola = lfol.acc

        lincentive = newla - veha + polite*(newlfola - lfola + newfola - fola) + bias
        return newlfola, newla, lincentive

    else:
        return newlfola, newla, 0

def get_new_hd_and_safety(new_fol, veh, timeind):
    new_acc = veh.get_cf(new_hd, veh.speed, new_lead, timeind)

    new_fol_hd = get_headway(new_fol, veh)
    new_fol_acc = new_fol.get_cf(new_fol_hd, new_fol.speed, veh, timeind)
    return new_hd, new_acc, new_fol_hd, new_fol_acc

def get_fol_safety(fol, new_lead, timeind):
    if new_lead is None:
        new_hd = None
    else:
        new_hd = get_headway(fol, new_lead)
    new_acc = fol.get_cf(new_hd, fol.speed, new_lead, timeind)


def coop_tact_model(veh, newlcsidefolhd, lcsidefolsafe, vehsafe, safe, lcsidefol, in_disc, use_coop=True,
                    use_tact=True, jam_spacing=2):
    """Cooperative and tactical model for a lane changing decision model.

    Explanation of model -
    first we assume that we can give vehicles one of two commands - accelerate or decelerate. These commands
    cause a vehicle to give more space, or less space, respectively. See any shift_eql function.
    There are three possible options - cooperation and tactical, or only cooperation, or only tactical

    In the tactical model, first we check the safety conditions to see what is preventing us from
    changing (either lcside fol or lcside lead). If only the follower's safety is violated, the vehicle
    accelerates. If the vehicle's own safety is violated; the vehicle decelerates. In the case where both
    safeties are violated, the vehicle accelerates. The tactical model only modifies the acceleration of veh.

    in the cooperative model, we try to identify a cooperating vehicle. A cooperating vehicle gets a
    deceleration added so that it will give extra space to let the ego vehicle successfully change lanes.
    Cooperation is only applied if the lcside fol is blocking the ego vehicle; if only the lcside leader
    is blocking, only the tactical model will be applied.
    If the cooperation is applied without tactical, then the cooperating vehicle must be the lcside follower,
    and the newlcsidefolhd must be > jam spacing. This prevents cooperation with vehicles that are right next
    to you.
    if cooperation is applied with tactical, then in addition to the above, it's also possible the
    cooperating vehicle is the lcside follower's follower, where additionally the newlcsidefolhd is
    < jam spacing, but the headway between the lcside follower's follower and veh is > jam spacing.
    In this case, the lcside follower cannot cooperate, so we allow its follower to cooperate instead.
    In the first case where the cooperating vehicle is the lcside follower, the tactical model is applied
    as normal.
    In the second case, since the issue is the the lcside follower is directly blocking the vehicle,
    the vehicle accelerates if the lcside follower has a slower speed than vehicle, and decelerates otherwise.
    The cooperative model only modifies the acceleration of the cooperating vehicle.

    when a vehicle requests cooperation, it has to additionally fulfill a condition which simulates
    the choice of the cooperating vehicle. All vehicles have a innate probability (coop_parameters attribute)
    of cooperating, and for a discretionary LC, this innate probability controls whether or not the
    cooperation is accepted.
    For a mandatory LC, vehicle can add to this probability, which simulates forcing the cooperation.
    vehicles have a lc_urgency attribute which is updated upon initiating a mandatory change. lc_urgency is a
    tuple of two positions, at the first position, only the follower's innate cooperation probability
    is used. at the second position, the follower is always forced to cooperate, even if it has 0
    innate cooperation probability, and the additional cooperation probability is interpolated linearally
    between these two positions.

    Implementation details -
    vehicles also have a coop_veh attribute which stores the cooperating vehicle. A cooperating
    vehicle does not have any attribute marking it as such. A vehicle's shift_eql method is what is
    used to widen/shrink gaps to allow lane changing.

    Args:
        veh: vehicle which wants to change lanes
        newlcsidefolhd: new lcside follower headway
        lcsidefolsafe: safety value for lcside follower; viable if > safe
        vehsafe: safety value for vehicle; viable if > safe
        safe: safe value for change
        lcsidefol: lane change side follower of veh
        in_disc: True if change is discretionary
        use_coop: bool, whether to apply cooperation model. if use_coop and use_tact are both False,
            this function does nothing.
        use_tact: bool, whether to apply tactical model
        jam_spacing: float, headway such that (jam_spacing, 0) is equilibrium solution for coop_veh

    Returns:
        None (modifies veh, veh.coop_veh)
    """
    # clearly it would be possible to modify different things, such as how the acceleration modifications
    # are obtained, and changing the conditions for entering/exiting the cooperative/tactical conditions
    # in particular we might want to add extra conditions for entering cooperative state
    
    coop_veh_is_lcsidefolfol = False
    if use_coop and use_tact and lcsidefolsafe < safe:
        coop_veh = veh.coop_veh
        if coop_veh is not None:  # first, check cooperation is valid, and apply cooperation if so
            if coop_veh is lcsidefol and newlcsidefolhd > jam_spacing:  # coop_veh = lcsidefol
                coop_veh.acc += coop_veh.shift_eql('decel')

            elif coop_veh is lcsidefol.fol and newlcsidefolhd < jam_spacing < coop_veh.hd + \
                    newlcsidefolhd + lcsidefol.len:  # coop_veh = lcsidefol.fol
                coop_veh.acc += coop_veh.shift_eql('decel')
                coop_veh_is_lcsidefolfol = True

            else:  # cooperation is not valid -> reset
                veh.coop_veh = None

        # if there is no coop_veh, then see if we can get vehicle to cooperate
        if veh.coop_veh is None and lcsidefol.cf_parameters is not None:
            coop_veh = None
            if newlcsidefolhd > jam_spacing:
                coop_veh = lcsidefol
            elif lcsidefol.fol.cf_parameters is not None and \
            lcsidefol.fol.hd+lcsidefol.len + newlcsidefolhd > jam_spacing:
                coop_veh = lcsidefol.fol
                coop_veh_is_lcsidefolfol = True

            if coop_veh is not None and check_if_veh_cooperates(veh, coop_veh, in_disc):
                veh.coop_veh = coop_veh
                coop_veh.acc += coop_veh.shift_eql('decel')

    elif not use_tact and use_coop and lcsidefolsafe < safe:
        coop_veh = veh.coop_veh
        if coop_veh is not None:
            if coop_veh is lcsidefol and newlcsidefolhd > jam_spacing:  # coop_veh = lcsidefol
                coop_veh.acc += coop_veh.shift_eql('decel')
            else:  # cooperating vehicle not valid -> reset
                veh.coop_veh = None
        if veh.coop_veh is None and lcsidefol.cf_parameters is not None and newlcsidefolhd > jam_spacing:
            # vehicle is valid, check cooperation condition
            if check_if_veh_cooperates(veh, lcsidefol, in_disc):
                veh.coop_veh = lcsidefol
                lcsidefol.acc += lcsidefol.shift_eql('decel')

    if use_tact:
        tactical_model(veh, lcsidefol, lcsidefolsafe, vehsafe, safe, coop_veh_is_lcsidefolfol)


def tactical_model(veh, lcsidefol, lcsidefolsafe, vehsafe, safe, coop_veh_is_lcsidefolfol):
    """Applies tactical model (see coop_tact_model for explanation)."""
    if coop_veh_is_lcsidefolfol:  # special rule for cooperation by lcsidefol.fol
        if lcsidefol.speed > veh.speed:
            tactstate = 'decel'
        else:
            tactstate = 'accel'
        veh.acc += veh.shift_eql(tactstate)
    else: # in normal rule, you find the safety that is blocking and move to widen that gap.
        if lcsidefolsafe < safe:
            tactstate = 'accel'

            # if both lcsidefol and lcsidelead are blocking, could look at lcsidelead speed
            # if vehsafe < safe:  # both unsafe
            #     if lcsidefol.lead.speed > veh.speed:  # edge case where lcsidefol.lead is None?
            #     # if lcsidefol.lead is not None and lcsidefol.lead.speed > veh.speed:
            #         tactstate = 'decel'
            #     else:
            #         tactstate = 'accel'
            # else:  # only follower unsafe
            #     tactstate = 'accel'

        else:  # only leader unsafe
            tactstate = 'decel'
        veh.acc += veh.shift_eql(tactstate)


def check_if_veh_cooperates(veh, coop_veh, in_disc):
    """Calculates condition for coop_veh to cooperate with veh. Returns bool (see coop_tact_model)."""
    coop = coop_veh.coop_parameters  # baseline probability for cooperation
    if not in_disc:
        start, end = veh.lc_urgency
        coop += (veh.pos - start)/(end - start+1e-6)
    return np.random.rand() < coop


def IDM_parameters(*args):
    """Suggested parameters for the IDM/MOBIL."""
    # time headway parameter = 1 -> always unstable in congested regime.
    # time headway = 1.5 -> restabilizes at high density
    cf_parameters = [35, 1.3, 2, 1.1, 1.5]  # note speed is supposed to be in m/s
    
    # note last 3 parameters have units in terms of timesteps, not seconds
    lc_parameters = [-8, -20, .6, .1, 0, .2, .1, 20, 20]

    return cf_parameters, lc_parameters
