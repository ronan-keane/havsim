
"""
The Lane class and boundary conditions.
"""
import numpy as np
import math


def downstream_wrapper(method='speed', time_series=None, congested=True, merge_side='l',
                       merge_anchor_ind=None, target_lane=None, self_lane=None, shift=1, minacc=-2,
                       stopping='car following'):
    """Defines call_downstream method for Lane. keyword options control behavior of call_downstream.

    call_downstream is used instead of the cf model in order to get the acceleration in cases where there is
    no lead vehicle (e.g. because the lead vehicle has left the simulation). Essentially, call_downstream
    will determine the rate at which vehicles can exit the simulation.
    call_downstream also gets used for on-ramps/merges where you have no leader not because you are
    leaving the simulation, but rather because the lane is about to end and you need to move over.
    Lastly, if the simulation starts with no Vehicles, call_downstream will entirely determine
    The keyword arg 'method' defines the behavior of call_downstream method for a Lane.

    Args:
        method: one of 'speed', 'free', 'flow', 'free merge', 'merge'
            'speed' - Give a function which explicitly returns the speed, and we compute the acceleration.
                Options -
                time_series: function takes in timeind, returns speed

            'free' - We use the vehicle's free_cf method to update the acceleration. This is as if vehicles
            can exit the simulation as quickly as possible.
                Options -
                None.

            'flow' - We get a flow from time_series; we then use the vehicle's inverse flow method to find
            the speed corresponding to the flow
                Options -
                time_series: function takes in timeind, returns flow
                congested: whether to assume flow is in congested or free flow branch

            'free merge' - We use the vehicle's free flow method to update, unless we are getting too close
            to the end of the road, in which case we ensure the Vehicle will stop before reaching the end of
            the lane. We assume that vehicles stop as if there was a stationary lead vehicle at the end of
            the lane. This is done by creating an AnchorVehicle at the end of the lane which is used as a
            lead vehicle.
                Options -
                minacc: We always compute the vehicle's acceleration using the anchor as a leader. If the
                    vehicle has an acceleration more negative than minacc, we use the anchor as a leader.
                    otherwise, we use the vehicle's free flow method
                stopping: if 'car following', we use the strategy with minacc. if 'ballistic', we stop
                    only when it is necessary as determined by the vehicle's minimum acceleration
                self_lane: vehicle needs to stop when reaching the end of self_lane
                time_series: Can use either free flow method or time_series; we use time_series if it is
                    not None

            'merge' - This is meant to give a longitudinal update in congested conditions while on a
            bottleneck (on ramp or lane ending, where you must merge). minacc and self_lane give behavior
            the same as in 'free merge'.
                Options -
                merge_side: either 'l' or 'r' depending on which side vehicles need to merge
                target_lane: lanes vehicles need to merge into
                merge_anchor_ind: index for merge anchor in target_lane
                self_lane: if not None, the vehicle needs to stop at the end of self_lane. If None, the
                    vehicle won't stop.
                minacc: if the acceleration needed to stop is more negative than minacc, we begin to stop.
                stopping: if 'car following', we use the strategy with minacc. if 'ballistic', we stop
                    only when it is necessary as determined by the vehicle's minimum acceleration
                shift: we infer a speed based on conditions in target_lane. We do this by shifting the speed
                    of a vehicle in target_lane by shift. (e.g. shift = 1, use the speed from 1 second ago)
                time_series: if we aren't stopping at the end of self_lane,  and can't find a vehicle to infer
                    the speed from, time_series controls the behavior. If None, the vehicle uses its free cf
                    method. Otherwise, time_series specifies a speed which is used.

    Returns:
        call_downstream method for a Lane. Takes in (veh, timeind, dt) and returns acceleration.
    """
    # options - time_series
    if method == 'speed':  # specify a function which takes in time and returns the speed
        def call_downstream(self, veh, timeind, dt):
            speed = time_series(timeind)
            return veh.acc_bounds((speed - veh.speed)/dt)
        return call_downstream

    # options - none
    elif method == 'free':  # use free flow method of the vehicle
        def free_downstream(self, veh, *args):
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream

    # options - time_series, congested
    elif method == 'flow':  # specify a function which gives the flow, we invert the flow to obtain speed
        def call_downstream(self, veh, timeind, dt):
            flow = time_series(timeind)
            speed = veh.inv_flow(flow, output_type='v', congested=congested)
            return (speed - veh.speed)/dt
        return call_downstream

    # options - minacc, self_lane
    elif method == 'free merge':  # use free flow method of the vehicle, stop at end of lane
        endanchor = AnchorVehicle(self_lane, None)
        endanchor.pos = self_lane.end

        def free_downstream(self, veh, timeind, dt):
            hd = get_headway(veh, endanchor)

            # more aggressive breaking strategy is based on car following model
            if stopping[0] == 'c':
                acc = veh.get_cf(hd, veh.speed, endanchor, veh.lane, timeind, dt, veh.in_relax)
                if acc < minacc:
                    return acc

            # another strategy is to only decelerate when absolutely necessary
            else:
                if hd < veh.speed**2*.5/-veh.minacc+dt*veh.speed:
                    return veh.minacc
            if time_series is not None:
                return (time_series(timeind) - veh.speed)/dt
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream
    # options - merge_side, merge_anchor_ind, target_lane, self_lane, shift, minacc, time_series
    elif method == 'merge':
        # first try to get a vehicle in the target_lane and use its shifted speed. Cannot be an AnchorVehicle
        # if we fail to find such a vehicle and time_series is not None: we use time_series
        # otherwise we will use the vehicle's free_cf method
        if merge_side == 'l':
            folside = 'lfol'
        elif merge_side == 'r':
            folside = 'rfol'
        if self_lane is not None:
            endanchor = AnchorVehicle(self_lane, None)
            endanchor.pos = self_lane.end
        else:
            endanchor = None

        def call_downstream(self, veh, timeind, dt):
            # stop if we are nearing end of self_lane
            if endanchor is not None:
                hd = get_headway(veh, endanchor)
                # more aggressive breaking strategy is based on car following model
                if stopping[0] == 'c':
                    acc = veh.get_cf(hd, veh.speed, endanchor, veh.lane, timeind, dt, veh.in_relax)
                    if acc < minacc:
                        return acc

                # another strategy is to only decelerate when absolutely necessary
                else:
                    if hd < veh.speed**2*.5/-veh.minacc+dt*veh.speed:
                        return veh.minacc

            # try to find a vehicle to use for shifted speed
            # first check if we can use your current lc side follower
            # if that fails, try using the merge anchor for the target_lane.
            # can also try the leader of either of the above.
            fol = getattr(veh, folside)
            if merge_anchor_ind is not None:
                if fol is None:
                    fol = target_lane.merge_anchors[merge_anchor_ind][0]
                if fol.cf_parameters is None:
                    fol = fol.lead
            elif fol is None:
                pass
            elif fol.cf_parameters is None:
                fol = fol.lead

            if fol is not None:  # fol must either be none or a vehicle (can't be anchor)
                speed = shift_speed(fol.speedmem, shift, dt)
            elif time_series is not None:
                speed = time_series(timeind)
            else:
                return veh.free_cf(veh.cf_parameters, veh.speed)
            return (speed - veh.speed)/dt

        return call_downstream


def get_inflow_wrapper(time_series=None, args=(None,), inflow_type='flow'):
    """Defines get_inflow method for Lane.

    get_inflow is used for a lane with upstream boundary conditions to increment the inflow_buffer
    attribute which controls when we attempt to add vehicles to the simulation.

    Args:
        time_series: function which takes in a timeind and returns either a flow (inflow_type = 'flow') or
            speed (inflow_type = 'speed' or 'congested').
        args: tuple of arguments to be passed to arrival_time_inflow if inflow_type is 'arrivals'.
        inflow_type: Method to add vehicles. One of 'flow', 'speed', 'congested'
            'flow' - time_series returns the flow explicitly

            'speed' - time_series returns a speed, we get a flow from the speed using the get_eql method of
            the Vehicle.

            'congested' - This is meant to add a vehicle with ~0 acceleration as soon as it is possible to do
            so. This is similar to 'speed', but instead of getting speed from time_series, we get it from
            the anchor's lead vehicle. This may help remove artifacts from the downstream boundary
            condition caused by simulations with different Vehicle parameters.
            Requires get_eql method of the Vehicle.

            'arrivals' - We sample from some distribution to generate the next (continuous) arrival time.
            When we pass that time in the simulation (time index >= next arrival time), we add 1 flow. *args
            is passed to arrival_time_inflow.__init__

    Returns:
        get_inflow method for a Lane. Takes in (timeind) and returns instantaneous flow, vehicle speed,
        at that time. If we return None for the speed, increment_inflow will obtain the speed.
    """
    # give flow series - simple
    if inflow_type == 'flow':
        def get_inflow(self, timeind):
            return time_series(timeind), None
    # give speed series, we convert to equilibrium flow using the parameters of the next vehicle to be added
    # note that if all vehicles have same parameters/length, this is exactly equivalent to the 'flow' method
    # (where the speeds correspond to the flows for the eql soln being used)
    elif inflow_type == 'speed':
        def get_inflow(self, timeind):
            spd = time_series(timeind)
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
            else:
                leadlen = self.newveh.len
            s = self.newveh.get_eql(spd, find='s')
            return spd / (s + leadlen), spd

    # in congested type it is similar to the 'speed' method but uses the speed from the anchor.lead instead of
    # a speed which is specified a priori. This is basically supposed to add a vehicle with 0 acceleration
    elif inflow_type == 'congested':
        def get_inflow(self, timeind):
            lead = self.anchor.lead
            if lead is not None:
                leadlen = lead.len
                spd = lead.speed
            else:
                leadlen = self.newveh.len
                spd = time_series(timeind)
            s = self.newveh.get_eql(spd, find='s')
            return spd / (s + leadlen), spd

    # in arrivals type we generate arrival times according to some stochastic process and add 1 inflow
    # whenever the time index passes the
    elif inflow_type == 'arrivals':
        get_inflow = arrival_time_inflow(*args)

    return get_inflow


def timeseries_wrapper(timeseries, starttimeind=0):
    """Decorator to convert a list or numpy array into a function which accepts a timeind."""
    def out(timeind):
        return timeseries[timeind-starttimeind]
    return out


class M3Arrivals:
    """Generates random arrival times according to the M3 Model (Cowan, 1975)."""

    def __init__(self, q, tm, alpha):
        """Inits object whose call method generates arrival times.

        Args:
            q (float): the (expected) flow rate q
            tm (float): the minimum possible time headway
            alpha (float): (1- alpha) is the probability having tm arrival time
        """
        self.tm = tm
        self.alpha = alpha
        self.lam = alpha*q/(1-tm*q)   # reads as lambda

    def __call__(self, *args):
        """Returns a random arrival time sampled from the distribution."""
        y = np.random.rand()
        if y >= self.alpha:
            return self.tm
        else:
            return -math.log(y/self.alpha)/self.lam + self.tm


class arrival_time_inflow:
    """Implements get_inflow method for a Lane where inflow is generated by stochastic arrival times."""

    def __init__(self, dist, dt, timeind=0):
        """
        Args:
            dist: calling dist() should generate an arrivals time. Can be a class, e.g. M3Arrivals.
            dt: timestep
            timeind: index of the first inflow
        """
        self.dist = dist  # distribution
        self.dt = dt
        self.next_timeind = timeind + dist(timeind)/dt

    def __call__(self, timeind):
        if timeind >= self.next_timeind:
            self.next_timeind += self.dist(timeind)/self.dt
            return 1
        else:
            return 0


def eql_inflow_congested(curlane, inflow, *args, c=.8, check_gap=True, **kwargs):
    """Condition when adding vehicles for use in congested conditions. Requires to invert flow.

    Suggested by Treiber, Kesting in their traffic flow book for congested conditions. Requires to invert
    the inflow to obtain the equilibrium (speed, headway) for the flow. The actual headway on the road must
    be at least c times the equilibrium headway for the vehicle to be added, where c is a constant.

    Args:
        curlane: Lane with upstream boundary condition, which will possibly have a vehicle added.
        inflow: current instantaneous flow. Note that the 'arrivals' method for get_inflow does not
            return the instantaneous flow, and therefore cannot be naively used in this formulation.
        c: Constant, should be less than or equal to 1. Lower is less strict - Treiber, Kesting suggest .8
        check_gap: If False, we don't check the Treiber, Kesting condition, so we don't have to invert
            the flow. We always just add the vehicle. Gets speed from headway.

    Returns:
        If The vehicle is not to be added, we return None. Otherwise, we return the (pos, spd, hd) for the
        vehicle to be added with.
    """
    # lead = curlane.anchor.lead
    # if lead is None:  # special case where the first vehicle is being added # treated in increment_inflow
    #     leadlen = curlane.newveh.len
    #     spd = curlane.newveh.inv_flow(inflow, leadlen = leadlen, output_type = 'v')
    #     return curlane.start, spd, None

    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    leadlen = lead.len
    if check_gap:  # treiber,kesting condition
        (spd, se) = curlane.newveh.inv_flow(inflow, leadlen=leadlen, output_type='both')
    else:
        se = 2/c
        spd = curlane.newveh.get_eql(hd, input_type='s')

    if hd > c*se:  # condition met
        return curlane.start, spd, hd
    else:
        return None


def eql_inflow_free(curlane, inflow, *args, **kwargs):
    """Suggested by Treiber, Kesting for free conditions. Requires to invert the inflow to obtain velocity."""
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    # get speed corresponding to current flow
    spd = curlane.newveh.inv_flow(inflow, leadlen=lead.len, output_type='v', congested=False)
    return curlane.start, spd, hd


def eql_speed(curlane, *args, c=.8, minspeed=0, **kwargs):
    """Like eql_inflow, but get equilibrium headway from leader's speed instead of inverting flow."""
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = lead.speed
    eqlspd = curlane.newveh.get_eql(hd, input_type='s')
    spd = max(minspeed, max(spd, eqlspd))  # Try to add with largest speed possible

    se = curlane.newveh.get_eql(spd, input_type='v')

    if hd > c*se:
        return curlane.start, spd, hd
    else:
        return None


def shifted_speed_inflow(curlane, inflow, timeind, dt, shift=1, accel_bound=-.5, **kwargs):
    """Extra condition for upstream boundary based on Newell model and a vehicle's car following model.

    We get the first speed for the vehicle based on the shifted speed of the lead vehicle (similar to Newell
    model). Then we compute the vehicle's acceleration using its own car following model. If the acceleration
    is too negative, we don't add it to the simulation. If we add it, it's with the shifted leader speed.

    Args:
        curlane: Lane with upstream boundary condition, which will possibly have a vehicle added.
        dt: timestep
        shift: amount (time) to shift the leader's speed by
        accel_bound: minimum acceleration a vehicle can be added with

    Returns:
        If The vehicle is not to be added, we return None. Otherwise, we return the (pos, spd, hd) for the
        vehicle to be added with.
    """
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = shift_speed(lead.speedmem, shift, dt)

    if accel_bound is not None:
        newveh = curlane.newveh
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound and hd > 0:  # headway required to be positive for IDM
            return curlane.start, spd, hd
        else:
            return None

    return curlane.start, spd, hd


def shift_speed(speed_series, shift, dt):
    """Given series of speeds, returns the speed shifted by 'shift' amount of time.

    speed_series is a list speeds with constant discretization dt. We assume that the last entry in
    speed_series is the current speed, and we want the speed from shift time ago. If shift is not a multiple
    of dt, we use linear interpolation between the two nearest speeds. If shift time ago is before the
    earliest measurement in speed_series, we return the first entry in speed_series.
    Returns a speed.
    """
    ind = int(shift // dt)
    if ind+1 > len(speed_series):
        return speed_series[0]
    remainder = shift - ind*dt
    spd = (speed_series[-ind-1]*(dt - remainder) + speed_series[-ind]*remainder)/dt  # weighted average
    return spd


def newell_inflow(curlane, inflow, timeind, dt, p=[1, 2], accel_bound=-2, **kwargs):
    """Extra condition for upstream boundary based on DE form of Newell model.

    This is like shifted_speed_inflow, but since we use the DE form of the Newell model, there is a maximum
    speed, and there won't be a problem if the shift amount is greater than the length of the lead vehicle
    trajectory (in which case shifted_speed defaults to the first speed).

    Args:
        curlane: Lane with upstream boundary condition
        inflow: unused
        timeind: unused
        dt: timestep
        p: parameters for Newell model, p[0] = time delay = 1/ speed-headway slope. p[1] = jam spacing
        accel_bound: vehicle must have accel greater than this to be added

    Returns: None if no vehicle is to be added, otherwise a (pos, speed, headway) tuple for IC of new vehicle.
    """
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    newveh = curlane.newveh
    spd = max(min((hd - p[1])/p[0], newveh.maxspeed), 0)

    if accel_bound is not None:
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound and hd > 0:
            return curlane.start, spd, hd
        else:
            return None

    return curlane.start, spd, hd


def speed_inflow(curlane, inflow, timeind, dt, speed_series=None, accel_bound=-2, **kwargs):
    """Like shifted_speed_inflow, but gets speed from speed_series instead of the shifted leader speed."""
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = speed_series(timeind)

    if accel_bound is not None:
        newveh = curlane.newveh
        acc = newveh.get_cf(hd, spd, lead, curlane, None, dt, False)
        if acc > accel_bound and hd > 0:
            return curlane.start, spd, hd
        else:
            return None
    return curlane.start, spd, hd


def increment_inflow_wrapper(method='ceql', kwargs={}):
    """Defines increment_inflow method for Lane.

    The increment_inflow method has two parts to it. First, it is responsible for determining when to add
    vehicles to the simulation. It does this by calling the Lane.get_inflow method every timestep, which
    returns the flow amount that timestep, which updates the attribute inflow_buffer. When inflow_buffer >= 1,
    it attempts to add a vehicle to the simulation. There are extra conditions required to add a vehicle,
    which are controlled by the 'method' keyword arg.
    Once it has been determined a new vehicle can be added, this function is also responsible for calling
    the initialize method of the new vehicle, adding the new vehicle with a correct leader/follower
    relationships, and also inits the next vehicle which is to be added.

    Args:
        method: One of 'ceql' (eql_inflow_congested), 'feql' (eql_inflow_free), 'seql' (eql_speed),
            'shifted' (shifted_speed_inflow), 'newell', or 'speed' (speed_inflow) - refer to those functions.
            We suggest using 'seql' method as it seems to provide the most consistent results and works in
            both congested/uncongested conditions, including the transition between those.
        kwargs: dictionary of keyword arguments for the method chosen.

    Returns:
        increment_inflow method -
        Args:
            vehicles: set of vehicles
            vehid: vehicle ID to be used for next created vehicle
            timeind: time index
            dt: timestep
        Returns:
            None. Modifies vehicles, Lanes in place.
    """
    if method == 'ceql':
        method_fun = eql_inflow_congested
    elif method == 'feql':
        method_fun = eql_inflow_free
    elif method == 'seql':
        method_fun = eql_speed
    elif method == 'shifted':
        method_fun = shifted_speed_inflow
    elif method == 'speed':
        method_fun = speed_inflow
    elif method == 'newell':
        method_fun == newell_inflow

    def increment_inflow(self, vehicles, vehid, timeind, dt):
        inflow, spd = self.get_inflow(timeind)
        self.inflow_buffer += inflow * dt

        if self.inflow_buffer >= 1:

            if self.anchor.lead is None:  # rule for adding vehicles when road is empty
                # old rule
                #     if spd is None:
                #         if speed_series is not None:
                #             spd = speed_series(timeind)
                #         else:
                #             spd = self.newveh.inv_flow(inflow, congested=True)

                # new rule
                spd = self.newveh.maxspeed*.9
                out = (self.start, spd, None)
            else:  # normal rule for adding vehicles
                out = method_fun(self, inflow, timeind, dt, **kwargs)

            if out is None:
                return vehid
            # add vehicle with the given initial conditions
            pos, speed, hd = out[:]
            newveh = self.newveh
            anchor = self.anchor
            lead = anchor.lead
            newveh.lead = lead

            # initialize state
            newveh.initialize(pos+1e-6, speed, hd, timeind+1)

            # update leader/follower relationships######
            # leader relationships
            if lead is not None:
                lead.fol = newveh
            for rlead in anchor.rlead:
                rlead.lfol = newveh
            newveh.rlead = anchor.rlead
            anchor.rlead = []
            for llead in anchor.llead:
                llead.rfol = newveh
            newveh.llead = anchor.llead
            anchor.llead = []

            # update anchor and follower relationships
            # Note that we assume that for an inflow lane, it's left/right lanes start at the same positions,
            # so that the anchors of the left/right lanes can be used as the lfol/rfol for a new vehicle.
            # This is because we don't update the lfol/rfol of AnchorVehicles during simulation.
            anchor.lead = newveh
            anchor.leadmem.append((newveh, timeind+1))
            newveh.fol = anchor

            llane = self.get_connect_left(pos)
            if llane is not None:
                leftanchor = llane.anchor
                newveh.lfol = leftanchor
                leftanchor.rlead.append(newveh)
            else:
                newveh.lfol = None
            rlane = self.get_connect_right(pos)
            if rlane is not None:
                rightanchor = rlane.anchor
                newveh.rfol = rightanchor
                rightanchor.llead.append(newveh)
            else:
                newveh.rfol = None

            # update simulation
            self.inflow_buffer += -1
            vehicles.add(newveh)

            # create next vehicle
            self.new_vehicle(vehid)
            vehid = vehid + 1
        return vehid

    return increment_inflow


class AnchorVehicle:
    """Anchors are 'dummy' Vehicles which can be used as placeholders (e.g. at the beginning/end of Lanes).

    Anchors are used at the beginning of Lanes, to maintain vehicle order, so that vehicles will have
    a correct vehicle order upon being added to the start of the lane. Because of anchors, it is not
    possible for vehicles to have None as a fol/lfol/rfol attribute (unless the left/right lane don't exist).
    Anchors can also be used at the end of the lanes, e.g. to simulate vehicles needing to stop because
    of a traffic light or because the lane ends.
    All Lanes have an anchor attribute which is an AnchorVehicle at the start of the Lanes' track. A track
    is a continuous series of lanes such that a vehicle can travel on all the constituent lanes without
    performing any lane changes (i.e. the end of any lane in the track connects to the start of the next
    lane in the track). Therefore comparing Lane's anchors can also be used to compare their tracks.
    Compared to Vehicles, Anchors don't have a cf or lc model, have much fewer attributes, and don't have
    any methods which update their attributes.
    The way we check for anchors is because they have cf_parameters = None.

    Attributes:
        cf_parameters: Always None, used to identify a vehicle as being an anchor
        lane, road, lfol, rfol, lead, rlead, llead, all have the same meaning as for Vehicle
        pos: position anchor is on, used for headway/dist calculations
        speed: speed of anchor (acts as placeholder)
        acc: acceleration of anchor (acts as placeholder)
        hd: always None
        len: length of anchor, should be 0
        leadmem: same format as Vehicle
    """

    def __init__(self, curlane, start, lead=None, rlead=None, llead=None):
        """Inits for Anchor."""
        self.cf_parameters = None
        self.lane = curlane
        self.road = curlane.road['name']

        self.lfol = None  # anchor vehicles just need the lead/llead/rlead attributes. no need for (l/r)fol
        self.rfol = None
        self.lead = lead
        self.rlead = [] if rlead is None else rlead
        self.llead = [] if llead is None else llead

        self.pos = curlane.start
        self.speed = 0
        self.acc = 0
        self.in_relax = False  # for mobil model
        self.hd = None
        self.len = 0

        self.leadmem = [[lead, start]]

    def get_cf(self, *args):
        """Dummy method returns 0 for cf model - so we don't have to check for anchors when calling set_lc."""
        return 0

    def set_relax(self, *args):
        """Dummy method does nothing - it's so we don't have to check for anchors when applying relax."""
        pass

    def __repr__(self):
        """Representation in ipython console."""
        return 'anchor for lane '+str(self.lane)

    def __str__(self):
        """Convert to string."""
        return self.__repr__()


def get_headway(veh, lead):
    """Calculates distance from Vehicle veh to the back of Vehicle lead."""
    hd = lead.pos - veh.pos - lead.len
    if veh.road != lead.road:
        hd += veh.lane.roadlen[lead.road]  # lead.road is hashable because its a string
    return hd


def get_dist(veh, lead):
    """Calculates distance from veh to the front of lead."""
    dist = lead.pos-veh.pos
    if veh.road != lead.road:
        dist += veh.lane.roadlen[lead.road]  # lead.road is hashable because its a string
    return dist


class Lane:
    """Basic building block for roads/road networks.

    Lanes are responsible for defining the topology (e.g. when lanes start/end, which lanes connect
    to what, what it is possible to change left/right into), which are handled by the events attribute.
    Lanes are the only object with a reference to roads, which are used for making/defining routes.
    They also are responsible for defining distances between vehicles, as positions are relative to the road
    a lane belongs to. This is handled by the roadlen attribute.
    They also define boundary conditions, and are responsible for creating new vehicles and adding them
    to the network.

    Attributes:
        start: starting position of lane
        end: ending position of lane
        road: road dictionary Lane belongs to. A road has keys -
            name: string name of the road
            len: length of road
            laneinds: number of Lanes which belong to the road
            Lanes: every Lane belonging to the road is hashed by its laneind. Lanes are ordered left - right
                where the leftmost lane has index 0 and rightmost laneinds -1
            connect to: tuple with information needed for routes, see make_cur_route
        roadname: string name of road
        laneind: index of Lane
        connect_left: defines left connections for Lane
        connect_right: defines right connections for Lane
        connect_to: what the end of Lane connects to
        anchor: AnchorVehicle for lane
        roadlen: defines distance between Lane's road and other roads.
        merge_anchors: any merge anchors for the lane (see update_merge_anchors)
        events: lane events (see update_lane_events)
    """
    # TODO need a RoadNetwork object, possibly Road object as well.
    # should create common road configurations.
    # E.g. of good design - create road network by specifying types of roads (e.g. road, on/off ramp, merge)
    # add boundary conditions to road network.

    def __init__(self, start, end, road, laneind, connect_left=None, connect_right=None, connect_to=None,
                 downstream=None, increment_inflow=None, get_inflow=None, new_vehicle=None):
        """Inits Lane. Note methods for boundary conditions are defined (and bound) here.

        Args:
            start: starting position for lane
            end: ending position for lane
            road: road dictionary Lane belongs to
            laneind: unique index for Lane (unique to road)
            connect_left: list of tuples where each tuple is a (Lane or None, position) pair such that Lane
                is the left connection of self starting at position.
            connect_right: list of tuples where each tuple is a (Lane or None, position) pair such that Lane
                is the right connection of self starting at position.
            connect_to: Lane or None which a Vehicle transitions to after reaching end of Lane
            downstream: dictionary of keyword args which defines call_downstream method, or None
            increment_inflow: dictionary of keyword args which defines increment_inflow method, or None
            get_inflow: dictionary of keyword args which defines increment_inflow method, or None
            new_vehicle: new_vehicle method
        """
        self.laneind = laneind
        self.road = road
        self.roadname = road['name']
        # starting position/end (float)
        self.start = start
        self.end = end
        # connect_left/right has format of list of (pos (float), lane (object)) tuples where lane
        # is the connection starting at pos
        self.connect_left = connect_left if connect_left is not None else [(start, None)]
        self.connect_right = connect_right if connect_right is not None else [(start, None)]
        self.connect_to = connect_to

        self.roadlen = {self.roadname: 0}
        self.events = []
        self.anchor = AnchorVehicle(self, self.start)
        self.merge_anchors = []

        if downstream is not None:
            self.call_downstream = downstream_wrapper(**downstream).__get__(self, Lane)
        """call_downstream returns an acceleration for Vehicles to use when they have no lead Vehicle.
        See downstream_wrapper for more details on specific methods and different options.

        Args:
            veh: Vehicle
            timeind: time index
            dt: timestep
        Returns:
            acceleration
        """

        if get_inflow is not None:
            self.get_inflow = get_inflow_wrapper(**get_inflow).__get__(self, Lane)
        """refer to get_inflow_wrapper for documentation"""

        if new_vehicle is not None:
            self.new_vehicle = new_vehicle.__get__(self, Lane)
        """new_vehicle generates new instance of Vehicle, and assigns it as the newveh attribute of self."""

        if increment_inflow is not None:
            self.inflow_buffer = 0
            self.newveh = None
            # cf_parameters, lc_parameters, kwargs = self.new_vehicle()  # done in Simulation.__init__
            # self.newveh = vehicle(vehid, self, cf_parameters, lc_parameters, **kwargs)
            self.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(self, Lane)
        """refer to increment_inflow_wrapper for documentation"""

    def get_headway(self, veh, lead):
        """Deprecated. Calculates distance from veh to the back of lead. Assumes veh.road = self.road."""
        hd = lead.pos - veh.pos - lead.len
        if self.roadname != lead.road:
            hd += self.roadlen[lead.road]  # lead.road is hashable because its a string
        return hd

    def get_dist(self, veh, lead):
        """Deprecated. Calculates distance from veh to the front of lead. Assumes veh.road = self.road."""
        dist = lead.pos-veh.pos
        if self.roadname != lead.road:
            dist += self.roadlen[lead.road]  # lead.road is hashable because its a string
        return dist

    def leadfol_find(self, veh, guess, side=None):
        """Find the leader/follower for veh, in the same track as guess (can be a different track than veh's).

        Used primarily to find the new lcside follower of veh. Note that we can't use binary search because
        it's inefficient to store a sorted list of vehicles. Since we are just doing a regular search, guess
        should be close to the leader/follower.

        Args:
            veh: Vehicle to find leader/follower of
            guess: Vehicle in the track we want the leader/follower in.
            side: if side is not None, we make sure that the side leader can actually have veh as a
                opside follower. Only used for the lead Vehicle.

        Returns:
            (lead Vehicle, following Vehicle) in that order, for veh
        """
        if side is None:
            checkfol = None
        elif side == 'r':
            checkfol = 'lfol'
        else:
            checkfol = 'rfol'

        hd = get_dist(veh, guess)
        if hd < 0:
            nextguess = guess.lead
            if nextguess is None:  # None -> reached end of network
                return nextguess, guess
            nexthd = get_dist(veh, nextguess)
            while nexthd < 0:
                # counter += 1
                guess = nextguess
                nextguess = guess.lead
                if nextguess is None:
                    return nextguess, guess
                nexthd = get_dist(veh, nextguess)

            if checkfol is not None and nextguess is not None:
                if getattr(nextguess, checkfol) is None:
                    nextguess = None
            return nextguess, guess
        else:
            nextguess = guess.fol
            if nextguess.cf_parameters is None:
                return guess, nextguess
            nexthd = get_dist(veh, nextguess)
            while nexthd > 0:
                # counter +=1
                guess = nextguess
                nextguess = guess.fol
                if nextguess.cf_parameters is None:  # reached anchor -> beginning of network
                    return guess, nextguess
                nexthd = get_dist(veh, nextguess)

            if checkfol is not None and guess is not None:
                if getattr(guess, checkfol) is None:
                    guess = None
            return guess, nextguess

    def get_connect_left(self, pos):
        """Takes in a position and returns the left connection (Lane or None) at that position."""
        return connect_helper(self.connect_left, pos)

    def get_connect_right(self, pos):
        """Takes in a position and returns the right connection (Lane or None) at that position."""
        return connect_helper(self.connect_right, pos)

    def __hash__(self):
        """Hash Lane based on its road name, and its lane index."""
        return hash((self.roadname, self.laneind))

    def __eq__(self, other):
        """Comparison for Lanes using ==."""
        return self.roadname == other.roadname and self.laneind == other.laneind
        # return self is other

    def __ne__(self, other):
        """Comparison for Lanes using !=."""
        return self is not other

    def __repr__(self):
        """Representation in ipython console."""
        return self.roadname+' ('+str(self.laneind)+')'

    def __str__(self):
        """Convert Lane to a string."""
        return self.__repr__()


def connect_helper(connect, pos):
    """Helper function takes in connect_left/right attribute, position, and returns the correct connection."""
    out = connect[-1][1]  # default to last lane for edge case or case when there is only one possible
    # connection
    for i in range(len(connect)-1):
        if pos < connect[i+1][0]:
            out = connect[i][1]
            break
    return out
