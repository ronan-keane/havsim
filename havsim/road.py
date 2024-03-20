
"""
The Lane and Road classes and boundary conditions.
"""
import numpy as np
import math
import copy


def downstream_wrapper(method='speed', time_series=None, p=1.5, congested=True, merge_side='l',
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
                p: speed adjustment time when following downstream boundary

            'free' - We use the vehicle's free_cf method to update the acceleration. This is as if vehicles
            can exit the simulation as quickly as possible.
                Options -
                None.

            'flow' - We get a flow from time_series; we then use the vehicle's inverse flow method to find
            the speed corresponding to the flow
                Options -
                time_series: function takes in timeind, returns flow
                congested: whether to assume flow is in congested or free flow branch
                p: speed adjustment time when following downstream boundary

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
        call_downstream method for a Lane.
        Args:
            veh: Vehicle which requests the downstream boundary condition
            timeind: int time index
            dt: float timestep
        Returns:
            acceleration: float acceleration which is set for veh
    """
    # options - time_series
    if method == 'speed':  # specify a function which takes in time and returns the speed
        def call_downstream(self, veh, timeind):
            speed = time_series(timeind)
            return veh.acc_bounds((speed - veh.speed)/p)
        return call_downstream

    # options - none
    elif method == 'free':  # use free flow method of the vehicle
        def free_downstream(self, veh, timeind):
            return veh.free_cf(veh.cf_parameters, veh.speed)
        return free_downstream

    # options - time_series, congested
    elif method == 'flow':  # specify a function which gives the flow, we invert the flow to obtain speed
        def call_downstream(self, veh, timeind):
            flow = time_series(timeind)
            speed = veh.inv_flow(flow, output_type='v', congested=congested)
            return veh.acc_bounds((speed - veh.speed)/p)
        return call_downstream

    # options - minacc, self_lane, time_series
    elif method == 'free merge':  # use free flow method of the vehicle, stop at end of lane
        endanchor = AnchorVehicle(self_lane, None)
        endanchor.pos = self_lane.end

        def free_downstream(self, veh, timeind):
            # more aggressive breaking strategy is based on car following model
            if stopping[0] == 'c':
                hd, acc = veh.get_cf(endanchor, timeind)
                if acc < minacc:
                    return acc
            # another strategy is to only decelerate when absolutely necessary
            else:
                hd = get_headway(veh, endanchor)
                if hd < veh.speed**2*.5/-veh.minacc+self.dt*veh.speed:
                    return veh.minacc
            if time_series is not None:
                return (time_series(timeind) - veh.speed)/self.dt
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

        def call_downstream(self, veh, timeind):
            # stop if we are nearing end of self_lane
            if endanchor is not None:
                # more aggressive breaking strategy is based on car following model
                if stopping[0] == 'c':
                    hd, acc = veh.get_cf(endanchor, timeind)
                    if acc < minacc:
                        return acc
                # another strategy is to only decelerate when absolutely necessary
                else:
                    hd = get_headway(veh, endanchor)
                    if hd < veh.speed**2*.5/-veh.minacc+self.dt*veh.speed:
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
                speed = shift_speed(fol.speedmem, shift, self.dt)
            elif time_series is not None:
                speed = time_series(timeind)
            else:
                return veh.free_cf(veh.cf_parameters, veh.speed)
            return (speed - veh.speed)/self.dt

        return call_downstream


def get_inflow_wrapper(inflow_type='flow', time_series=None, args=(None,)):
    """Defines get_inflow method for Lane.

    get_inflow is used for a lane with upstream boundary conditions to increment the inflow_buffer
    attribute which controls when we attempt to add vehicles to the simulation.

    Args:
        inflow_type: Method to add vehicles. One of 'flow', 'flow speed', or 'stochastic' is recommended.
            'flow' - function time_series returns the flow explicitly

            'flow speed' - time_series returns both flow and speed. Note that if the speed is not given, we assume
                a speed close to the maximum speed.

            'speed' - time_series returns a speed, we get a flow from the speed using the get_eql method of
                the Vehicle.

            'congested' - This is meant to add a vehicle with ~0 acceleration as soon as it is possible to do
                so. This is similar to 'speed', but instead of getting speed from time_series, we get it from
                the anchor's lead vehicle.
                Requires get_eql method of the Vehicle.

            'stochastic' - We sample from some distribution to generate the next (continuous) arrival time. This is
                converted to an instantaneous arrival rate. See also StochasticArrivalFlow.
        time_series: function which takes in timeind and returns the requested type ('flow', 'flow speed' or 'speed')
            Args:
                timeind: int time index
            Returns:
                flow: flow in veh/sec if inflow_type = 'flow', or speed in m/s if inflow_type = 'speed'
                speed: speed if inflow_type = 'flow speed'
        args: tuple of arguments to be passed to StochasticArrivalFlow if inflow_type is 'stochastic'.

    Returns:
        get_inflow method for a Lane. (note that get_inflow must be bound to the Lane)
        Args:
            timeind: int time index
        Returns:
            flow: instantaneous flow value (units of veh/second) (float)
            speed: vehicle speed (float) or None. If None, increment_inflow will get the speed automatically. Note
                that the returned speed is only used when the Lane is empty (no leader)
    """
    match inflow_type:
        case 'flow':
            def get_inflow(self, timeind):
                return time_series(timeind), None

        case 'flow speed':
            def get_inflow(self, timeind):
                return time_series(timeind)

        case 'speed':
            def get_inflow(self, timeind):
                spd = time_series(timeind)
                lead = self.anchor.lead
                if lead is not None:
                    leadlen = lead.len
                else:
                    leadlen = self.newveh.len
                s = self.newveh.get_eql(spd, input_type='v')
                return spd / (s + leadlen), spd

        case 'congested':
            def get_inflow(self, timeind):
                lead = self.anchor.lead
                if lead is not None:
                    leadlen = lead.len
                    spd = lead.speed
                else:
                    leadlen = self.newveh.len
                    spd = time_series(timeind)
                s = self.newveh.get_eql(spd, input_type='v')
                return spd / (s + leadlen), spd

        case 'stochastic':
            dist_wrapper = StochasticArrivalFlow(*args)

            def get_inflow(self, timeind):
                return dist_wrapper(timeind), None

        case _:
            raise RuntimeError('invalid inflow_type. Should be one of \'flow\', \'flow speed\', \'speed\', '
                               '\'stochastic\'. Received '+str(inflow_type))

    return get_inflow


def timeseries_wrapper(timeseries, starttimeind=0):
    """Decorator to convert a list or numpy array into a function which accepts a timeind."""
    def out(timeind):
        return timeseries[timeind-starttimeind]
    return out


class M3Arrivals:
    """Generates random arrival times according to the M3 Model (Cowan, 1975)."""

    def __init__(self, q, tm, alpha, max_t=30.):
        """Inits object whose call method generates arrival times.

        Args:
            q (callable): function which accepts timeind, returns current flow rate  (units of veh/sec)
            tm (float): the minimum possible time headway
            alpha (float): (1- alpha) is the probability having tm arrival time
            max_t (float): maximum possible arrival time. Used in case q ~= 0.
        """
        self.q = q
        self.tm = tm
        self.alpha = alpha
        self.max_t = max_t

    def __call__(self, timeind):
        """Returns a random arrival time sampled from the distribution."""
        y = np.random.rand()
        q = self.q(timeind)
        lam = self.alpha*q/(1-self.tm*q) if q != 0 else 1e-6
        if y >= self.alpha:
            return self.tm
        else:
            return min(-math.log(y/self.alpha)/lam + self.tm, self.max_t)


class M3ArrivalsFixed:
    """Like M3Arrivals, but the flow q is constant, instead of a callable which accepts a timeind."""
    def __init__(self, q, tm, alpha):
        self.tm = tm
        self.alpha = alpha
        self.lam = alpha*q/(1-tm*q)   # reads as lambda

    def __call__(self, *args):
        y = np.random.rand()
        if y >= self.alpha:
            return self.tm
        else:
            return -math.log(y/self.alpha)/self.lam + self.tm


class StochasticArrivalFlow:
    """Implements get_inflow method for a Lane where inflow is generated by stochastic arrival times."""

    def __init__(self, dist, dt, start=0):
        """
        Args:
            dist: calling dist() should generate an arrivals time. Can be a class, e.g. M3Arrivals.
            dt: timestep
            start: timeind of the first inflow
        """
        self.dist = dist  # distribution
        self.dt = dt
        assert start >= 0
        self.start = start

        self.arrival_time = dist(start)
        assert self.arrival_time > self.dt
        self.flow = 1/self.arrival_time
        self.next_time = start + self.arrival_time / self.dt
        self.next_timeind = int(self.next_time) + 1

    def __call__(self, timeind):
        if timeind == self.start:  # automatically reset
            self.arrival_time = self.dist(timeind)
            self.flow = 1/self.arrival_time
            self.next_time = self.start + self.arrival_time / self.dt
            self.next_timeind = int(self.next_time) + 1
        if timeind == self.next_timeind:
            self.arrival_time = self.dist(timeind)
            self.next_time += self.arrival_time/self.dt
            assert self.next_time > timeind
            new_flow = 1/self.arrival_time
            overlap_time = timeind - (self.next_time - self.arrival_time/self.dt)
            cur_flow = self.flow*(1 - overlap_time) + new_flow*overlap_time
            self.next_timeind = int(self.next_time) + 1
            self.flow = new_flow
            return cur_flow
        else:
            return self.flow


def estimate_speed_for_zero_acc(cf_model, cf_params, hd, lspd, spd, acc, maxspeed, tol):
    """Given acceleration acc = cf_model(cf_params, [hd, spd, lspd]), estimate a speed which gives acc ~= 0."""
    if acc > 0:
        while acc > 0:
            old_spd, old_acc = spd, acc
            spd = min(spd + tol, maxspeed)
            acc = cf_model(cf_params, [hd, spd, lspd])
            if spd == maxspeed:
                break
    else:
        while acc <= 0:
            old_spd, old_acc = spd, acc
            spd = max(spd - tol, 0)
            acc = cf_model(cf_params, [hd, spd, lspd])
            if spd == 0:
                break

    return (old_spd - spd) / (acc - old_acc + 1e-6) * old_acc + old_spd


def eql_speed_headway(curlane, inflow, timeind, v_low=1, a_min=-0.35, min_speed=1.5, tol=2.5, **kwargs):
    """Recommended upstream boundary condition. Attempts to add vehicles with close to zero acceleration.

    Suitable for both congested or free conditions. Also good for stochastic boundary conditions or simulations with
    considerable heterogeneity.
    Requires Vehicle to have implemented get_eql and cf_model methods, and to have a defined maxspeed attribute.

    Args:
        curlane: Lane object with upstream boundary condition, to possibly add vehicles to
        inflow: float current flow at boundary (veh/sec)
        timeind: int time index
        v_low: float, speed in m/s, when vehicles are added closely after each other, the following vehicle
            must have at least a_min acceleration if it's speed is the leader's speed minus v_low. Larger values
            more aggressively adds vehicles but may cause the boundary to become unrealistic.
        a_min: float, acceleration in m/s/s. See v_low. More negative values will more aggressively add vehicles.
        min_speed: float, minimum speed (m/s) to use for checking a_min
        tol: float, tolerance for estimating the speed which gives zero acceleration
    Returns:
        pos: position to add the new vehicle. If no vehicle is to be added, return None instead.
        spd: speed to give newly added vehicle
        hd: headway between new vehicle and its leader
    """
    lead = curlane.anchor.lead
    newveh = curlane.newveh
    hd, lspd = get_headway(curlane.anchor, lead), lead.speed
    eql_hd = newveh.get_eql(lspd, input_type='v')
    cf_model, cf_params = newveh.cf_model, newveh.cf_parameters

    if eql_hd >= hd:
        if hd < 0:
            return None
        spd = max(lspd-v_low, min_speed)
        acc = cf_model(cf_params, [hd, spd, lspd])
        if acc > a_min:
            spd = estimate_speed_for_zero_acc(cf_model, cf_params, hd, lspd, spd, acc, newveh.maxspeed, tol)
            return curlane.start, spd, hd
        else:
            return None
    else:
        acc = cf_model(cf_params, [hd, lspd, lspd])
        spd = estimate_speed_for_zero_acc(cf_model, cf_params, hd, lspd, lspd, acc, newveh.maxspeed, tol)
        return curlane.start, spd, hd


def eql_inflow_congested(curlane, inflow, timeind, c=.8, check_gap=True, **kwargs):
    """Condition when adding vehicles for use in congested conditions. Requires to invert flow (needs get_eql method).

    Suggested by Treiber, Kesting in their traffic flow book for congested conditions. Requires to invert
    the inflow to obtain the equilibrium (speed, headway) for the flow. The actual headway on the road must
    be at least c times the equilibrium headway for the vehicle to be added, where c is a constant.
    The speed the vehicle is added with corresponds to the equilibrium speed at that flow.

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


def eql_inflow_free(curlane, inflow, timeind, **kwargs):
    """Suggested by Treiber, Kesting for free conditions. Requires to invert the inflow to obtain velocity."""
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    # get speed corresponding to current flow
    spd = curlane.newveh.inv_flow(inflow, leadlen=lead.len, output_type='v', congested=False)
    return curlane.start, spd, hd


def eql_speed(curlane, inflow, timeind, c=.8, minspeed=2, use_eql=True, **kwargs):
    """Add a vehicle with a speed determined from the vehicle's equilibrium solution.

    This is similar to the eql_inflow but uses microscopic quantities instead of flow. First, calculate the
    speed to add the new vehicle. We use the maximum of the lead vehicle speed, and the equilibrium speed
    based on the current gap. We then calculate the equilibrium headway based on the speed of the new vehicle;
    for the vehicle to be added, the gap must be at least c times the equilibrium headway.

    This method allows the inflow speed to transition to a congested state after initially
    being in free flow. The eql_speed2 method can be used to specify what the entry speeds of vehicle should
    be after the transition to a congested state. In that version, you can specify a transition speed;
    if below the transition speed, we treat c=1 - this will make it so vehicles will enter with a speed
    roughly equal to transition.

    Args:
        curlane: Lane with upstream boundary condition
        inflow:
        timeind:
        c: Constant. If less than 1, then vehicles can be added wtih deceleration in congested conditions.
            If greater than or equal to one, vehicles must have 0 or positive acceleration when being added.
        minspeed: if eql_speed is True, Vehicles must have at least minspeed when being added
        use_eql: If False, vehicles are added with the speed of their leader. If True, we take the max
            of the lead.speed and equilibrium speed corresponding to the lead gap.
    """
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = lead.speed

    # use both lead.speed and equilibrium speed corresponding to the gap to lead
    if use_eql:
        eqlspd = curlane.newveh.get_eql(hd, input_type='s')
        spd = max(minspeed, max(spd, eqlspd))  # Try to add with largest speed possible

    se = curlane.newveh.get_eql(spd, input_type='v')
    if hd > c*se:
        return curlane.start, spd, hd
    else:
        return None


def eql_speed2(curlane, inflow, timeind, c=.8, minspeed=2, use_eql=True, transition=20, **kwargs):
    """Allow transition back to uncongested state corresponding to inflow with speed transition."""
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = lead.speed

    # use both lead.speed and equilibrium speed corresponding to the gap to lead
    if use_eql:
        eqlspd = curlane.newveh.get_eql(hd, input_type='s')
        spd = max(minspeed, max(spd, eqlspd))  # Try to add with largest speed possible

    se = curlane.newveh.get_eql(spd, input_type='v')
    if spd > transition:
        if hd > c*se:
            return curlane.start, spd, hd
        else:
            return None
    elif hd >= se:
        return curlane.start, spd, hd
    else:
        return None


def shifted_speed_inflow(curlane, inflow, timeind, shift=1, accel_bound=-.5, **kwargs):
    """Extra condition for upstream boundary based on Newell model and a vehicle's car following model.

    We get the first speed for the vehicle based on the shifted speed of the lead vehicle (similar to Newell
    model). Then we compute the vehicle's acceleration using its own car following model. If the acceleration
    is too negative, we don't add it to the simulation. If we add it, it's with the shifted leader speed.

    Args:
        curlane: Lane with upstream boundary condition, which will possibly have a vehicle added.
        shift: amount (time) to shift the leader's speed by
        accel_bound: minimum acceleration a vehicle can be added with

    Returns:
        If The vehicle is not to be added, we return None. Otherwise, we return the (pos, spd, hd) for the
        vehicle to be added with.
    """
    # might make more sense to use the lead.lead if leader trajectory is not long enough.
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = shift_speed(lead.speedmem, shift, curlane.dt)

    if accel_bound is not None:
        newveh = curlane.newveh
        acc = newveh.cf_model(newveh.cf_parameters, [hd, spd, lead.speed])
        if acc > accel_bound and hd > 0:
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


def newell_inflow(curlane, inflow, timeind, p=None, accel_bound=-2, **kwargs):
    """Extra condition for upstream boundary based on DE form of Newell model.

    This is like shifted_speed_inflow, but since we use the DE form of the Newell model, there is a maximum
    speed, and there won't be a problem if the shift amount is greater than the length of the lead vehicle
    trajectory (in which case shifted_speed defaults to the first speed).

    Args:
        curlane: Lane with upstream boundary condition
        inflow: unused
        timeind: unused
        p: parameters for Newell model, p[0] = time delay. p[1] = jam spacing
        accel_bound: vehicle must have accel greater than this to be added

    Returns: None if no vehicle is to be added, otherwise a (pos, speed, headway) tuple for IC of new vehicle.
    """
    p = p if p else [1, 2]
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    newveh = curlane.newveh
    spd = max(min((hd - p[1])/p[0], newveh.maxspeed), 0)

    if accel_bound is not None:
        acc = newveh.cf_model(newveh.cf_parameters, [hd, spd, lead.speed])
        if acc > accel_bound and hd > 0:
            return curlane.start, spd, hd
        else:
            return None

    return curlane.start, spd, hd


def speed_inflow(curlane, inflow, timeind, speed_series=None, accel_bound=-2, **kwargs):
    """Like shifted_speed_inflow, but gets speed from speed_series instead of the shifted leader speed."""
    lead = curlane.anchor.lead
    hd = get_headway(curlane.anchor, lead)
    spd = speed_series(timeind)

    if accel_bound is not None:
        newveh = curlane.newveh
        acc = newveh.cf_model(newveh.cf_parameters, [hd, spd, lead.speed])
        if acc > accel_bound and hd > 0:
            return curlane.start, spd, hd
        else:
            return None
    return curlane.start, spd, hd


def increment_inflow_wrapper(boundary_type='heql', kwargs=None):
    """Defines increment_inflow method for Lane.

    The increment_inflow method implements the upstream boundary condition, which determines when new vehicles should
    be added to the simulation. It also is responsible for actually initializing new Vehicles and adding them to
    the simulation with correct leader/follower relationships.

    To implement the upstream boundary, the method Lane.get_inflow is called every timestep, which
    returns the flow amount that timestep, which updates the attribute inflow_buffer. When inflow_buffer >= 1,
    it attempts to add a vehicle to the simulation. There are extra conditions required to add a vehicle,
    which are controlled by the 'boundary_type' keyword arg. Note that the boundary condition defines not only
    when it is possible to add vehicles, but also determines the speed that the vehicle should be added with.

    Args:
        boundary_type: str, defines the type of upstream boundary condition to use. It is recommended
            to use 'heql' (see eql_speed_headway), 'seql' (see eql_speed) or 'seql2'.
            Refer to the relevant function for more information on the boundary condition and keyword arguments.
        kwargs: dictionary of keyword arguments for the boundary_type chosen.
    Returns:
        increment_inflow method -
        Args:
            vehicles: set of vehicles
            vehid: vehicle ID to be used for next created vehicle
            timeind: time index
        Returns:
            vehid: id of the next vehicle to instantiate
    """
    kwargs = {} if kwargs is None else kwargs
    match boundary_type:
        case 'heql':
            boundary_method = eql_speed_headway
        case 'seql':
            boundary_method = eql_speed
        case 'seql2':
            boundary_method = eql_speed2
        case 'ceql':
            boundary_method = eql_inflow_congested
        case 'feql':
            boundary_method = eql_inflow_free
        case 'shifted':
            boundary_method = shifted_speed_inflow
        case 'speed':
            boundary_method = speed_inflow
        case 'newell':
            boundary_method = newell_inflow
        case _:
            raise RuntimeError('invalid boundary_type. Should be one of \'heql\', \'seql\', or \'seql2\'. ' +
                               'Received ' + str(boundary_type))

    def increment_inflow(self, vehicles, vehid, timeind):
        inflow, spd = self.get_inflow(timeind)
        self.inflow_buffer += inflow * self.dt

        if self.inflow_buffer >= 1:

            if self.anchor.lead is None:  # rule for adding vehicles when road is empty
                spd = self.newveh.maxspeed*.9 if spd is None else spd
                out = (self.start, spd, None)
            else:  # normal rule for adding vehicles
                out = boundary_method(self, inflow, timeind, **kwargs)

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
                new_lfol = leftanchor if (leftanchor.lead is None or
                                          leftanchor.lead.pos > newveh.pos) else leftanchor.lead
                newveh.lfol = new_lfol
                new_lfol.rlead.append(newveh)
            else:
                newveh.lfol = None
            rlane = self.get_connect_right(pos)
            if rlane is not None:
                rightanchor = rlane.anchor
                new_rfol = rightanchor if (rightanchor.lead is None or
                                           rightanchor.lead.pos > newveh.pos) else rightanchor.lead
                newveh.rfol = new_rfol
                new_rfol.llead.append(newveh)
            else:
                newveh.rfol = None

            # update simulation
            self.inflow_buffer += -1
            vehicles.add(newveh)

            # create next vehicle
            self.new_vehicle(vehid, timeind)
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
        cf_parameters: None, used to identify a vehicle as being an anchor
        lane, road, lfol, rfol, lead, rlead, llead, all have the same meaning as for Vehicle
        pos: position anchor is on, used for headway/dist calculations
        speed: speed of anchor (acts as placeholder)
        acc: acceleration of anchor (acts as placeholder)
        hd: always None
        len: length of anchor, should be 0
        leadmem: same format as Vehicle
        vehid: str of roadname + land index + -anchor
    """

    def __init__(self, curlane, start, lead=None, rlead=None, llead=None):
        self.cf_parameters = None
        self.lane = curlane
        self.road = curlane.road
        self.vehid = str(self.road)+'-'+str(self.lane.laneind)+'-anchor'

        self.init_lead = lead
        self.init_rlead = rlead
        self.init_llead = llead
        self.start = start

        self.lfol = None  # anchor vehicles just need the lead/llead/rlead attributes. no need for (l/r)fol
        self.rfol = None
        self.fol = None
        self.lead = None if self.init_lead is None else copy.deepcopy(self.init_lead)
        self.rlead = [] if self.init_rlead is None else copy.deepcopy(self.init_rlead)
        self.llead = [] if self.init_llead is None else copy.deepcopy(self.init_llead)
        self.leadmem = [[self.lead, self.start]]

        self.pos = curlane.start
        self.speed = 1e-6
        self.acc = 0
        self.hd = None
        self.len = 0

        self.is_coop = 0  # for cooperation model

    def get_cf(self, lead, timeind):
        """Dummy method - so we don't have to check for anchors when calling set_lc."""
        if lead is not None:
            return get_headway(self, lead), 0
        else:
            return None, 0

    def set_relax(self, *args):
        """Dummy method does nothing - it's so we don't have to check for anchors when applying relax."""
        pass

    def reset(self):
        self.lfol = None
        self.rfol = None
        self.lead = None if self.init_lead is None else copy.deepcopy(self.init_lead)
        self.rlead = [] if self.init_rlead is None else copy.deepcopy(self.init_rlead)
        self.llead = [] if self.init_llead is None else copy.deepcopy(self.init_llead)
        self.leadmem = [[self.lead, self.start]]

    def __repr__(self):
        """Representation in ipython console."""
        return 'anchor for lane '+str(self.lane)

    def __str__(self):
        """Convert to string."""
        return self.__repr__()

    def __eq__(self, other):
        return self.vehid == other.vehid

    def __ne__(self, other):
        return not self.vehid == other.vehid

    def __hash__(self):
        return hash(self.vehid)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['lead'] = veh_to_id(state['lead'])
        rlead, llead, leadmem = [], [], []
        for veh in state['rlead']:
            rlead.append(veh_to_id(veh))
        for veh in state['llead']:
            llead.append(veh_to_id(veh))
        state['leadmem'], state['rlead'], state['llead'] = leadmem, rlead, llead
        return state


def get_headway(veh, lead):
    """Calculates distance from Vehicle veh to the back of Vehicle lead."""
    hd = lead.pos - veh.pos - lead.len
    if veh.road != lead.road:
        hd += veh.lane.roadlen[lead.road]
    return hd


def get_dist(veh, lead):
    """Calculates distance from veh to the front of lead."""
    dist = lead.pos-veh.pos
    if veh.road != lead.road:
        dist += veh.lane.roadlen[lead.road]
    return dist


def veh_to_id(veh):
    if hasattr(veh, 'vehid'):
        return veh.vehid
    else:
        return veh


class Lane:
    """Represents a continuous single lane of a road. Roads are made up of several lanes.

    Lanes are not usually instantiated by the user. Rather, typically one would represent their traffic network
    as several Roads, and the Roads will create the applicable Lanes.

    Lanes are responsible for defining the topology (e.g. when lanes start/end, which lanes connect
    to what, what it is possible to change left/right into), which are handled by the events attribute.
    Lanes are the only object with a reference to roads, which are used for making/defining routes.
    They also are responsible for defining distances between vehicles, as positions are relative to the road
    a lane belongs to. This is handled by the roadlen attribute.

    They also define boundary conditions, and are responsible for creating new vehicles and adding them
    to the network. Boundary conditions are defined by extra methods which must be bound to the Lane. This is
    typically done by calling the set_upstream and set_downstream method of the Road to which the Lane belongs to.
    See also downstream_wrapper (defines call_downstream method), get_inflow_wrapper (defines get_inflow),
    increment_inflow_wrapper (defines increment_inflow), and make_newveh (defines new_vehicle method).
    If defining manually, those functions must also be bound to the Lane using __get__

    Attributes:
        start: starting position of lane
        end: ending position of lane
        road: the Road that Lane belongs to. Roads have a unique str name
        roadname: string name of road
        laneind: index of Lane
        connect_left: defines left connections for Lane
        connect_right: defines right connections for Lane
        connect_to: what the end of Lane connects to. Can be None (no connection or exit)
        connections: for making routes. Dict where keys are road names (str) and value is a tuple of:
            pos: for 'continue' change_type, a float which gives the position that the current road
                changes to the desired road.
                for 'merge' type, a tuple of the first position that changing into the desired road
                becomes possible, and the last position where it is still possible to change into that road.
            change_type: if 'continue', this corresponds to the case where the current road turns into
                the next road in the route; the vehicle still needs to make sure it is in the right lane
                (different lanes may transition to different roads)
                if 'merge', this is the situation where the vehicle needs to change lanes onto a different road.
                Thus in the 'merge' case after completing its lane change, the vehicle is on the next desired
                road, in contrast to the continue case where the vehicle actually needs to reach the end of
                the lane in order to transition.
            laneind: if 'continue', a tuple of 2 ints, giving the leftmost and rightmost lanes which will
                continue to the desired lane. if 'merge', the laneind of the lane we need to be on to merge.
            side: for 'merge' type only, gives whether we want to do a left or right change upon reaching
                laneind ('l_lc' or 'r_lc')
            nextroad: desired road
            refer also to hs.update_lane_routes.make_cur_route.
        anchor: AnchorVehicle for lane
        roadlen: defines distance between Lane's road and other roads. Keys are roads, value is float distance
            which is added to the base headway to get the correct headway.
        merge_anchors: any merge anchors for the lane (see hs.update_lane_routes.update_merge_anchors).
            List of list, where each inner list is a pair of [merge_anchor, position].
        events: lane events (see hs.update_lane_routes.update_lane_events)
        dt: timestep length
        newveh: instance of next Vehicle to be added (only if increment_inflow and new_vehicle methods are defined)
        inflow_buffer: float inflow amount (only if increment_inflow and new_vehicle methods are defined)
    """

    def __init__(self, start, end, road, laneind, connect_left=None, connect_right=None, connect_to=None,
                 roadlen=None, downstream=None, increment_inflow=None, get_inflow=None, new_vehicle=None, dt=None):
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
            roadlen: roadlen dictionary, if available, or None
            downstream: dictionary of keyword args which defines call_downstream method, or None
            increment_inflow: dictionary of keyword args which defines increment_inflow method, or None
            get_inflow: dictionary of keyword args which defines increment_inflow method, or None
            new_vehicle: new_vehicle method
        """
        self.laneind = laneind
        self.road = road
        self.roadname = road.name
        # starting position/end (float)
        self.start = start
        self.end = end
        # connect_left/right has format of list of (pos (float), lane (object)) tuples where lane
        # is the connection starting at pos
        self.connect_left = connect_left if connect_left is not None else [(start, None)]
        self.connect_right = connect_right if connect_right is not None else [(start, None)]
        self.connect_to = connect_to
        self.connections = {}
        self.dt = dt

        self.roadlen = {self.road: 0} if roadlen is None else roadlen
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
            self.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(self, Lane)
        """refer to increment_inflow_wrapper for documentation"""

    def initialize_inflow(self, vehid, timeind):
        """Set inflow to initial state."""
        assert hasattr(self, 'increment_inflow')
        assert hasattr(self, 'get_inflow')
        assert hasattr(self, 'new_vehicle')

        del self.newveh
        self.inflow_buffer = 0
        self.new_vehicle(vehid, timeind)
        return vehid+1

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
            lead Vehicle: lead Vehicle for veh
            following Vehicle: Vehicle which is following veh
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
        if hasattr(self, 'roadname'):
            return hash((self.roadname, self.laneind))
        else:  # needed so pickle can work correctly, since it uses __hash__
            return super().__hash__()

    def __eq__(self, other):
        """Comparison for Lanes using ==."""
        if type(other) != Lane:
            return False
        return self.roadname == other.roadname and self.laneind == other.laneind

    def __ne__(self, other):
        """Comparison for Lanes using !=."""
        return self is not other

    def __repr__(self):
        """Representation in ipython console."""
        return self.roadname+'-'+str(self.laneind)

    def __str__(self):
        """Convert Lane to a string."""
        return self.__repr__()

    def __getstate__(self):
        """Save as serializable object for pickle."""
        my_dict = self.__dict__.copy()
        my_dict.pop('increment_inflow', None)  # remove all bound methods to make lane serializable
        my_dict.pop('new_vehicle', None)
        my_dict.pop('get_inflow', None)
        my_dict.pop('call_downstream', None)
        my_dict.pop('newveh', None)
        return my_dict


def connect_helper(connect, pos):
    """Helper function takes in connect_left/right attribute, position, and returns the correct connection."""
    out = connect[-1][1]  # default to last lane for edge case or case when there is only one possible
    # connection
    for i in range(len(connect)-1):
        if pos < connect[i+1][0]:
            out = connect[i][1]
            break
    return out


def select_route(routes, od, interval):
    if len(routes) > 1:
        p = np.cumsum(od, axis=1)
        rng = np.random.default_rng()

        def make_route(timeind):
            probs = p[timeind // interval]
            rand = rng.random()
            ind = (rand < probs).nonzero()[0][0]
            return routes[ind].copy()
        return make_route

    else:
        def make_route(timeind):
            return routes[0].copy()
        return make_route


def make_newveh(veh_parameters_fn, vehicle, routes, od, interval):
    """Defines newveh function for Lanes.

    Args:
        veh_parameters_fn: callable that accepts time index, returns dictionary of kwargs
        vehicle: subclass of havsim.simulation.Vehicle to instantiate
        routes: list of routes, where each route is a list of str road names
        od: np.array of shape(n_time, n_routes), where the (i, j) index gives the probabilities of each route for time
            index i
        interval: int number of timesteps per interval of od. The time interval is (timeind // interval)
    Returns:
        newveh: function to bind to Lane as new_vehicle method
    """
    route_picker = select_route(routes, od, interval)

    def newveh(self, vehid, timeind):
        route = route_picker(timeind)
        kwargs = veh_parameters_fn(timeind)
        self.newveh = vehicle(vehid, self, route=route, **kwargs)

    return newveh


def compute_route(start_road, start_pos, exit):
    """
    start_road: object for the start road
    start_pos: position on the start road
    exit: name for the exit road (string)
    """
    # The exit must be of str type
    assert isinstance(exit, str)
    # A map from (in_road, out_road) to (shortest_dist, pos, last_in_road_out_road_pair_in_the_shortest_path)
    # (in_road, out_road) uniquely identify a connection point between two roads
    dist = dict()
    dist[(None, start_road)] = (0, start_pos, None)

    # Visited (in_road, out_road) pair
    visited = set()

    def get_road_name(rd):
        if isinstance(rd, str):
            return rd
        return rd.name

    def get_min_distance_road_info():
        ret = None
        min_dist = float('inf')
        for k, v in dist.items():
            if k not in visited and v[0] < min_dist:
                min_dist = v[0]
                ret = k, v
        return ret

    while True:
        cur = get_min_distance_road_info()
        if cur is None:
            break
        (in_road, out_road), (shortest_dist, pos, last_road_info) = cur

        # If the road is an exit
        if isinstance(out_road, str):
            if out_road == exit:
                res = []
                # Construct the shortest path backwards
                while last_road_info is not None:
                    if not res or res[-1] != get_road_name(out_road):
                        res.append(get_road_name(out_road))
                    in_road, out_road = last_road_info
                    last_road_info = dist[last_road_info][2]
                res.reverse()
                # Ignore the start road on the path
                return res
        else:
            for neighbor_name, neighbor_info in out_road.connect_to.items():
                # We only deal with unvisited neighbors
                if neighbor_name not in visited:
                    if neighbor_info[1] == 'continue':
                        connect_pos_neighbor = neighbor_info[0]
                    else:
                        connect_pos_neighbor = neighbor_info[0][0]
                    if connect_pos_neighbor >= pos:
                        new_dist_neighbor = shortest_dist + connect_pos_neighbor - pos
                        neighbor_entry = out_road, (neighbor_name if neighbor_info[-1] is None else neighbor_info[-1])
                        if neighbor_entry not in dist or dist[neighbor_entry][0] > new_dist_neighbor:
                            if neighbor_info[-1] is None:
                                pos_neighbor = 0
                            else:
                                pos_neighbor = connect_pos_neighbor + out_road[0].roadlen[neighbor_name]
                            dist[neighbor_entry] = (new_dist_neighbor, pos_neighbor, (in_road, out_road))
        visited.add((in_road, out_road))
    # If we reach here, it means the exit is not reachable, reach empty route
    return []


def add_lane_events(events, event_to_add):
    for cur_event in events:
        # If we found an event that we can update
        if cur_event['event'] == event_to_add['event'] and abs(cur_event['pos'] - event_to_add['pos']) < 1e-6:
            assert cur_event['event'] == 'update lr'
            if event_to_add['left'] is not None:
                if cur_event['left'] is None or cur_event['left'] == 'remove':
                    cur_event['left'] = event_to_add['left']
                    if 'left anchor' in event_to_add:
                        cur_event['left anchor'] = event_to_add['left anchor']
                else:
                    assert 0, (event_to_add, event_to_add)
            if event_to_add['right'] is not None:
                if cur_event['right'] is None or cur_event['right'] == 'remove':
                    cur_event['right'] = event_to_add['right']
                    if 'right anchor' in event_to_add:
                        cur_event['right anchor'] = event_to_add['right anchor']
                else:
                    assert 0, (event_to_add, event_to_add)
            return
    # If we did not find an event to update, we append the event to the end and sort the list
    events.append(event_to_add)
    events.sort(key=lambda x: x['pos'])


def add_or_get_merge_anchor_index(lane, pos):
    """
    Add a merge anchor at pos if needed, return the index of the merge anchor
    """
    if not hasattr(lane, "merge_anchors"):
        lane.merge_anchors = []
    for ind, e in enumerate(lane.merge_anchors):
        if (e[1] is None and pos == lane.start) or e[1] == pos:
            return ind
    lane.merge_anchors.append([lane.anchor, None if pos == lane.start else pos])
    return len(lane.merge_anchors) - 1


def connect_lane_left_right(left_lane, right_lane, left_connection, right_connection):
    def update_or_insert(connect_obj, pos, lane_obj):
        for i in range(len(connect_obj)):
            if abs(connect_obj[i][0] - pos) < 1e-6:
                if lane_obj is not None:
                    connect_obj[i] = (pos, lane_obj)
                return
        connect_obj.append((pos, lane_obj))
        connect_obj.sort(key=lambda d: d[0])

    if left_lane is None or right_lane is None:
        return

    if left_connection[0] == left_lane.start:
        update_or_insert(left_lane.connect_right, left_lane.start, right_lane)
    else:
        assert left_connection[0] > left_lane.start
        update_or_insert(left_lane.connect_right, left_connection[0], right_lane)
        merge_anchor_ind = add_or_get_merge_anchor_index(right_lane, right_connection[0])
        add_lane_events(left_lane.events,
            {'event': 'update lr', 'right': 'add', 'left': None, 'right anchor': merge_anchor_ind,
             'pos': left_connection[0]})

    if left_connection[1] < left_lane.end:
        update_or_insert(left_lane.connect_right, left_connection[1], None)
        add_lane_events(left_lane.events,
            {'event': 'update lr', 'right': 'remove', 'left': None,
             'pos': left_connection[1]})

    if right_connection[0] == right_lane.start:
        update_or_insert(right_lane.connect_left, right_lane.start, left_lane)
    else:
        assert right_connection[0] > right_lane.start
        update_or_insert(right_lane.connect_left, right_connection[0], left_lane)
        merge_anchor_ind = add_or_get_merge_anchor_index(left_lane, left_connection[0])
        add_lane_events(right_lane.events,
            {'event': 'update lr', 'left': 'add', 'right': None, 'left anchor': merge_anchor_ind,
             'pos': right_connection[0]})

    if right_connection[1] < right_lane.end:
        update_or_insert(right_lane.connect_left, right_connection[1], None)
        add_lane_events(right_lane.events,
            {'event': 'update lr', 'left': 'remove', 'right': None,
             'pos': right_connection[1]})


class Road:
    """Provided for convenience in defining road networks as a collection of Lanes.

    Attributes:
        num_lanes: int number of lanes
        name: str name of road. Should be unique.
        lanes: list of len num_lanes, giving the lane corresponding to the index. Note that the lane can be indexed
            from the Road like Road[lane_index].
    """
    def __init__(self, num_lanes, length, name, connections=None):
        """
        num_lanes: number of lanes (int)
        length: if float/int, the length of all lanes. If a list, each entry gives the (start, end) tuple
          for the corresponding lane
        name: str name of the road. Should be unique.
        connections: If None, we assume the lanes are connected where possible. Otherwise, It is a list where
          each entry is a tuple of ((left start, left end), (right start, right end)) giving the connections
        """
        self.num_lanes = num_lanes
        self.name = name

        # Validate connections arg
        if connections is not None:
            assert len(connections) == num_lanes

        # Validate and canonicalize length arg
        if isinstance(length, int) or isinstance(length, float):
            assert length > 0
            length = [(0, length) for _ in range(num_lanes)]
        else:
            assert isinstance(length, list) and len(length) == num_lanes and isinstance(length[0], tuple)

        # Construct lane objects for the road
        self.lanes = []
        roadlen = {self: 0}
        for i in range(num_lanes):
            lane = Lane(start=length[i][0], end=length[i][1], road=self, laneind=i, roadlen=roadlen)
            self.lanes.append(lane)

        # Connect adjacent lanes
        for i in range(num_lanes - 1):
            left_lane = self.lanes[i]
            right_lane = self.lanes[i + 1]
            if connections is not None:
                left_connection = connections[i][1]
                right_connection = connections[i+1][0]
            else:
                left_connection = right_connection = (
                    max(left_lane.start, right_lane.start), min(left_lane.end, right_lane.end))
            connect_lane_left_right(left_lane, right_lane, left_connection, right_connection)

    def connect(self, new_road, self_indices=None, new_road_indices=None, is_exit=False):
        """Connects Lanes, letting vehicles travel to the next Lane after reaching the end of the first.

        Args:
            new_road: road object to make the connection to. If passing in a string, it's assumed
                to be the name of the exit road (need to mark is_exit as True)
            self_indices: a list of indices of the current road for making the connection
            new_road_indices: a list of indices of the new road to connect to
            is_exit: whether the new road is an exit. An exit road won't have a road object. It is simply a string
                name which specifies that vehicles leave the simulation after reaching that exit road.
        Returns:
            None. Updates the connections/events of specified Lanes.
        """

        if self_indices is None:
            self_indices = list(range(self.num_lanes))
        self_indices.sort()

        # We should check that self_indices is a continuous list of numbers, because the laneind in lane's
        # connections attribute makes this assumption (if 'continue', it's a tuple of 2 ints, giving the
        # leftmost and rightmost lanes which will continue to the desired lane. if 'merge', it's the laneind
        # of the lane we need to be on to merge.)
        def is_continuously_increasing(nums):
            for i in range(len(nums) - 1):
                if nums[i] + 1 != nums[i+1]:
                    return False
            return True
        assert is_continuously_increasing(self_indices)

        # If passing in a string, new_road is assumed to be the name of the exit
        if isinstance(new_road, str):
            assert is_exit
            all_lanes_end = tuple([self.lanes[i].end for i in self_indices])
            # It's assumed that all exits have the same end position
            assert all_lanes_end and min(all_lanes_end) == max(all_lanes_end)
            # We don't need to update roadlen for exit type roads
            for i in self_indices:
                add_lane_events(self.lanes[i].events, {'event': 'exit', 'pos': self.lanes[i].end})
            for i in range(self.num_lanes):
                cur_lane = self.lanes[i]
                cur_lane.connections[new_road] = \
                    (all_lanes_end[0], 'continue', (self_indices[0], self_indices[-1]), None, None)
        else:
            if new_road_indices is None:
                new_road_indices = list(range(new_road.num_lanes))

            new_road_indices.sort()
            # It is assumed that self_indices and new_road_indices have the same length
            assert len(self_indices) == len(new_road_indices)

            all_self_lanes_end = tuple([self.lanes[i].end for i in self_indices])
            all_new_lanes_start = tuple([new_road.lanes[i].start for i in new_road_indices])
            # It's assumed that all lanes should share the same distance measurements
            assert all_self_lanes_end and min(all_self_lanes_end) == max(all_self_lanes_end)
            assert all_new_lanes_start and min(all_new_lanes_start) == max(all_new_lanes_start)

            # update roadlen
            self_roadlen = self.lanes[0].roadlen
            new_roadlen = new_road.lanes[0].roadlen
            new_to_self = - all_new_lanes_start[0] + all_self_lanes_end[0]
            self_to_new = - all_self_lanes_end[0] + all_new_lanes_start[0]
            self_roadlen[new_road] = new_to_self
            new_roadlen[self] = self_to_new
            for cur_new_road in new_roadlen.keys():
                if cur_new_road == new_road or cur_new_road == self:
                    continue
                cur_to_new = new_roadlen[cur_new_road]
                self_roadlen[cur_new_road] = cur_to_new + new_to_self
                cur_new_road.lanes[0].roadlen[self] = - cur_to_new - new_to_self
            for cur_self_road in self_roadlen.keys():
                if cur_self_road == new_road or cur_self_road == self:
                    continue
                cur_to_self = self_roadlen[cur_self_road]
                new_roadlen[cur_self_road] = cur_to_self + self_to_new
                cur_self_road.lanes[0].roadlen[new_road] = - cur_to_self - self_to_new

            # Update connections attribute for all lanes
            new_connection = (all_self_lanes_end[0], 'continue', (min(self_indices), max(self_indices)), None, new_road)
            for i in range(self.num_lanes):
                if new_road.name not in self.lanes[i].connections:
                    self.lanes[i].connections[new_road.name] = new_connection
                else:
                    if self.lanes[i].connections[new_road.name][1] == 'continue':
                        left, right = self.lanes[i].connections[new_road.name][2]
                        old_dist = 0 if i in range(left, right + 1) else min(abs(left - i), abs(right - i))
                    else:
                        left_or_right = self.lanes[i].connections[new_road.name][2]
                        old_dist = abs(i - left_or_right)
                    new_dist = 0 if i in range(self_indices[0], self_indices[-1] + 1) else min(abs(i - self_indices[0]), abs(i - self_indices[-1]))
                    if new_dist < old_dist:
                        self.lanes[i].connections[new_road.name] = new_connection

            # Update lane events, connect_to, merge anchors
            for self_ind, new_road_ind in zip(self_indices, new_road_indices):
                self_lane = self.lanes[self_ind]
                new_lane = new_road[new_road_ind]
                # Update connect_to attribute for the self_lane
                self_lane.connect_to = new_lane

                # Update merge anchors for current track
                new_anchor = self_lane.anchor
                lane_to_update = new_lane
                while lane_to_update is not None:
                    lane_to_update.anchor = new_anchor
                    for merge_anchor in lane_to_update.merge_anchors:
                        merge_anchor[0] = new_anchor
                        if merge_anchor[1] is None:
                            merge_anchor[1] = lane_to_update.start
                    if hasattr(lane_to_update, "connect_to"):
                        lane_to_update = lane_to_update.connect_to
                        # Assert that connect_to won't store an exit lane
                        assert not isinstance(lane_to_update, str)
                    else:
                        lane_to_update = None

                # Create lane events to be added for self_lane
                event_to_add = {'event': 'new lane', 'pos': self_lane.end}

                # Update left side
                self_lane_left = self_lane.get_connect_left(self_lane.end)
                new_lane_left = new_lane.get_connect_left(new_lane.start)
                if self_lane_left == new_lane_left:
                    event_to_add['left'] = None
                elif new_lane_left is None:
                    event_to_add['left'] = 'remove'
                elif self_lane_left is None or not hasattr(self_lane_left, 'connect_to') or self_lane_left.connect_to != new_lane_left:
                    event_to_add['left'] = 'add'
                    merge_anchor_pos = (new_lane.start if new_lane.road is new_lane_left.road
                                        else new_lane.start - new_lane.roadlen[new_lane_left.road])
                    merge_anchor_ind = add_or_get_merge_anchor_index(new_lane_left, merge_anchor_pos)
                    event_to_add['left anchor'] = merge_anchor_ind
                else:
                    event_to_add['left'] = 'update'

                # Update right side
                self_lane_right = self_lane.get_connect_right(self_lane.end)
                new_lane_right = new_lane.get_connect_right(new_lane.start)
                if self_lane_right == new_lane_right:
                    event_to_add['right'] = None
                elif new_lane_right is None:
                    event_to_add['right'] = 'remove'
                elif self_lane_right is None or not hasattr(self_lane_right, 'connect_to') or self_lane_right.connect_to != new_lane_right:
                    event_to_add['right'] = 'add'
                    merge_anchor_pos = (new_lane.start if new_lane.road is new_lane_right.road
                                        else new_lane.start - new_lane.roadlen[new_lane_right.road])
                    merge_anchor_ind = add_or_get_merge_anchor_index(new_lane_right, merge_anchor_pos)
                    event_to_add['right anchor'] = merge_anchor_ind
                else:
                    event_to_add['right'] = 'update'

                add_lane_events(self_lane.events, event_to_add)

    def merge(self, new_road, self_index, new_lane_index, self_pos, new_lane_pos, side=None):
        """Adds a left/right connection between two Lanes (allowing vehicles to change between the lanes).

        Args:
            new_road: new road to be merged into
            self_index: index of the self road's lane
            new_lane_index: index of the new road's lane
            self_pos: a tuple indicating the start/end position of the merge connection for the current lane
            new_lane_pos: a tuple indicating the start/end position of the merge connection for the new lane
            side: 'l' or 'r', if not specified a side, will infer it automatically
        Returns:
            None. Updates the events/connections of specified Lanes of the self and new_road.
        """
        if side == 'l':
            change_side = 'l_lc'
        elif side == 'r':
            change_side = 'r_lc'
        else:
            assert side is None
            change_side = 'l_lc' if self_index == 0 else 'r_lc'
        # Update connections attribute for all lanes
        new_connection = (self_pos, 'merge', self_index, change_side, new_road)
        for i in range(self.num_lanes):
            if new_road.name not in self.lanes[i].connections:
                self.lanes[i].connections[new_road.name] = new_connection
            else:
                if self.lanes[i].connections[new_road.name][1] == 'continue':
                    left, right = self.lanes[i].connections[new_road.name][2]
                    old_dist = 0 if i in range(left, right + 1) else min(abs(left - i), abs(right - i))
                else:
                    left_or_right = self.lanes[i].connections[new_road.name][2]
                    old_dist = abs(i - left_or_right)
                new_dist = abs(i - self_index)
                if new_dist < old_dist:
                    self.lanes[i].connections[new_road.name] = new_connection

        assert isinstance(self_pos, tuple) and isinstance(new_lane_pos, tuple)

        # Update roadlen
        self_roadlen = self.lanes[0].roadlen
        new_roadlen = new_road.lanes[0].roadlen
        new_to_self = - new_lane_pos[0] + self_pos[0]
        self_to_new = - self_pos[0] + new_lane_pos[0]
        self_roadlen[new_road] = new_to_self
        new_roadlen[self] = self_to_new
        for cur_new_road in new_roadlen.keys():
            if cur_new_road == new_road or cur_new_road == self:
                continue
            cur_to_new = new_roadlen[cur_new_road]
            self_roadlen[cur_new_road] = cur_to_new + new_to_self
            cur_new_road.lanes[0].roadlen[self] = - cur_to_new - new_to_self
        for cur_self_road in self_roadlen.keys():
            if cur_self_road == new_road or cur_self_road == self:
                continue
            cur_to_self = self_roadlen[cur_self_road]
            new_roadlen[cur_self_road] = cur_to_self + self_to_new
            cur_self_road.lanes[0].roadlen[new_road] = - cur_to_self - self_to_new

        # Update lane events and connect_left/connect_right for both lanes
        if change_side == 'l_lc':
            connect_lane_left_right(new_road[new_lane_index], self.lanes[self_index], new_lane_pos, self_pos)
        else:
            connect_lane_left_right(self.lanes[self_index], new_road[new_lane_index], self_pos, new_lane_pos)

        # We might need to update the new lane event
        if hasattr(self.lanes[self_index], "connect_to") and abs(self_pos[-1] - self.lanes[self_index].end) < 1e-6:
            for event in self.lanes[self_index].events:
                if event['event'] == 'new lane':
                    side_k = 'left' if change_side == 'l_lc' else 'right'
                    if event[side_k] is None:
                        event[side_k] = 'remove'

    def set_downstream(self, downstream, self_indices=None):
        """
        Args:
            downstream: dictionary of keyword args which defines call_downstream method
                (see road_networks.downstream_wrapper)
            self_indices: a list of lane indices to set downstream condition to
        Returns:
            None. (binds call_downstream method for all Lanes in self)
        """
        if downstream is not None:
            if self_indices is None:
                self_indices = list(range(self.num_lanes))
            else:
                assert isinstance(self_indices, list)
            for ind in self_indices:
                # Setup downstream conditions on all lanes on the same track
                cur_lane = self.lanes[ind].anchor.lane
                while True:
                    cur_lane.call_downstream = downstream_wrapper(**downstream).__get__(cur_lane, Lane)
                    if hasattr(cur_lane, 'connect_to') and isinstance(cur_lane.connect_to, Lane):
                        cur_lane = cur_lane.connect_to
                    else:
                        break

    def set_upstream(self, increment_inflow=None, get_inflow=None, new_vehicle=None, self_indices=None):
        """
        Args:
            increment_inflow: dictionary of keyword args which defines increment_inflow method, or None
                (see road_networks.increment_inflow_wrapper)
            get_inflow: dictionary of keyword args which defines get_inflow method, or None
                (see road_networks.get_inflow_wrapper)
            new_vehicle: new_vehicle method, or None. The new_vehicle is a function with the following signature -
                Args:
                    self: (this will be the instance of Lane after new_vehicle is bound to the Lane)
                    vehid: vehicle id
                    timeind: time index the vehicle is initiated for
                Returns:
                    None
                It should instantiate a new Vehicle with id vehid and set it to the newveh attribute of self.
                I.e. self.newveh = Vehicle(vehid, self, ...)
                See also road_networks.make_newveh
            self_indices: a list of lane indices to set upstream condition to
        Returns:
            None. (binds new_vehicle, get_inflow, increment_inflow methods for all Lanes in self)
        """
        # TODO should have a better way to define and set these, especially the new_vehicle method
        if self_indices is None:
            self_indices = list(range(self.num_lanes))
        else:
            assert isinstance(self_indices, list)
        for ind in self_indices:
            lane = self.lanes[ind]
            if get_inflow is not None:
                lane.get_inflow = get_inflow_wrapper(**get_inflow).__get__(lane, Lane)
            if new_vehicle is not None:
                lane.new_vehicle = new_vehicle.__get__(lane, Lane)
            if increment_inflow is not None:
                lane.inflow_buffer = 0
                lane.newveh = None

                lane.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(lane, Lane)

    def __getitem__(self, index):
        assert isinstance(index, int) and 0 <= index < self.num_lanes
        return self.lanes[index]

    def __eq__(self, other):
        """Comparison for Roads using ==."""
        if type(other) != Road:
            return False
        return self.name == other.name

    def __hash__(self):
        """Hash Road based on road name and number of lanes."""
        if hasattr(self, 'name'):
            return hash(self.name)
        else:
            return super().__hash__()

    def __repr__(self):
        return self.name

    def __getstate__(self):
        my_dict = self.__dict__.copy()
        return my_dict
