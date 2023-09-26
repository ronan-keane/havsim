
"""
Base Vehicle class and its helper functions.
"""
import scipy.optimize as sc
import numpy as np
import math

from havsim.simulation.road_networks import get_dist, get_headway
from havsim.simulation.relaxation import new_relaxation
from havsim.simulation import models
from havsim.simulation import update_lane_routes

def get_eql_helper(veh, x, input_type='v', eql_type='v', spdbounds=(0, 1e4), hdbounds=(0, 1e4), tol=.1):
    """Solves for the equilibrium solution of vehicle veh given input x.

    To use this method, the Vehicle must have an eqlfun method defined. The eqlfun can typically be defined
    analyticallly for one input type.
    The equilibrium (eql) solution is defined as a pair of (headway, speed) such that the car following model
    will give 0 acceleration. For any possible speed, (0 through maxspeed) there is a unique headway which
    defines the equilibrium solution.

    Args:
        veh: Vehicle to obtain equilibrium solution for
        x: float of either speed or headway
        input_type: if input_type is 'v' (v for velocity), then x is a speed. Otherwise x is a headway.
            If x is a speed, we return a headway. Otherwise we return a speed.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed. If input_type != eql_type, we numerically invert the
            eqlfun.
        spdbounds: Bounds on speed. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        hdbounds: Bounds on headway. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        tol: tolerance for solver.

    Raises:
        RuntimeError: If the solver cannot invert the equilibrium function, we return an error. If this
            happens, it's very likely because your equilibrium function is wrong, or because your bounds
            are wrong.

    Returns:
        float value of either headway or speed which together with input x defines the equilibrium solution.
    """
    if input_type == 'v':
        if x < spdbounds[0]:
            x = spdbounds[0]
        elif x > spdbounds[1]:
            x = spdbounds[1]
    elif input_type == 's':
        if x < hdbounds[0]:
            x = hdbounds[0]
        elif x > hdbounds[1]:
            x = hdbounds[1]
    if input_type == eql_type:
        return veh.eqlfun(veh.cf_parameters, x)
    elif input_type != eql_type:
        def inveql(y):
            return x - veh.eqlfun(veh.cf_parameters, y)
        if eql_type == 'v':
            bracket = spdbounds
        else:
            bracket = hdbounds
        ans = sc.root_scalar(inveql, bracket=bracket, xtol=tol, method='brentq')
        if ans.converged:
            return ans.root
        else:
            raise RuntimeError('could not invert provided equilibrium function')


def inv_flow_helper(veh, x, leadlen=None, output_type='v', congested=True, eql_type='v',
                    spdbounds=(0, 1e4), hdbounds=(0, 1e4), tol=.1, ftol=.01):
    """Solves for the equilibrium solution corresponding to a given flow x.

    To use this method, a vehicle must have an eqlfun defined. A equilibrium solution can be converted to
    a flow. This function takes a flow and finds the corresponding equilibrium solution. To do this,
    it first finds the maximum flow possible, and then based on whether the flow corresponds to the
    congested or free flow regime, we solve for the correct equilibrium.

    Args:
        veh: Vehicle to invert flow for.
        x: flow (float) to invert
        leadlen: When converting an equilibrium solution to a flow, we must use a vehicle length. leadlen
            is that vehicle length. If None, we will infer the vehicle length.
        output_type: if 'v', we want to return a velocity, if 's' we want to return a 'headway'. If 'both',
            we want to return a tuple of the (v,s).
        congested: True if we assume the given flow corresponds to the congested regime, otherwise
            we assume it corresponds to the free flow regime.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed. If input_type != eql_type, we numerically invert the
            eqlfun.
        spdbounds: Bounds on speed. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        hdbounds: Bounds on headway. If x is outside the bounds, we project it onto bounds. Also used
            if eql_type != input_type.
        tol: tolerance for solver.
        ftol: tolerance for solver for finding maximum flow.

    Raises:
        RuntimeError: Raised if either solver fails.

    Returns:
        float velocity if output_type = 'v', float headway if output_type = 's', tuple of (velocity, headway)
        if output_type = 'both'
    """
    if leadlen is None:
        lead = veh.lead
        if lead is not None:
            leadlen = lead.len
        else:
            leadlen = veh.len

    if eql_type == 'v':
        bracket = spdbounds
    else:
        bracket = hdbounds

    def maxfun(y):
        return -veh.get_flow(y, leadlen=leadlen, input_type=eql_type)
    res = sc.minimize_scalar(maxfun, bracket=bracket, options={'xatol': ftol}, method='bounded',
                             bounds=bracket)

    if res['success']:
        if congested:
            invbounds = (bracket[0], res['x'])
        else:
            invbounds = (res['x'], bracket[1])
        if -res['fun'] < x:
            print('warning - inputted flow is too large to be achieved')
            if output_type == eql_type:
                return res['x']
            else:
                return veh.get_eql(res['x'], input_type=eql_type)
    else:
        raise RuntimeError('could not find maximum flow')

    if eql_type == 'v':
        def invfun(y):
            return x - y/(veh.eqlfun(veh.cf_parameters, y) + leadlen)
    elif eql_type == 's':
        def invfun(y):
            return x - veh.eqlfun(veh.cf_parameters, y)/(y+leadlen)

    ans = sc.root_scalar(invfun, bracket=invbounds, xtol=tol, method='brentq')

    if ans.converged:
        if output_type == 'both':
            if eql_type == 'v':
                return (ans.root, ans.root/x - leadlen)
            else:
                return ((ans.root+leadlen)*x, ans.root)
        else:
            if output_type == eql_type:
                return ans.root
            elif output_type == 's':
                return ans.root/x - leadlen
            elif output_type == 'v':
                return (ans.root+leadlen)*x
    else:
        raise RuntimeError('could not invert provided equilibrium function')


def set_lc_helper(veh, l_lc, r_lc, timeind, chk_lc=True, chk_lc_prob=1):
    """Calculates the new headways to be passed to the lane changing (LC) model.

    Evaluates the lane changing situation to decide if we need to evaluate lane changing model on the
    left side, right side, both sides, or neither. For any sides we need to evaluate, finds the new headways
    (new vehicle headway, new lcside follower headway).
    If new headways are negative, returns positive instead.

    Args:
        veh: Vehicle to have their lane changing model called.
        l_lc: state of left lane change (None, 'discretionary' or 'mandatory')
        r_lc: state of right lane change (None, 'discretionary' or 'mandatory')
        timeind: time index
        chk_lc: bool, if True we are in mandatory or active discretionary state. If False we are in normal
            discretionary state and don't necessarily evaluate the LC model every timestep.
        chk_lc_prob: float between 0 and 1 which gives the probability of checking the lane changing model
            when the vehicle is in the normal discretionary state.

    Returns:
        bool: True if we want to call the lane changing model.
        tuple of floats: (newlfolhd, newlhd, newrfolhd, newrhd, newfolhd). float headways,
            giving the new headway for that vehicle. If get_fol = False, newfolhd is not present.
            If a vehicle would have no leader in the new configuration, None is returned as the headway. If
            a AnchorVehicle acts as a (l/r)fol, the headway is computed as normal.
    """
    if not chk_lc:  # decide if we want to evaluate lc model or not - this only applies to discretionary state
        # when vehicle is not actively trying to change
        if timeind < veh.disc_cooldown:
            return False, None
        elif np.random.rand() > chk_lc_prob:
            return False, None

    # next we compute quantities to send to LC model for the required sides
    if l_lc is not None:
        newlfolhd, newlhd = get_new_hd(veh.lfol, veh, veh.llane)
    else:
        newlfolhd = newlhd = None

    if r_lc is not None:
        newrfolhd, newrhd = get_new_hd(veh.rfol, veh, veh.rlane)
    else:
        newrfolhd = newrhd = None

    if veh.lead is None:
        newfolhd = None
    else:
        newfolhd = get_headway(veh.fol, veh.lead)

    return True, (newlfolhd, newlhd, newrfolhd, newrhd, newfolhd)


def get_new_hd(lcsidefol, veh, lcsidelane):
    """Calculates new headways for a vehicle and its left or right follower.

    Args:
        lcsidefol: either the lfol or rfol of veh.
        veh: Vehicle whose lane changing model is being evaluated
        lcsidelane: the lcside lane of veh.

    Returns:
        newlcsidefolhd: new float headway for lcsidefol
        newlcsidehd: new float headway for veh
    """
    # helper function for set_lc_helper
    lcsidelead = lcsidefol.lead
    if lcsidelead is None:
        newlcsidehd = None
    else:
        newlcsidehd = get_headway(veh, lcsidelead)

    newlcsidefolhd = get_headway(lcsidefol, veh)

    return newlcsidefolhd, newlcsidehd


class Vehicle:
    """Base Vehicle class. Implemented for a second order ODE car following model.

    Vehicles are responsible for implementing the rules to update their positions. This includes a
    'car following' (cf) model which is used to update the longitudinal (in the direction of travel) position.
    There is also a 'lane changing' (lc) model which is used to update the latitudinal (which lane) position.
    Besides these two fundamental components, Vehicles also need an update method, which updates their
    longitudinal positions and memory of their past (past memory includes any quantities which are needed
    to differentiate the simulation).
    Vehicles also contain the quantities lead, fol, lfol, rfol, llead, and rlead, (i.e. their vehicle
    relationships) which define the order of vehicles on the road, and is necessary for calling the cf
    and lc models.
    Vehicles also maintain a route, which defines what roads they want to travel on in the road network.
    Besides their actual lc model, Vehicles also handle any additional components of lane changing,
    such as relaxation, cooperation, or tactical lane changing models.
    Lastly, the vehicle class has some methods which may be used for certain boundary conditions.

    Attributes:
        vehid: unique vehicle ID for hashing
        len: length of vehicle (float)
        lane: Lane object vehicle is currently on
        road: str name of the road lane belongs to
        cf_parameters: list of float parameters for the cf model
        lc_parameters: list of float parameters for the lc model
        relax_parameters: float parameter for relaxation; if None, no relaxation
        relax: if there is currently relaxation, a list of floats or list of tuples giving the relaxation
            values.
        in_relax: bool, True if there is currently relaxation
        relax_start: time index corresponding to relax[0]. (int)
        relax_end: The last time index when relaxation is active. (int)
        route_parameters: parameters for the route model (list of floats)
        route: list of road names (str). When the vehicle first enters the simulation or enters a new road,
            the route gets pop().
        routemem: route which was used to init vehicle.
        minacc: minimum allowed acceleration (float)
        maxacc: maxmimum allowed acceleration(float)
        maxspeed: maximum allowed speed (float)
        hdbounds: tuple of minimum and maximum possible headway.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
        shift_parameters: list of float parameters for the tactical/cooperative model. shift_parameters
            control how a vehicle can modify its acceleration in order to facilitate lane changing.
        coop_parameters: float between (0, 1) which gives the base probability of the vehicle
            cooperating with a vehicle wanting to change lanes
        lc_urgency: for mandatory lane changes, lc_urgency is a tuple of floats which control if
            the ego vehicle can force cooperation (simulating aggressive behavior)
        coop_veh: For cooperation, coop_veh is a reference the vehicle giving cooperation. There is no
            attribute (currently) which allows the ego vehicle to see if itself is giving cooperation.
        disc_cooldown: when a vehicle makes a discretionary change, it cannot make another discretionary
            change until after time index disc_cooldown. Initialized as -math.inf
        disc_endtime: when a vehicle enters the active discretionary state, it stays in that state until
            time index disc_endtime
        lead: leading vehicle (Vehicle)
        fol: following vehicle (Vehicle)
        lfol: left follower (Vehicle)
        rfol: right follower (Vehicle)
        llead: list of all vehicles which have the ego vehicle as a right follower
        rlead: list of all vehicles which have the ego vehicle as a left follower
        start: first time index a vehicle is simulated.
        end: the last time index a vehicle is simulated. (or None)
        leadmem: list of tuples, where each tuple is (lead vehicle, time) giving the time the ego vehicle
            first begins to follow the lead vehicle.
        lanemem: list of tuples, where each tuple is (Lane, time) giving the time the ego vehicle
            first enters the Lane.
        posmem: list of floats giving the position, where the 0 index corresponds to the position at start
        speedmem: list of floats giving the speed, where the 0 index corresponds to the speed at start
        relaxmem: list of tuples where each tuple is (relaxation, first_time) where relaxation
            gives the relaxation values and first_time is the starting time index
        pos: position (float)
        speed: speed (float)
        hd: headway (float)
        acc: acceleration (float)
        llane: the Lane to the left of the current lane the vehicle is on, or None
        rlane: the Lane to the right of the current lane the vehicle is on, or None
        l_lc: the current lane changing state for the left side, None, 'discretionary' or 'mandatory'
        r_lc: the current lane changing state for the right side, None, 'discretionary' or 'mandatory'
        lside: If True, we need to evaluate making a left lane change
        rside: If True, we need to evaluate making a right lane change
        in_disc: If True, we are in a discretionary lane changing state
        chk_lc: If True, we are either in a mandatory or active discretionary lane changing state
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle with parameters p needs to take on that lane
        route_events: list of current route events for current lane
        lane_events: list of lane events for current lane
    """
    # TODO implementation of adjoint method for cf, relax, shift parameters
    # TODO set_route_events should be a method of vehicle? use cases -
        # 1. need to customize route model
        # 2. want to get rid of route model (e.g. ring road simulation)

    def __init__(self, vehid, curlane, cf_parameters, lc_parameters, lead=None, fol=None, lfol=None,
                 rfol=None, llead=None, rlead=None, length=3, eql_type='v', relax_parameters=15,
                 shift_parameters=None, coop_parameters=.2, route_parameters=None, route=None, accbounds=None,
                 maxspeed=1e4, hdbounds=None):
        """Inits Vehicle. Cannot be used for simulation until initialize is also called.

        After a Vehicle is created, it is not immediatley added to simulation. This is because different
        upstream (inflow) boundary conditions may require to have access to the vehicle's parameters
        and methods before actually adding the vehicle. Thus, to use a vehicle you need to first call
        initialize, which sets the remaining attributes.

        Args:
            vehid: unique vehicle ID for hashing
            curlane: lane vehicle starts on
            cf_parameters: list of float parameters for the cf model
            lc_parameters: list of float parameters for the lc model
            lead: leading vehicle (Vehicle). Optional, can be set by the boundary condition.
            fol: following vehicle (Vehicle). Optional, can be set by the boundary condition.
            lfol: left follower (Vehicle). Optional, can be set by the boundary condition.
            rfol: right follower (Vehicle). Optional, can be set by the boundary condition.
            llead: list of all vehicles which have the ego vehicle as a right follower. Optional, can be set
                by the boundary condition.
            rlead: list of all vehicles which have the ego vehicle as a left follower. Optional, can be set
                by the boundary condition.
            length: float (optional). length of vehicle.
            eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
            relax_parameters: float parameter for relaxation; if None, no relaxation
            shift_parameters: list of float parameters for the tactical/cooperative model. shift_parameters
                control how a vehicle can modify its acceleration in order to facilitate lane changing.
                By default, gives the deceleration/acceleration amounts which are added to the car following
                acceleration when the tactical/cooperative model is activated.
                can modify the car following acceleration
            coop_parameters: float between (0, 1) which gives the base probability of the vehicle
                cooperating with a vehicle wanting to change lanes
            route: list of road names (str) which defines the route for the vehicle.
            route_parameters: parameters for the route model (list of floats). See make_cur_route for
                explanation of the default route model.
            accbounds: tuple of bounds for acceleration.
            maxspeed: maximum allowed speed.
            hdbounds: tuple of bounds for headway.
        """
        self.vehid = vehid
        self.len = length
        self.lane = curlane
        self.road = curlane.road.name if curlane is not None else None
        # model parameters
        self.cf_parameters = cf_parameters
        self.lc_parameters = lc_parameters

        # relaxation
        self.relax_parameters = relax_parameters
        self.in_relax = False
        self.relax = None
        self.relax_start = None

        # route parameters
        self.route_parameters = [200, 200] if route_parameters is None else route_parameters
        self.route = [] if route is None else route
        # TODO check if route is empty
        self.routemem = self.route.copy()

        # bounds
        if accbounds is None:
            self.minacc, self.maxacc = -12, 5
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed
        self.hdbounds = (0, 1e4) if hdbounds is None else hdbounds
        self.eql_type = eql_type

        # cooperative/tactical model
        self.shift_parameters = [-3, 1.5] if shift_parameters is None else shift_parameters
        self.coop_parameters = coop_parameters
        # any attributes not set by update_lc_state must be set in __init__
        self.lc_urgency = None
        self.coop_veh = None
        self.disc_cooldown = -math.inf
        self.disc_endtime = -math.inf
        self.chk_lc = None

        # leader/follower relationships
        self.lead = lead
        self.fol = fol
        self.lfol = lfol
        self.rfol = rfol
        self.llead = llead
        self.rlead = rlead

        # memory
        self.end = None
        self.leadmem = []
        self.lanemem = []
        self.posmem = []
        self.speedmem = []
        self.relaxmem = []

    def initialize(self, pos, spd, hd, start):
        """Sets the remaining attributes of the vehicle, making it able to be simulated.

        Args:
            pos: position at start
            spd: speed at start
            hd: headway at start
            start: first time index vehicle is simulated

        Returns:
            None.
        """
        # state
        self.pos = pos
        self.speed = spd
        self.hd = hd

        # memory
        self.start = start
        self.leadmem.append((self.lead, start))
        self.lanemem.append((self.lane, start))
        self.posmem.append(pos)
        self.speedmem.append(spd)

        # initialize LC model - set llane/rlane, l/r_lc and initial lc state
        self.llane = self.lane.get_connect_left(pos)
        if self.llane is None:
            self.l_lc = None
        elif self.llane.roadname == self.road:
            self.l_lc = 'discretionary'
        else:
            self.l_lc = None
        self.rlane = self.lane.get_connect_right(pos)
        if self.rlane is None:
            self.r_lc = None
        elif self.rlane.roadname == self.road:
            self.r_lc = 'discretionary'
        else:
            self.r_lc = None
        self.update_lc_state(start)

        # set lane/route events - sets lane_events, route_events, cur_route attributes
        self.cur_route = update_lane_routes.make_cur_route(
            self.route_parameters, self.lane, self.route.pop(0))

        # self.route_events = self.cur_route[self.lane].copy()
        update_lane_routes.set_lane_events(self)
        update_lane_routes.set_route_events(self, start)

    def cf_model(self, p, state):
        """Defines car following model.

        Args:
            p: parameters for model (cf_parameters)
            state: list of headway, speed, leader speed

        Returns:
            float acceleration of the model.
        """
        return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2])) /
                                            (2*(p[3]*p[4])**(1/2)))/(state[0]))**2)

    def get_cf(self, hd, spd, lead, curlane, timeind, dt, userelax):
        """Responsible for the actual call to cf_model / call_downstream.

        Args:
            hd (float): headway
            spd (float): speed
            lead (Vehicle): lead Vehicle
            curlane (Lane): lane self Vehicle is on
            timeind (int): time index
            dt (float): timestep
            userelax (bool): boolean for relaxation

        Returns:
            acc (float): longitudinal acceleration for current timestep
        """
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)
        elif userelax:
            # safeguard for relaxation  # todo add safeguard parameters to vehicle
            ttc = hd - 2 - .6*spd
            ttc = 0 if ttc < 0 else ttc / (spd - lead.speed + 1e-6)
            if 1.5 > ttc >= 0:
                currelax, currelax_v = self.relax[timeind-self.relax_start]
                currelax = currelax*(ttc/1.5)**2 if currelax > 0 else currelax
                currelax_v = currelax_v*(ttc/1.5)**2 if currelax_v > 0 else currelax_v
                acc = self.cf_model(self.cf_parameters, [max(hd + currelax, .1), spd, lead.speed + currelax_v])
            else:
                currelax, currelax_v = self.relax[timeind - self.relax_start]
                acc = self.cf_model(self.cf_parameters, [max(hd + currelax, .1), spd, lead.speed + currelax_v])
        else:
            acc = self.cf_model(self.cf_parameters, [hd, spd, lead.speed])

        return acc

    def set_cf(self, timeind, dt):
        """Sets a vehicle's acceleration by calling get_cf."""
        self.acc = self.get_cf(self.hd, self.speed, self.lead, self.lane, timeind, dt, self.in_relax)

    def set_relax(self, timeind, dt):
        """Applies relaxation - make sure get_cf is set up to correctly use relaxation."""
        new_relaxation(self, timeind, dt, True)

    def free_cf(self, p, spd):
        """Defines car following model in free flow.

        The free flow model can be obtained simply by letting the headway go to infinity for cf_model.

        Args:
            p: parameters for model (cf_parameters)
            spd: speed (float)

        Returns:
            float acceleration corresponding to the car following model in free flow.
        """
        return p[3]*(1-(spd/p[0])**4)

    def eqlfun(self, p, v):
        """Equilibrium function.

        Args:
            p:. car following parameters
            v: velocity/speed

        Returns:
            s: headway such that (v,s) is an equilibrium solution for parameters p
        """
        s = ((p[2]+p[1]*v)**2/(1-(v/p[0])**4))**.5
        return s

    def get_eql(self, x, input_type='v'):
        """Get equilibrium using provided function eqlfun, possibly inverting it if necessary."""
        return get_eql_helper(self, x, input_type, self.eql_type, (0, self.maxspeed), self.hdbounds)

    def get_flow(self, x, leadlen=None, input_type='v'):
        """Input a speed or headway, and output the flow based on the equilibrium solution.

        This will automatically apply bounds to the headway/speed based on the maxspeed and hdbounds
        attributes. Also, note that it is possible to get the maximum possible flow and corresponding
        equilibrium solution - call inv_flow with an unattainable flow and it will return the speed/headway
        corresponding to the maximum possible flow. Then you can call get_flow on the returned equilibrium.

        Args:
            x: Input, either headway or speed
            leadlen: When converting an equilibrium solution to a flow, we must use a vehicle length. leadlen
                is that vehicle length. If None, we will use the vehicle's own length.
            input_type: if input_type is 'v' (v for velocity), then x is a speed. Otherwise x is a headway.
                If x is a speed, we return a headway. Otherwise we return a speed.

        Returns:
            flow (float)

        """
        if leadlen is None:
            lead = self.lead
            if lead is not None:
                leadlen = lead.len
            else:
                leadlen = self.len
        if input_type == 'v':
            s = self.get_eql(x, input_type=input_type)
            return x / (s + leadlen)
        elif input_type == 's':
            v = self.get_eql(x, input_type=input_type)
            return v / (x + leadlen)

    def inv_flow(self, x, leadlen=None, output_type='v', congested=True):
        """Get equilibrium solution corresponding to the provided flow."""
        return inv_flow_helper(self, x, leadlen, output_type, congested, self.eql_type,
                               (0, self.maxspeed), self.hdbounds)

    def shift_eql(self, state):
        """Model used for applying tactical/cooperative acceleration during lane changes.

        It is assumed that we can give one of two commands - either 'decel' or 'accel' to make the vehicle
        give more or less space, respectively.

        Args:
            state: either 'decel' if we want vehicle to increase its headway, otherwise 'accel'

        Returns:
            TYPE: float acceleration
        """
        return models.generic_shift(0, 0, self.shift_parameters, state)

    def set_lc(self, lc_actions, timeind, dt):
        """Evaluates a vehicle's lane changing model, recording the result in lc_actions.

        The result of the lane changing (lc) model can be either 'l' or 'r' for left/right respectively,
        or None, in which case there is no lane change. If the model has tactical/cooperative elements added,
        calling the lc model may cause some vehicles to enter into a tactical or cooperative state, which
        modifies the vehicle's acceleration by using the shift_eql method.

        Args:
            lc_actions: dictionary where keys are Vehicles which changed lanes, values are the side of change
            timeind: time index
            dt: timestep

        Returns:
            None. (Modifies lc_actions, some vehicle attributes, in place)
        """
        call_model, args = set_lc_helper(self, self.l_lc, self.r_lc, timeind, self.chk_lc, self.lc_parameters[6])
        if call_model:
            models.mobil(self, lc_actions, *args, timeind, dt)

    def update_lc_state(self, timeind, lc=None):
        """Updates the lane changing internal state when completing, aborting, or beginning lane changing.

        Cases when this is called -
            -after a route event ends a discretionary state or begins a mandatory state
            -after the network topology changes (e.g. a new left lane, or right lane ends)
            -after a lane changing is completed (pass lc=lc_actions[veh] if veh completes the lane change)

        The complete lane changing interal state consists of the attributes lside, rside, in_disc, chk_lc,
        disc_cooldown, disc_endtime, coop_veh, and lc_urgency. Some of these (disc_endtime, coop_veh) are
        set by set_lc method (i.e. set by mobil, tact_coop_model in simulation.models). lc_urgency is set
        by the route model. Therefore this updates the lside, rside, in_disc, chk_lc, and disc_cooldown.
        (See attributes of Vehicle for explanation on each attribute).

        The attributes llane, rlane, l_lc, and r_lc are also relevant for understanding the lane changing
        model state. However, these attributes are intrinsic properties of the road network and route,
        as opposed to special states of the lane changing model.
        """
        # do not allow multiple discretionary within short time period
        if lc and self.in_disc:
            self.disc_cooldown = timeind+self.lc_parameters[8]

        # activated_disc_side is 'lside' if vehicle is actively trying to make a left discretionary change
        if self.chk_lc and self.in_disc:
            activated_disc_side = 'lside' if self.lside else 'rside'
        else:
            activated_disc_side = False

        # main logic is to determine lside, rside, in_disc attributes.
        # note that this assumes that only one of lside, rside can be mandatory at a time.
        l_lc, r_lc = self.l_lc, self.r_lc
        if l_lc is None:
            if r_lc is None:
                self.lside, self.rside, self.in_disc = False, False, False
            elif r_lc == 'discretionary':
                self.lside, self.rside, self.in_disc = False, True, True
            else:
                self.lside, self.rside, self.in_disc = False, True, False
        elif l_lc == 'discretionary':
            if r_lc is None:
                self.lside, self.rside, self.in_disc = True, False, True
            elif r_lc == 'discretionary':
                self.lside, self.rside, self.in_disc = True, True, True
            else:
                self.lside, self.rside, self.in_disc = False, True, False
        else:
            self.lside, self.rside, self.in_disc = True, False, False

        # chk_lc should be True for mandatory or activated discretionary state, False otherwise
        if activated_disc_side:
            # can't discretionary change anymore -> end activated state
            if not getattr(self, activated_disc_side):
                self.chk_lc = False
            else:
                # if we stay in activated state, only the activated side of lside, rside, can be True
                opside = 'rside' if activated_disc_side == 'lside' else 'lside'
                setattr(self, opside, False)
        else:
            self.chk_lc = not self.in_disc

    def acc_bounds(self, acc):
        """Apply acceleration bounds."""
        if acc > self.maxacc:
            acc = self.maxacc
        elif acc < self.minacc:
            acc = self.minacc
        return acc

    def update(self, timeind, dt):
        """Applies bounds and updates a vehicle's longitudinal state/memory."""
        # bounds on acceleration
        # acc = self.acc_bounds(self.acc)
        acc = max(self.minacc, self.acc)

        # bounds on speed
        temp = acc*dt
        nextspeed = self.speed + temp
        if nextspeed < 0:
            nextspeed = 0
        # elif nextspeed > self.maxspeed:
        #     nextspeed = self.maxspeed

        # update state
        # self.pos += self.speed*dt + .5*temp*dt  # ballistic update
        self.pos += self.speed*dt  # euler update
        self.speed = nextspeed

        # update memory
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_end:
                self.in_relax = False
                self.relaxmem.append((self.relax, self.relax_start))

    def __hash__(self):
        """Vehicles need to be hashable. We hash them with a unique vehicle ID."""
        return hash(self.vehid)

    def __eq__(self, other):
        """Used for comparing two vehicles with ==."""
        return self.vehid == other.vehid

    def __ne__(self, other):
        """Used for comparing two vehicles with !=."""
        return not self.vehid == other.vehid

    def __repr__(self):
        """Display for vehicle in interactive console."""
        if not self.end:
            return 'vehicle '+str(self.vehid)+' on lane '+str(self.lane)+' at position '+str(self.pos)
        else:
            return 'vehicle '+str(self.vehid)

    def __str__(self):
        """Convert vehicle to a str representation."""
        return self.__repr__()

    def _leadfol(self):
        """Summarize the leader/follower relationships of the Vehicle."""
        print('-------leader and follower-------')
        if self.lead is None:
            print('No leader')
        else:
            print('leader is '+str(self.lead))
        print('follower is '+str(self.fol))
        print('-------left and right followers-------')
        if self.lfol is None:
            print('no left follower')
        else:
            print('left follower is '+str(self.lfol))
        if self.rfol is None:
            print('no right follower')
        else:
            print('right follower is '+str(self.rfol))

        print('-------'+str(len(self.llead))+' left leaders-------')
        for i in self.llead:
            print(i)
        print('-------'+str(len(self.rlead))+' right leaders-------')
        for i in self.rlead:
            print(i)
        return

    def _chk_leadfol(self, verbose=True):
        """Returns True if the leader/follower relationships of the Vehicle are correct."""
        # If verbose = True, we print whether each test is passing or not.
        lfolpass = True
        lfolmsg = []
        if self.lfol is not None:
            if self.lfol is self:
                lfolpass = False
                lfolmsg.append('lfol is self')
            if self not in self.lfol.rlead:
                lfolpass = False
                lfolmsg.append('rlead of lfol is missing self')
            if self.lfol.lane.anchor is not self.llane.anchor:
                lfolpass = False
                lfolmsg.append('lfol is not in left lane')
            if get_dist(self, self.lfol) > 0:
                lfolpass = False
                lfolmsg.append('lfol is in front of self')
            if self.lfol.lead is not None:
                if get_dist(self, self.lfol.lead) < 0:
                    lfolpass = False
                    lfolmsg.append('lfol leader is behind self')
            lead, fol = self.lane.leadfol_find(self, self.lfol)
            if fol is not self.lfol:
                if self.lfol.pos == self.pos:
                    pass
                else:
                    lfolpass = False
                    lfolmsg.append('lfol is not correct vehicle - should be '+str(fol))
        elif self.llane is not None:
            unused, fol = self.llane.leadfol_find(self, self.llane.anchor)
            lfolpass = False
            lfolmsg.append('lfol is None - should be '+str(fol))
        rfolpass = True
        rfolmsg = []
        if self.rfol is not None:
            if self.rfol is self:
                rfolpass = False
                rfolmsg.append('rfol is self')
            if self not in self.rfol.llead:
                rfolpass = False
                rfolmsg.append('llead of rfol is missing self')
            if self.rfol.lane.anchor is not self.rlane.anchor:
                rfolpass = False
                rfolmsg.append('rfol is not in right lane')
            if get_dist(self, self.rfol) > 0:
                rfolpass = False
                rfolmsg.append('rfol is in front of self')
            if self.rfol.lead is not None:
                if get_dist(self, self.rfol.lead) < 0:
                    rfolpass = False
                    rfolmsg.append('rfol leader is behind self')
            lead, fol = self.lane.leadfol_find(self, self.rfol)
            if fol is not self.rfol:
                if self.rfol.pos == self.pos:
                    pass
                else:
                    rfolpass = False
                    rfolmsg.append('rfol is not correct vehicle - should be '+str(fol))
        elif self.rlane is not None:
            unused, fol = self.rlane.leadfol_find(self, self.rlane.anchor)
            rfolpass = False
            rfolmsg.append('lfol is None - should be '+str(fol))
        rleadpass = True
        rleadmsg = []
        for i in self.rlead:
            if i.lfol is not self:
                rleadpass = False
                rleadmsg.append('rlead does not have self as lfol')
            if get_dist(self, i) < 0:
                rleadpass = False
                rleadmsg.append('rlead is behind self')
        if len(self.rlead) != len(set(self.rlead)):
            rleadpass = False
            rleadmsg.append('repeated rlead')
        lleadpass = True
        lleadmsg = []
        for i in self.llead:
            if i.rfol is not self:
                lleadpass = False
                lleadmsg.append('llead does not have self as rfol')
            if get_dist(self, i) < 0:
                lleadpass = False
                lleadmsg.append('llead is behind self')
        if len(self.llead) != len(set(self.llead)):
            lleadpass = False
            lleadmsg.append('repeated llead')
        leadpass = True
        leadmsg = []
        if self.lead is not None:
            if self.lead.fol is not self:
                leadpass = False
                leadmsg.append('leader does not have self as follower')
            if get_headway(self, self.lead) < 0:
                leadpass = False
                leadmsg.append('leader is behind self')

        folpass = True
        folmsg = []
        if self.fol.lead is not self:
            folpass = False
            folmsg.append('follower does not have self as leader')
        if get_headway(self, self.fol) > 0:
            folpass = False
            folmsg.append('follower is ahead of self')

        res = lfolpass and rfolpass and rleadpass and lleadpass and leadpass and folpass
        if verbose:
            if res:
                print('passing results for '+str(self))
            else:
                print('errors for '+str(self))
            if not lfolpass:
                for i in lfolmsg:
                    print(i)
            if not rfolpass:
                for i in rfolmsg:
                    print(i)
            if not rleadpass:
                for i in rleadmsg:
                    print(i)
            if not lleadpass:
                for i in lleadmsg:
                    print(i)
            if not leadpass:
                for i in leadmsg:
                    print(i)
            if not folpass:
                for i in folmsg:
                    print(i)

        return res


def add_crash_behavior(vehicle, decel=-4., hold_timesteps=20):
    """Defines what a vehicle should do after crashing."""
    # decel: after crash, decelerate at this rate
    # hold_timesteps: remove the vehicle after hold_timesteps at 0 speed
    assert issubclass(vehicle, Vehicle)

    class VehicleCanCrash(vehicle):
        def __init__(self, *args, **kwargs):
            self.crashed = False
            self.crash_time = None
            self.crash_decel = decel
            self.hold_timesteps = hold_timesteps
            super().__init__(*args, **kwargs)

        def update_after_crash(self, timeind):
            """Called after a crash to enforce the crash behavior."""
            self.crashed = True
            self.crash_time = timeind

            # turn off lane changing
            self.l_lc = None
            self.r_lc = None
            self.update_lc_state(timeind)
            # remove all lane and route events to ensure state will not change
            self.lane_events = []
            self.route_events = []

        def set_cf(self, timeind, dt):
            if self.crashed:
                self.acc = self.crash_decel
                if self.speed == 0.:
                    if len(self.speedmem) >= hold_timesteps:
                        if self.speedmem[-hold_timesteps] == 0.:
                            # remove vehicle by creating lane event
                            self.lane_events = [{'pos': -1e6, 'event': 'exit'}]
            else:
                super().set_cf(timeind, dt)

    return VehicleCanCrash
