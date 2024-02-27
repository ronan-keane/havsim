"""Vehicle class for calling model, storing/saving data."""
import scipy.optimize as sc
import numpy as np

from havsim.road import get_dist, get_headway
from havsim import models
from havsim import update_lane_routes


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
                return ans.root, ans.root/x - leadlen
            else:
                return (ans.root+leadlen)*x, ans.root
        else:
            if output_type == eql_type:
                return ans.root
            elif output_type == 's':
                return ans.root/x - leadlen
            elif output_type == 'v':
                return (ans.root+leadlen)*x
    else:
        raise RuntimeError('could not invert provided equilibrium function')


class Vehicle:
    """Base Vehicle class for simulating vehicle trajectories.

    Vehicles are responsible for implementing the rules to update their positions. This includes a
    'car following' (cf) model which is used to update the position (in the direction of travel).
    There is also a 'lane changing' (lc) model which can update the vehicle's lane and also add
    a separate acceleration which affects the car following behavior.

    The cf model is defined primarily through the method cf_model; the method set_cf implements the per-timestep
    call to cf_model, and the method get_cf exists to evaluate a potential call to cf_model.
    The lc model is defined through the set_lc method. Note that set_lc relies on the get_cf method by default.
    The vehicle is updated by the update method.

    Attributes:
        vehid: unique vehicle ID for hashing
        len: length of vehicle (float)
        lane: Lane object vehicle is currently on
        road: str name of the road lane belongs to
        cf_parameters: list of float parameters for the cf model
        lc_parameters: list of float parameters for the lc model
        lc2_parameters: list of float parameters for the lc model (continued)
        relax_parameters: list of float parameters for relaxation, or None if no relaxation
        route_parameters: list of float parameters for the route model
        relax: if there is currently relaxation, a list of floats or list of tuples giving the relaxation
            values.
        in_relax: bool, True if there is currently relaxation
        relax_start: time index corresponding to relax[0]. (int)
        relax_end: The last time index when relaxation is active. (int)
        route: list of road names (str). When the vehicle first enters the simulation or enters a new road,
            the route gets pop(). Note the last entry of route (-1 index) is the next road in the planned route.
        routemem: route which was used to init vehicle.
        minacc: minimum allowed acceleration (float)
        maxacc: maxmimum allowed acceleration(float)
        maxspeed: maximum allowed speed (float)
        hdbounds: tuple of minimum and maximum possible headway.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
        lc_urgency: for mandatory lane changes, lc_urgency is a tuple of floats which control if
            the ego vehicle can force cooperation (simulating aggressive behavior)
        disc_cooldown: when a vehicle makes a discretionary change, it cannot make another discretionary
            change until after time index disc_cooldown.
        disc_endtime: when a vehicle enters the active discretionary state, it stays in that state until
            time index disc_endtime.
        l_lc: the current lane changing state for the left side, None, 'd' (discretionary) or 'm' (mandatory)
        r_lc: the current lane changing state for the right side, None, 'd' (discretionary) or 'm' (mandatory)
        chk_disc: If True, do additional check to enter into lane change model calculation (for discretionary state).
        in_disc: If True, lane changing model checks discretionary condition. Check mandatory condition otherwise.
        is_coop: if not None, is_coop is a Vehicle that we are currently cooperating with (is_coop is changer).
            Can set the is_coop = 0 to manually remove cooperation.
        has_coop: if not None, has_coop is a Vehicle that is cooperating with self (self is the changer)
        coop_side_fol: if requesting cooperation on left side, has value 'lfol' otherwise 'rfol'
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle needs to take on that lane
        route_events: list of route events for current lane
        lane_events: list of lane events for current lane
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
        pos: current position (float)
        speed: current speed (float)
        hd: current headway (float), or if leader is None, the previous headway
        acc: acceleration (float)
        lc_acc: acceleration due to lane change model
        llane: the Lane to the left of the current lane the vehicle is on, or None
        rlane: the Lane to the right of the current lane the vehicle is on, or None
    """
    # TODO set_route_events should be a method of vehicle? (more generally, better compartmentalization of core methods)
    # TODO numba/cython implementation
    # TODO GradVehicle implementation

    def __init__(self, vehid, curlane, cf_parameters=None, lc_parameters=None, lc2_parameters=None,
                 relax_parameters=None, route_parameters=None, route=None, lead=None, fol=None,
                 lfol=None, rfol=None, llead=None, rlead=None, length=4,
                 eql_type='v', accbounds=None, maxspeed=None, hdbounds=None, seed=None):
        """Inits Vehicle. Cannot be used for simulation until initialize is also called.

        After a Vehicle is created, it is not immediately added to simulation. This is because different
        upstream (inflow) boundary conditions may require to have access to the vehicle's parameters
        and methods before actually adding the vehicle. Thus, to use a vehicle you need to first call
        initialize, which sets the remaining attributes.

        Args:
            vehid: unique vehicle ID for hashing
            curlane: lane vehicle starts on
            cf_parameters: list of float parameters for the cf model (see simulation.models.IDM)
            lc_parameters: list of float parameters for the lc model (see simulation.models.lc_havsim)
            lc2_parameters: list of float parameters for the lc model (see simulation.models.lc_havsim)
            relax_parameters: list of float parameters for relaxation (see simulation.models.lc_havsim)
            route_parameters: list of float parameters for the route model (see simulation.models.lc_havsim)
            route: list of road names (str) which defines the route for the vehicle, in order of the route.
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
            accbounds: tuple of min/max bounds for acceleration.
            maxspeed: maximum allowed speed.
            hdbounds: tuple of bounds for headway.
            seed: integer to seed random number generation (or None if using default seed)
        """
        self.vehid = vehid
        self.len = length
        self.lane = curlane
        self.road = curlane.road if curlane is not None else None
        self.route = [] if route is None else route[-1::-1]

        self.npr = np.random.default_rng(seed=seed)

        # parameters
        self.cf_parameters = cf_parameters if cf_parameters is not None else [37.5, 1.17, 3, 1.7, 1.5]
        self.lc_parameters = lc_parameters if lc_parameters is not None else [-11, -10, .3, .03, 0, 0, .1, 5, 100]
        self.lc2_parameters = lc2_parameters if lc2_parameters is not None else [1, 2, 1, -1, 1, .5]
        self.relax_parameters = relax_parameters if relax_parameters is not None else [11., 4.5, .6, 2.]
        self.route_parameters = route_parameters if route_parameters is not None else [300, 500]
        # bounds
        if accbounds is None:
            self.minacc, self.maxacc = -12, 8
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = self.cf_parameters[0] - .2 if maxspeed is None else maxspeed
        self.hdbounds = (self.cf_parameters[2] + 1e-6, 200) if hdbounds is None else hdbounds
        self.eql_type = eql_type
        # relaxation
        self.in_relax = False
        self.relax = None
        self.relax_start = None
        self.relax_end = None
        # lane changing model
        self.lc_urgency = None
        self.is_coop = None
        self.has_coop = None
        self.coop_side_fol = None
        self.chk_disc = None
        self.in_disc = None
        self.disc_cooldown = -1e10
        self.disc_endtime = -1e10
        # initialize any other attributes
        self.pos = None
        self.hd = None
        self.speed = None
        self.acc = None
        self.lc_acc = None
        self.llane = None
        self.rlane = None
        self.l_lc = None
        self.r_lc = None
        self.start = None
        self.end = None
        self.lane_events = None
        self.route_events = None
        self.cur_route = None

        # leader/follower relationships
        self.lead = lead
        self.fol = fol
        self.lfol = lfol
        self.rfol = rfol
        self.llead = [] if llead is None else llead
        self.rlead = [] if rlead is None else rlead

        # memory
        self.leadmem = []
        self.lanemem = []
        self.posmem = []
        self.speedmem = []
        self.relaxmem = []
        self.routemem = self.route.copy()

    def initialize(self, pos, spd, hd, start):
        """Updates the remaining attributes of the vehicle, making it able to be simulated.

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
        self.lc_acc = 0

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
        elif self.llane.road == self.road:
            self.l_lc = 'd'
        else:
            self.l_lc = None
        self.rlane = self.lane.get_connect_right(pos)
        if self.rlane is None:
            self.r_lc = None
        elif self.rlane.road == self.road:
            self.r_lc = 'd'
        else:
            self.r_lc = None
        self.update_lc_state(start)

        # set lane/route events - sets lane_events, route_events, cur_route attributes
        if len(self.route) > 0:
            update_lane_routes.add_cur_route_to_veh(self)
            update_lane_routes.set_lane_events(self)
            update_lane_routes.set_route_events(self, start)

    def cf_model(self, p, state):
        """Car following model.

        Args:
            p: parameters for model (cf_parameters)
            state: list of headway, speed, leader speed
        Returns:
            float acceleration of the model.
        """
        return p[3]*(1-(state[1]/p[0])**4-((p[2]+state[1]*p[1]+(state[1]*(state[1]-state[2])) /
                                            (2*(p[3]*p[4])**(1/2)))/(state[0]))**2)

    def get_cf(self, lead, timeind):
        """Evaluates car following model if lead was the lead vehicle.

        Args:
            lead (Vehicle): lead Vehicle
            timeind (int): time index
        Returns:
            hd (float or None): if lead is not None, the bumper to bumper distance between self and lead.
            acc (float): longitudinal acceleration for current timestep
        """
        if lead is None:
            acc = self.lane.call_downstream(self, timeind)
            return None, acc
        hd = get_headway(self, lead)
        if hd < 0:
            return hd, -100
        acc = self.cf_model(self.cf_parameters, [hd, self.speed, lead.speed])
        return hd, acc

    def set_cf(self, timeind):
        """Sets a vehicle's acceleration, with relaxation added after lane changing."""
        hd, spd, lead = self.hd, self.speed, self.lead
        if lead is None:
            self.acc = self.lane.call_downstream(self, timeind)
            return
        if self.in_relax:
            p = self.relax_parameters
            if timeind >= self.relax_end:
                self.in_relax = False
                self.relaxmem.append((self.relax, self.relax_start))
                if timeind == self.relax_end:
                    currelax, currelax_v = self.relax[timeind - self.relax_start]
                else:
                    currelax, currelax_v = 0, 0
            else:
                currelax, currelax_v = self.relax[timeind - self.relax_start]

            ttc = max(hd - 2 - p[2]*spd, 0) / (spd - lead.speed + 1e-6)
            if p[3] > ttc >= 0:
                currelax = currelax * (ttc / p[3]) ** 2 if currelax > 0 else currelax
                currelax_v = currelax_v * (ttc / p[3]) ** 2 if currelax_v > 0 else currelax_v

            hd += currelax
            lspd = lead.speed + currelax_v
        else:
            lspd = lead.speed
        if hd < 0:
            self.acc = -100
            return
        self.acc = self.cf_model(self.cf_parameters, [hd, spd, lspd])

    def set_relax(self, timeind, dt):
        """Creates a new relaxation after lane change."""
        models.new_relaxation(self, timeind, dt)

    def free_cf(self, p, spd):
        """Defines car following model in free flow.

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

    def set_lc(self, lc_actions, lc_followers, timeind):
        """Evaluates a vehicle's lane changing action.

        If a vehicle makes a lane change, it must be recorded in the dictionary lc_actions. The LC will be completed
        except in the case where multiple vehicles try to change in front of the same vehicle. In that case, only
        one of the lane changes can be completed.

        The set_lc method may also affect the acceleration by setting the attribute lc_acc. This is a separate
        acceleration from the 'acc' attribute set by set_cf. The lc_acc and acc are added to get the final acceleration.

        See havsim.models.lc_havsim for more information.

        Args:
            lc_actions: dictionary where keys are Vehicles which request to change lanes, values are the side of change
                (either 'l' or 'r')
            lc_followers: For any Vehicle which request to change lanes, the new follower must be a key in lc_followers,
                value is a list of all vehicles which requested change. Used to prevent multiple vehicles from changing
                in front of same follower in the same timestep.
            timeind: time index
        Returns:
            lc_actions, lc_followers. (Note that the self is also modified in place.)
        """
        return models.lc_havsim(self, lc_actions, lc_followers, timeind)

    def update_lc_state(self, timeind, lc=None):
        """Updates the lane changing internal state when completing, aborting, or beginning lane changing.

        Cases when this is called -
            -after a route event ends a discretionary state or begins a mandatory state
            -after the network topology changes (e.g. a new left lane, or right lane ends)
            -after a lane changing is completed (lc=True)
            -when the vehicle is removed from the simulation

        The purpose of this function is to ensure that the lane changing model is in a correct state. This means
        setting the attributes in_disc, chk_disc, and clearing any cooperation that may be applied.
        """
        if lc:
            if self.in_disc:
                self.disc_cooldown = timeind + self.lc_parameters[8]

        if self.is_coop:
            self.is_coop.has_coop = self.is_coop = None
        if self.has_coop:
            coop_veh = self.has_coop
            coop_veh.chk_disc = timeind > coop_veh.disc_endtime if coop_veh.in_disc else False
            coop_veh.is_coop = self.has_coop = None

        l_lc, r_lc = self.l_lc, self.r_lc
        in_disc = (l_lc == 'd' and r_lc != 'm') or (r_lc == 'd' and l_lc != 'm')
        self.in_disc = in_disc
        self.chk_disc = in_disc

    def acc_bounds(self, acc):
        """Apply acceleration bounds."""
        if acc > self.maxacc:
            acc = self.maxacc
        elif acc < self.minacc:
            acc = self.minacc
        return acc

    def update(self, timeind, dt):
        """Applies bounds and updates a vehicle's longitudinal state/memory."""
        acc = self.acc + self.lc_acc
        self.lc_acc = 0  # lc_acc must be reset each timestep

        # bounds on acceleration
        acc = max(self.minacc, acc)
        # bounds on speed
        nextspeed = self.speed + acc*dt
        if nextspeed < 0:
            nextspeed = 0

        # update state
        self.pos += self.speed*dt
        self.speed = nextspeed

        # update memory
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)

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

    def __getstate__(self):
        """Serialize (pickle) the Vehicle."""
        state = self.__dict__.copy()
        del state['npr']
        # remove references to other vehicles/anchors
        state['lead'] = veh_to_id(state['lead'])
        state['fol'] = veh_to_id(state['fol'])
        state['lfol'] = veh_to_id(state['lfol'])
        state['rfol'] = veh_to_id(state['rfol'])
        rlead = [veh_to_id(veh) for veh in state['rlead']]
        llead = [veh_to_id(veh) for veh in state['llead']]
        leadmem = [(veh_to_id(curmem[0]), curmem[1]) for curmem in state['leadmem']]
        state['leadmem'], state['rlead'], state['llead'] = leadmem, rlead, llead
        # simplify memory
        if len(state['speedmem']) > 1 and state['speedmem'][0] != 0:
            dt = (state['posmem'][1] - state['posmem'][0])/state['speedmem'][0]
            state['posmem'] = (state['posmem'][0], dt)
        return state

    def __setstate__(self, state):
        """Load Vehicle from pickle. To get references to other vehicles, must also call vehicles.reload."""
        self.__dict__ = state
        self.npr = np.random.default_rng()
        if len(self.speedmem) > 1 and self.speedmem[0] != 0:
            pos, dt = self.posmem
            posmem = [pos]
            for spd in self.speedmem[:-1]:
                pos = pos + dt * spd
                posmem.append(pos)
            self.posmem = posmem

    def _remove_veh_references(self):
        """Replace reference to other vehicles with vehid (python garbage collection purposes)."""
        self.lead = veh_to_id(self.lead)
        self.fol = veh_to_id(self.fol)
        self.lfol = veh_to_id(self.lfol)
        self.rfol = veh_to_id(self.rfol)
        rlead = [veh_to_id(veh) for veh in self.rlead]
        llead = [veh_to_id(veh) for veh in self.llead]
        leadmem = [(veh_to_id(curmem[0]), curmem[1]) for curmem in self.leadmem]
        self.leadmem, self.rlead, self.llead = leadmem, rlead, llead

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
                print('\npassing results for '+str(self))
            else:
                print('\nerrors for '+str(self))
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


def veh_to_id(veh):
    if hasattr(veh, 'vehid'):
        return veh.vehid
    else:
        return veh


def id_to_veh_maybe(a, veh_dict):
    if a in veh_dict:
        return veh_dict[a]
    else:
        return a


def reload(all_vehicles, lanes=None):
    """Given list of all vehicles, convert int memory back to Vehicle references.

    Args:
        all_vehicles: list of Vehicles which will have their references added back
        lanes: dictionary of Lane objects as keys, values are int index which represents the lane index (for plots)
    """
    # make dict to map str vehids to Vehicles/AnchorVehicles
    veh_dict = {}
    if lanes is not None:
        for lane in lanes.keys():
            if lane.anchor.vehid not in veh_dict:
                veh_dict[lane.anchor.vehid] = lane.anchor
    for veh in all_vehicles:
        veh_dict[veh.vehid] = veh
    for veh in all_vehicles:
        veh.lead = id_to_veh_maybe(veh.lead, veh_dict)
        veh.fol = id_to_veh_maybe(veh.fol, veh_dict)
        veh.lfol = id_to_veh_maybe(veh.lfol, veh_dict)
        veh.rfol = id_to_veh_maybe(veh.rfol, veh_dict)
        llead, rlead, leadmem = veh.llead, veh.rlead, veh.leadmem
        for count, a in enumerate(rlead):
            rlead[count] = id_to_veh_maybe(a, veh_dict)
        for count, a in enumerate(llead):
            llead[count] = id_to_veh_maybe(a, veh_dict)
        for count, curmem in enumerate(leadmem):
            leadmem[count] = (id_to_veh_maybe(curmem[0], veh_dict), curmem[1])
    return all_vehicles


class StochasticVehicle(Vehicle):
    """Adds error-prone behavior to Vehicle, which makes it possible for crashes to occur.

    Note that this class does not define what should happen after a crash. That is done by CrashesVehicle.
    This class adds stochastic behavior to the Vehicle.

    Attributes:
        gamma_parameters: list of float parameters, controlling the human reaction time/distraction
            0 - mean for log normal. Larger = higher average reactions (less safe)
            1 - stdev for log normal. Larger = heavier tail reactions (less safe)
            2 - scaling factor for squared acceleration. The gamma distraction length is scaled by
                p[2]*acc**2 + p[3]*abs(acc) where acc is the acceleration. (larger = more safe)
            3 - scaling factor for acceleration. (larger = more safe)
            4 - scaling factor for lane changes. The gamma distraction length is scaled by p[4] if
                a lane change occurs. (larger = more safe)
        xi_parameters:
            0 - scale for pareto  (Larger = larger xi = less safe)
            1 - shape for pareto  (Larger = smaller tail = more safe)
        lc_accmem: mem of lc_acc
        r_acc: acceleration with no relaxation model, used for computing scaling of gamma
        prev_acc: previous acceleration
        prev_lc_acc: previous lc acceleration
        beta: float percentage of timestep that we should be distracted for in next_t_ind
        next_t_ind: next timestep we can update the acceleration
    """
    def __init__(self, vehid, curlane, gamma_parameters=None, xi_parameters=None, **kwargs):
        super().__init__(vehid, curlane, **kwargs)
        self.gamma_parameters = gamma_parameters if gamma_parameters is not None else [-.1, .3, .5, 2., 2.]
        self.xi_parameters = xi_parameters if xi_parameters is not None else [.15, 3]
        self.lc_accmem = []
        # self.gammamem = []  # memory of random variables (not currently used)
        # self.ximem = []

        self.r_acc = 0
        self.prev_acc = 0  # old cf acc
        self.prev_lc_acc = 0  # old lc acc
        self.beta = 0  # next attention at time next_t_ind + beta
        self.next_t_ind = None

    def initialize(self, pos, spd, hd, start):
        super().initialize(pos, spd, hd, start)
        self.next_t_ind = start

    def set_cf(self, timeind):
        if timeind == self.next_t_ind:
            super().set_cf(timeind)
            if self.in_relax:
                if self.lead is not None:
                    self.r_acc = abs(self.cf_model(self.cf_parameters, [self.hd, self.speed, self.lead.speed]))
                    return
            self.r_acc = 0.
        else:
            self.acc = self.prev_acc

    def set_lc(self, lc_actions, lc_followers, timeind):
        return models.stochastic_lc_havsim(self, lc_actions, lc_followers, timeind)

    def set_relax(self, timeind, dt):
        # modify gamma when lane change occurs, according to gamma_parameters[-1] (if gamma was not already adjusted)
        if not self.in_relax:
            t_left = self.next_t_ind + self.beta - timeind
            new_gamma = t_left/self.gamma_parameters[-1]
            bar_gamma = new_gamma // 1.
            if bar_gamma == 0. and timeind < self.next_t_ind:
                self.beta = 0
                self.next_t_ind = timeind + 1
            else:
                self.beta = new_gamma - bar_gamma
                self.next_t_ind = timeind + int(bar_gamma)
            # self.gammamem[-1] = (*self.gammamem[1][:3], True)

        models.new_relaxation(self, timeind, dt)

    def update(self, timeind, dt):
        if timeind == self.next_t_ind:
            new_acc = self.acc
            self.acc = self.prev_acc * self.beta + new_acc * (1 - self.beta)
            gamma_acc = max(abs(new_acc), self.r_acc, abs(self.acc + self.lc_acc), abs(new_acc + self.lc_acc))
            gamma = self.sample_gamma(gamma_acc, timeind)
            bar_gamma = (gamma / dt) // 1.
            self.beta = gamma / dt - bar_gamma
            self.next_t_ind = timeind + int(bar_gamma) + 1
            self.prev_lc_acc = self.lc_acc
            self.prev_acc = new_acc
        else:
            self.lc_acc = self.prev_lc_acc
        self.lc_accmem.append(self.lc_acc)

        super().update(timeind, dt)

    def sample_gamma(self, acc, timeind):
        p = self.gamma_parameters
        if self.in_relax:
            scale = p[4] * (p[2]*acc**2 + p[3]*acc + 1)
        else:
            scale = p[2]*acc**2 + p[3]*acc + 1
        gamma = np.exp(self.npr.standard_normal()*p[1] + p[0])/scale
        # self.gammamem.append((timeind, gamma, acc, self.in_relax))
        return gamma

    def sample_xi(self, timeind):
        p = self.xi_parameters
        xi = p[0]/(self.npr.random()**(1/p[1]))-p[0]
        # self.ximem.append((timeind, xi))
        return xi


def update_after_crash(veh, timeind, crashed):
    """Called after a crash to implement the crash behavior."""
    veh.crashed = crashed
    veh.crash_time = timeind

    # turn off lane changing
    veh.l_lc = None
    veh.r_lc = None
    veh.update_lc_state(timeind)
    veh.is_coop = 0
    # remove all lane and route events to ensure state will not change
    veh.lane_events = []
    veh.route_events = []


def set_cf_crashed(veh, timeind):
    """After crash, set constant deceleration at -2. Remove vehicle after 50 timesteps."""
    if veh.lead is not None:
        if veh.hd < 0:
            veh.acc = -100
        else:
            test_acc = veh.cf_model(veh.cf_parameters, [veh.hd, veh.speed, veh.lead.speed])
            veh.acc = min(-2, test_acc)
    else:
        veh.acc = -2

    if timeind >= veh.crash_time + 50:
        veh.lane_events = [{'pos': -1e6, 'event': 'exit'}]


class CrashesVehicle(Vehicle):
    """Keeps memory of crashes/near misses and implement the crash behavior if a crash occurs.

    Attributes:
        crashed: if False, the vehicle has not been in a crash. Otherwise, is a tuple of (crash_type, crash_time)
            where crash_type is a str and crash_time is the earliest crash time (int) of any vehicle in the crash
        crash_time: if not None, the time (int) that the vehicle crashed
        near_misses: if not [], list of tuples of (start_time, end_time) of near miss status
    """
    def __init__(self, *args, **kwargs):
        self.crashed = False
        self.crash_time = None
        self.near_misses = []
        super().__init__(*args, **kwargs)

    def update_after_crash(self, timeind, crashed):
        update_after_crash(self, timeind, crashed)

    def set_cf(self, timeind):
        if self.crashed:
            set_cf_crashed(self, timeind)
        else:
            super().set_cf(timeind)


class CrashesStochasticVehicle(StochasticVehicle):
    def __init__(self, *args, **kwargs):
        self.crashed = False
        self.crash_time = None
        self.near_misses = []
        super().__init__(*args, **kwargs)

    def update_after_crash(self, timeind, crashed):
        update_after_crash(self, timeind, crashed)

    def set_cf(self, timeind):
        if self.crashed:
            set_cf_crashed(self, timeind)
        else:
            super().set_cf(timeind)
