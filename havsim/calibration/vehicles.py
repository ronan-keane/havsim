"""
CalibrationVehicles are the base Vehicle class used in Calibrations
"""
import havsim.simulation as hs
import numpy as np

class CalibrationVehicleCF(hs.Vehicle):
    """Base CalibrationVehicle class for a second order ODE model.

    CalibrationVehicle/CalibrationVehicleLC is the base class for simulated Vehicles in a Calibration.
    In a Calibration, the order of vehicles is fixed a priori, which allows direct comparison between the
    simulation and some trajectory data. Fixed vehicle order means lane changes, lead vehicles,
    and following vehicles are all predetermined.
    Calibrations can be used to calibrate either just a CF model, just LC model, or both.
    For simulations which do not have fixed vehicle orders, use the simulation api. Compared to the
    Vehicle class, CalibrationVehicles have no road, an unchanging lane used for downstream boundary
    conditions, and have no routes, route events, or lane events. CalibrationVehicleCF is meant for CF
    calibration only, whereas CalibrationVehicle is more general and can be used for both CF/LC calibration.

    Attributes:
        vehid: unique vehicle ID for hashing (float)
        lane: A Lane object which has a get_downstream method, used to apply downstream boundary to the
            Vehicle if it is ever simulated with lead=None
        road: None
        cf_parameters: list of float parameters for cf model
        relax_parameters: float parameter(s) for relaxation model, or None
        relax: if there is currently relaxation, a list of floats or list of tuples giving the relaxation
            values.
        in_relax: bool, True if there is currently relaxation
        relax_start: time index corresponding to relax[0]. (int)
        relax_end: The last time index when relaxation is active. (int)
        minacc: minimum allowed acceleration (float)
        maxacc: maxmimum allowed acceleration(float)
        maxspeed: maximum allowed speed (float)
        hdbounds: tuple of minimum and maximum possible headway.
        eql_type: If 'v', the vehicle's eqlfun accepts a speed and returns a headway. Otherwise it
            accepts a headway and returns a speed.
        lead: leading vehicle, can be either a (subclassed) Vehicle or LeadVehicle
        start: time index of the first simulated time
        end: time index of the last simulated time (or None)
        initpos: position at start
        initspd: speed at start
        leadveh: If the Vehicle has it's own LeadVehicle, leadveh is a reference to it. Otherwise None.
            Only the CalibrationVehicleCF has a reference to leadveh, and is responsible for updating it.
        in_leadveh: True if the leadveh attribute is the current leader.
        leadmem: list of tuples, where each tuple is (lead vehicle, time) giving the time the ego vehicle
            first begins to follow the lead vehicle.
        posmem: list of floats giving the position, where the 0 index corresponds to the position at start
        speedmem: list of floats giving the speed, where the 0 index corresponds to the speed at start
        relaxmem: list of tuples where each tuple is (first time, last time, relaxation) where relaxation
            gives the relaxation values for between first time and last time
        pos: position (float)
        speed: speed (float)
        hd: headway (float)
        len: vehicle length (float)
        acc: acceleration (float)
        y: target given to loss function (e.g. the position time series from data)
    """
    def __init__(self, vehid, y, initpos, initspd, start, leadstatemem, leadinittime, length=3, lane=None,
                 accbounds=None, maxspeed=1e4, hdbounds=None, eql_type='v'):
        """Inits CalibrationVehicleCF. Cannot be used for simulation until initialize is called.

        Args:
            vehid: unique vehicle ID for hashing, float
            y: target for loss function, e.g. a 1d numpy array of np.float64
            initpos: initial position, float
            initspd: initial speed, float
            start: first time of simulation, float
            leadstatemem: list of tuples of floats. Gives the LeadVehicle state at the corresponding
                time index.
            leadinittime: float of time index that 0 index of leadstatemem corresponds to
            length: float vehicle length
            lane: Lane object, its get_downstream is used for downstream boundary conditions.
            accbounds: list of minimum/maximum acceleration. If None, defaults to [-7, 3]
            maxspeed: float of maximum speed.
            hdbounds: list of minimum/maximum headway. Defaults to [0, 10000].
            eql_type: 'v' If eqlfun takes in speed and outputs headway, 's' if vice versa. Defaults to 'v'.
        """
        self.vehid = vehid
        self.len = length
        self.y = y
        self.initpos = initpos
        self.initspd = initspd
        self.start = start

        self.road = None
        self.lane = lane

        if accbounds is None:
            self.minacc, self.maxacc = -7, 3
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed
        self.hdbounds = (0, 1e4) if hdbounds is None else hdbounds
        self.eql_type = eql_type

        if leadstatemem is not None:
            self.leadveh = LeadVehicle(leadstatemem, leadinittime)
        self.in_leadveh = False

    def set_relax(self, relaxamounts, timeind, dt):
        """Applies relaxation given the relaxation amounts."""
        rp = self.relax_parameters
        if rp is None:
            return
        relaxamount_s, relaxamount_v = relaxamounts
        hs.relaxation.relax_helper_vhd(rp, relaxamount_s, relaxamount_v, self, timeind, dt)

    def update(self, timeind, dt):
        """Update for longitudinal state. Updates LeadVehicle if applicable."""
        super().update(timeind, dt)
        if self.in_leadveh:
            self.leadveh.update(timeind+1)

    def loss(self):
        """Calculates loss."""
        T = self.leadmem[-1][1] if self.leadmem[-1][0] is None else len(self.posmem)+self.start
        endind = min(T-self.start, len(self.y))
        return sum(np.square(self.posmem[:endind] - self.y[:endind]))/endind

    def initialize(self, parameters):
        """Resets memory, applies initial conditions, and sets the parameters for the next simulation."""
        # initial conditions
        self.lead = self.fol = None
        self.pos = self.initpos
        self.speed = self.initspd
        # reset relax
        self.in_relax = False
        self.relax = None
        self.relax_start = None
        # memory
        self.end = None
        self.leadmem = []
        self.posmem = [self.pos]
        self.speedmem = [self.speed]
        self.relaxmem = []

        self.set_parameters(parameters)

    def set_parameters(self, parameters):
        """Set cf_parameters and any other parameters which can change between runs."""
        self.cf_parameters = parameters[:-1]
        self.maxspeed = parameters[0]-.1
        self.relax_parameters = parameters[-1]

    def __repr__(self):
        return ' vehicle '+str(self.vehid)


class CalibrationVehicle(CalibrationVehicleCF):
    """Base CalibrationVehicle, which calibrates car following and full lane changing model.

    Extra attributes compared to CalibrationVehicleCF
    y_lc: used to calculate loss of lane changing actions
    lcmem: holds lane changing actions
    folmem: memory for follower, lfol, rfol, llead, rlead
    lfolmem
    rfolmem
    lleadmem
    rleadmem
    l_lc: None, 'discretionary' or 'mandatory' - controls what state lane changing model is in
    r_lc: None, 'discretionary' or 'mandatory' - controls what state lane changing model is in
    """
    def __init__(self, vehid, y, y_lc, initpos, initspd, start, length=3, lane=None, accbounds=None,
                 maxspeed=1e4, hdbounds=None, eql_type='v'):
        super().__init__(vehid, y, initpos, initspd, start, None, None, length=length, lane=lane,
                         accbounds=accbounds, maxspeed=maxspeed, hdbounds=hdbounds, eql_type=eql_type)
        del self.in_leadveh
        self.y_lc = y_lc
        self.llane = self.rlane = lane  # give values to rlane/llane for mobil model, not needed in general

    def update(self, timeind, dt):
        super(CalibrationVehicleCF, self).update(timeind, dt)

    def set_relax(self, timeind, dt):
        super(CalibrationVehicleCF, self).set_relax(timeind, dt)

    def loss(self):
        # TODO do something to calculate the loss over lane changing actions
        lc_loss = 0
        return super().loss() + lc_loss

    def initialize(self, parameters):
        super().initialize(parameters)

        # vehicle orders
        self.lfol = self.rfol = self.llead = self.rlead = None
        # memory for lc model
        self.folmem = []
        self.lfolmem = []
        self.rfolmem = []
        self.lleadmem = []
        self.rleadmem = []
        self.lcmem = []

    def set_parameters(self, parameters):
        self.cf_parameters = parameters[:5]
        self.relax_parameters = parameters[5]
        self.maxspeed = parameters[0]-.1
        self.lc_parameters = parameters[6:15]
        self.shift_parameters = parameters[15:17]
        self.coop_parameter = parameters[17]

    def set_lc(self, timeind, dt):
        """Call Vehicle.set_lc, and append result to lcmem."""
        # need to set the correct leaders for lfol, rfol
        self.lfol.lead = self.llead
        self.rfol.lead = self.rlead

        lc_action = {}
        super().set_lc(lc_action, timeind, dt)
        if self in lc_action:
            self.lcmem.append((lc_action[self], timeind))


class LeadVehicle:
    """Used for simulating a vehicle which follows a predetermined trajectory - it has no models.

    A LeadVehicle acts as a regular Vehicle, but has no models or parameters. It has cf_parameters = None
    which marks it as a LeadVehicle. Their state is updated from a predfined memory. They are used to hold
    trajectories which are not simulated, but which interact with simulated Vehicles.
    """
    def __init__(self, leadstatemem, start, length=None, initstate=(None, None)):
        """
        leadstatemem - list of tuples, each tuple is a pair of (position, speed)
        start - leadstatemem[0] corresponds to time start
        length - length of vehicle (can possibly change)
        initstate - sets the initial state of the vehicle when initialize is called
        """
        self.leadstatemem = leadstatemem
        self.start = start
        self.end = self.start+len(leadstatemem)-1
        self.road = None
        self.lane = None
        self.cf_parameters = None
        self.len = length
        self.initstate = initstate

    def initialize(self, *args):
        """Sets initial state."""
        self.pos, self.speed = self.initstate

    def update(self, timeind, *args):
        """Update position/speed."""
        self.pos, self.speed = self.leadstatemem[timeind - self.start]

    def set_len(self, length):
        """Set len so headway can be computed correctly."""
        self.len = length

    def get_cf(self, *args):
        """Return 0 for cf model - so you don't have to check for LeadVehicles inside set_lc."""
        return 0

    def set_relax(self, *args):
        """Do nothing - so you don't have to check for LeadVehicles when applying relax."""
        pass