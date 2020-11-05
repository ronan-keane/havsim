"""Refactors the functionality of the calibration.opt module.

The simulation module does an entire micro simulation. The calibration module supports simulations where
the lane changing times and vehicle orders fixed apriori to match trajecotry data. This allows direct
comparison with the trajectory data, removing the need to only calibrate to the macroscopic, aggreagted data.
Either single vehicles, or strings (platoons) of vehicles can be simulated.
"""

import numpy as np
import havsim.simulation as hs
from havsim.simulation.road_networks import get_headway
import math


class CalibrationVehicle(hs.Vehicle):
    """Base CalibrationVehicle class for a second order ODE model.

    CalibrationVehicle is the base class for simulated Vehicles in a Calibration. In a Calibration, the
    order of vehicles is fixed a priori, which allows direct comparison between the simulation and some
    trajectory data. Fixed vehicle order means lane changes, lead vehicles, and following vehicles are all
    predetermined. Calibrations can be used to calibrate either just a CF model, just LC model, or both.
    For simulations which do not have fixed vehicle orders, use the simulation api. Compared to the
    Vehicle class, CalibrationVehicles have no road, an unchanging lane used for downstream boundary
    conditions, and have no routes, route events, or lane events. CalibrationVehicle is meant for CF
    calibration only, whereas CalibrationVehicleLC can be used for both CF and LC calibration.

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
            Only the CalibrationVehicle has a reference to leadveh, and is responsible for updating it.
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
        """Inits CalibrationVehicle. Cannot be used for simulation until initialize is called.

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


class CalibrationVehicleLC(CalibrationVehicle):
    """Base CalibrationVehicle, which calibrates car following and full lane changing model.

    Extra attributes compared to CalibrationVehicle
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
        super(CalibrationVehicle, self).update(timeind, dt)

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


class Calibration:
    """Does a simulation of a single CalibrationVehicle, and returns the loss.

    Attributes:
        all_vehicles: list of all vehicles in the simulation
        all_add_events: list of all add events in the simulation
        all_lc_events: list of all lead change events in the simulation
        start: first time index
        end: last time index
        dt: timestep
        addtime: the next time index when an add event occurs
        lctime: the next time index when a lead change event occurs
        timeind: current time index of simulation
        vehicles: set of all vehicles currently in simulation
        add_events: sorted list of remaining add events
        lc_events: sorted list of remaining lead change events
        lc_event: function that can apply a lc_events
    """
    def __init__(self, vehicles, add_events, lc_events, dt, lc_event_fun=None, end=None,
                 run_type='max endtime', parameter_dict=None, ending_position=math.inf):
        """Inits Calibration.

        Args:
            vehicles: list of all Vehicle objects to be simulated.
            add_events: list of add events, sorted in time
            lc_events: list of lead change (lc) events, sorted in time
            dt: timestep, float
            lc_event_fun: can give a custom function for handling lc_events, otherwise we use the default
            end: last time index which is simulated. The start is inferred from add_events.
            run_type: type of calibration we are running. 'max endtime' simulates to the last end while
                'all vehicles' simulates until all vehs have left the simulation
            parameter_dict: dictionary where keys are indices and values are starting, ending indices
                for slicing the parameters. If None, we infer the dictionary by assuming all vehicles
                have the same number of parameters.
            ending_position: ending position used as position to begin removing vehs from simulation -
                defaults to math.inf, so vehicles will not be removed.
        """
        self.run_type = run_type
        self.parameter_dict = parameter_dict
        self.ending_position = ending_position
        self.all_vehicles = vehicles
        self.all_add_events = add_events
        self.all_lc_events = lc_events
        if lc_event_fun is None:
            self.lc_event = lc_event
        else:
            self.lc_event = lc_event_fun

        self.start = min([add_events[i][0] for i in range(len(add_events))])
        self.end = end
        self.dt = dt

    def step(self):
        """Logic for a single simulation step. Main logics are in update_calibration."""
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)


        self.addtime, self.lctime = update_calibration(self.vehicles, self.add_events, self.lc_events,
                                                       self.addtime, self.lctime, self.timeind, self.dt,
                                                       self.lc_event, self.ending_position)

        self.timeind += 1

    def simulate(self, parameters):
        """Does a full simulation and returns the loss.

        Args:
            parameters: list of parameters for the vehicles.

        Returns:
            loss (float).
        """
        self.initialize(parameters)

        # do simulation by calling step repeatedly
        if self.run_type == "max endtime":
            for i in range(self.end - self.start):
                if not self.add_events and not self.vehicles:
                    break
                self.step()

        elif self.run_type == "all vehicles":
            while self.vehicles and self.add_events:
                self.step()

        # calculate and return loss
        loss = 0
        for veh in self.all_vehicles:
            loss += veh.loss()
        return loss

    def initialize(self, parameters):
        """Assigns parameters, initializes vehicles and events."""
        # assign parameters and initialize Vehicles.
        if not self.parameter_dict:
            param_size = int(len(parameters) / len(self.all_vehicles))
            for veh_index in range(len(self.all_vehicles)):
                veh = self.all_vehicles[veh_index]
                param_start_index = int(veh_index * param_size)
                veh.initialize(parameters[param_start_index:param_start_index+param_size])
        else:
            for k, v in self.parameter_dict.items():
                veh = self.all_vehicles[k]
                start_index, end_index = v[0], v[1]
                veh.initialize(parameters[start_index:end_index])

        # initialize events, add the first vehicle(s), initialize vehicles headways/LeadVehicles
        self.vehicles = set()
        self.add_events = self.all_add_events.copy()
        self.lc_events = self.all_lc_events.copy()
        self.addtime = self.add_events[-1][0]
        self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
        self.timeind = self.start
        self.addtime = update_add_event(self.vehicles, self.add_events, self.addtime, self.timeind-1, self.dt,
                                        self.lc_event)
        # TODO
        # for veh in self.vehicles:  # pretty sure we can remove this?
        #     if veh.in_leadveh:
        #         veh.leadveh.update(self.timeind)
        #     veh.hd = get_headway(veh, veh.lead)


def update_calibration(vehicles, add_events, lc_events, addtime, lctime, timeind, dt, lc_event,
                       ending_position):
    """Main logic for a single step of the Calibration simulation.

    At the beginning of the timestep, vehicles/states/events are assumed to be fully updated. Then, in order,
        -call each vehicle's cf model (done in Calibration.step).
        -check for lead change events in the next timestep, and apply them if applicable.
        -update all vehicle's states and headway, for the next timestep.
        -check if any vehicles reach the end of the network, and remove them if so.
        -check for add events, and add the vehicles if applicable.
    Then, when the timestep is incremented, all vehicles/states/events are fully updated, and the iteration
    can continue.
    Note that this logic for the update order is the same in havsim.simulation, albeit the actual update
    functions themselves are simplified.

    Args:
        vehicles: set of vehicles currently in simulation
        add_events: sorted list of current add events
        lc_events: sorted list of current lead change events
        addtime: next time an add event occurs
        lctime: next time a lead change event occurs
        timeind: time index
        dt: timestep
        lc_event: function to apply a single entry in lc_events
    """
    lctime = update_lc_event(lc_events, lctime, timeind, dt, lc_event)

    for veh in vehicles:
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None:
            veh.hd = get_headway(veh, veh.lead)


    #removing vehs that have an end position above the ending_position attribute
    remove_list = remove_vehicles(vehicles, ending_position, timeind)
    for remove_vec in remove_list:
        vehicles.remove(remove_vec)

    addtime = update_add_event(vehicles, add_events, addtime, timeind, dt, lc_event)

    return addtime, lctime


class CalibrationLC(Calibration):
    def __init__(self, vehicles, leadvehicles, add_events, lc_events, dt, end=None, run_type='max endtime',
                 parameter_dict=None, ending_position=math.inf):
        super().__init__(vehicles, add_events, lc_events, dt, end=end, run_type=run_type,
                         parameter_dict=parameter_dict, ending_position=ending_position)
        self.all_leadvehicles = leadvehicles
        del self.lc_event

    def step(self):
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)
        for veh in self.vehicles:
            veh.set_lc(self.timeind, self.dt)

        # TODO need add events and lc events. I think add and lc events should be methods of Calibration?
        # we need to have seperate add events and lc events, but the order of updates is exactly the same.
        self.addtime, self.lctime = update_calibration_lc(self.vehicles, self.leadvehicles, self.add_events,
                                                          self.lc_events, self.addtime, self.lctime, self.timeind, self.dt)
        # only difference is that when we call veh.update for veh in vehicles, we also need to call
        # veh.update for veh in leadvehicles. Suggest to just write a new update_calibration_lc as it is only
        # like 15 lines of code.

        self.timeind += 1

    def initialize(self, parameters):
        super().initialize(parameters)  # don't actually call update_add_event as we need its own add events
        self.leadvehicles = set()
        for veh in self.all_leadvehicles:
            veh.initialize

######### List of requirements for new add and lc events
# add events, in addition to adding vehicles, now need to also add and remove leadvehicles as necessary.
# lc events should keep ALL the vehicle orders updated (currently only update a vehicle's lead, sometimes update fol)
# lc events should apply l_lc, r_lc if needed. Whenever setting l_lc or r_lc, the update_lc_state method of vehicle must be called
# if a calibration vehicle completes a lane change:
    # apply relaxation existing lc event already does this, but now things are simpler because there is no special case for veh.in_leadveh
    # also need to call update_lc_state after completing a lane change

############## Notes on leadvehicles
# the same lead vehicle may act as lfol, rfol, etc. for several vehicles at the same time.
# all lead, fol, etc. must be either a CalibrationVehicle or LeadVehicle at all times. The vehicle order is
# always defined by the data. Replace any None for a follower, lfol, rfol with a LeadVehicle at the starting position
# also, LeadVehicles need an initstate now

#psuedo code for making the add/lc events in make_calibration
# for all vehicles, compute the set of all vehicles which are needed to be leadvehicles. Then for each
# leadvehicle, find the times it needs to be in the simulation, so it can be created.
# find anytime a vehicle order changes, - make a lc event for all of them. Should a single lc event
# be able to update several vehicle orders (e.g. update lead, lfol, rfol, llead, rlead in the same event)?
# maybe there should be seperate events for each update.


# INPROGRESS: add/lc events for make_calibration for LC model
def make_lc_events_new(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list, all_leadvehicles):
    for veh in vehicles:
        curveh = id2obj[veh]
        t0, t1 = vehdict[veh].longest_lead_times

        info_list = [(vehdict[veh].leadmem.intervals(t0, t1), "l"),
        (vehdict[veh].rleadmem.intervals(t0, t1), "rl"),
        (vehdict[veh].lleadmem.intervals(t0, t1), "ll"),
        (vehdict[veh].folmem.intervals(t0, t1), "f"),
        (vehdict[veh].lfolmem.intervals(t0, t1), "lf"),
        (vehdict[veh].rfolmem.intervals(t0, t1), "rl")]


        for tup_index in range(len(info_list)):
            tup = info_list[tup_index]
            info, fl_type = tup[0], tup[1]
            if info == 0:
                create_add_events(info, id2obj, curveh, vehdict, vehicles, dt, addevent_list, lcevent_list, all_leadvehicles, fl_type, True)
            else:
                create_add_events(info, id2obj, curveh, vehdict, vehicles, dt, addevent_list, lcevent_list, all_leadvehicles, fl_type)

    #Remove and add events for lead_vehicles
    for leadveh_id in all_leadvehicles:
        start = all_leadvehicles[leadveh_id][0]
        end = all_leadvehicles[leadveh_id][1]
        fol_lead_veh = id2obj[leadveh_id]

        curevent = (start, 'add', fol_lead_veh)
        addevent_list.append(curevent)

        curevent = (end, 'remove', fol_lead_veh)
        addevent_list.append(curevent)


    return addevent_list, lcevent_list, leadveh_list


def make_leadvehicles(vehicles, id2obj, vehdict, dt):
    # find all LeadVehicles and their start/end times
    all_leadvehicles = {}  # keys are LeadVehicles id, values are tuples of (start, end)
    vehmem_list = ['leadmem', 'lleadmem', 'rleadmem', 'folmem', 'lfolmem', 'rfolmem']
    for veh in vehicles:
        for curvehmem in vehmem_list:
            vehmem = getattr(vehdict[veh], curvehmem)
            for i in vehmem.data:
                curveh, curstart, curend = i
                if curveh in all_leadvehicles:
                    start, end = all_leadvehicles[curveh]
                    all_leadvehicles[curveh] = (min(curstart, start), max(curend, end))
                else:
                    all_leadvehicles[curveh] = (curstart, end)

    # init all leadVehicles and add to id2obj
    for curlead in all_leadvehicles:
        start, end = all_leadvehicles[curlead]
        curleaddata = vehdict[curlead]

        leadstatemem = list(zip(curleaddata.posmem[start:end+1], curleaddata.speedmem[start:end+1]))
        length = curleaddata.len
        if start-1 < curleaddata.start:
            initstate = (curleaddata.posmem[start]-dt*curleaddata.speedmem[start], curleaddata.speedmem[start])
        else:
            initstate = (curleaddata.posmem[start-1], curleaddata.speedmem[start-1])

        id2obj[curlead] = LeadVehicle(leadstatemem, start, length=length, initstate = initstate)

    return all_leadvehicles


# INPROGRESS: Helper function to create lc/add events for LC model
def create_add_events(veh_data, id2obj, curveh, vehdict, vehicles, dt, addevent_list, lcevent_list, all_leadvehicles, fl_type, first_call=False):
    # enumerating through the data for fol/lead vehs for a certain simulated veh
    for count, j in enumerate(veh_data):
        # need a better variable name
        fol_lead_veh, start, end = j
        # if there is a switch to no veh in this fl_type relationship
        if not fol_lead_veh:
            curevent = (start, 'lc', curveh, None, fl_type)
            lcevent_list.append(curevent)
        else:
            fol_lead_veh, curlen = id2obj[fol_lead_veh], None


            if count == 0:
                # lc event for first tims
                curevent = (start, 'lc', curveh, fol_lead_veh, curlen, fl_type)
                # only needed for the first time to create the add cur_veh event
                if first_call:
                    curevent = (start, 'add', curveh, curevent)
                addevent_list.append(curevent)

            else:
                # lc event for fol_lead_veh with respect to curveh
                curevent = (start, 'lc', curveh, fol_lead_veh, fl_type)
                lcevent_list.append(curevent)



def remove_vehicles(vehicles, endpos, timeind):
    """See if vehicle needs to be removed from simulation."""
    remove_list = []
    for veh in vehicles:
        if veh.pos > endpos:
            veh.end = timeind+1
            remove_list.append(veh)
            if veh.fol is not None:
                veh.fol.lead = None
                if not veh.fol.end:  # handles edge case with collisions
                    veh.fol.leadmem.append([None, timeind+1])

    return remove_list


def update_lc_event(lc_events, lctime, timeind, dt, lc_event):
    """Check if we need to apply the next lc event, apply it and update lctime if so.

    Args:
        lc_events: sorted list of current lead change events
        lctime: next time a lead change event occurs
        timeind: time index
        dt: timestep
        lc_event: function to apply an lc_event
    """
    if lctime == timeind+1:
        lc_event(lc_events.pop(), timeind, dt)
        lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
        if lctime == timeind+1:
            while lctime == timeind+1:
                lc_event(lc_events.pop(), timeind, dt)
                lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
    return lctime


def update_add_event(vehicles, add_events, addtime, timeind, dt, lc_event):
    """Check if we need to apply the next add event, apply it and update addtime if so.

    Args:
        add_events: sorted list of current add events
        addtime: next time an add event occurs
        timeind: time index
        dt: timestep
        lc_event: function to apply an lc_event
    """
    if addtime == timeind+1:
        add_event(add_events.pop(), vehicles, timeind, dt, lc_event)
        addtime = add_events[-1][0] if len(add_events)>0 else math.inf
        if addtime == timeind+1:
            while addtime == timeind+1:
                add_event(add_events.pop(), vehicles, timeind, dt, lc_event)
                addtime = add_events[-1][0] if len(add_events)>0 else math.inf
    return addtime

# TODO What to do for making the new CalibrationLC? Seperate make_calibration function? Refactor this version?
def make_calibration(vehicles, vehdict, dt, vehicle_class=None, calibration_class=None,
                     event_maker=None, lc_event_fun=None, lanes={}, calibration_kwargs={}):
    """Sets up a Calibration object.

    Extracts the relevant quantities (e.g. LeadVehicle, initial conditions, loss) from the data
    and creates the add/lc event.

    Args:
        vehicles: list of vehicles to add to the Calibration
        vehdict: dictionary of all VehicleData
        dt: timestep
        vehicle_class: subclassed Vehicle to use - if None defaults to CalibrationVehicle
        calibration_class: subclassed Calibration to use - if None defaults to Calibration
        event_maker: specify a function to create custom (lc) events
        lc_event_fun: specify function which handles custom lc events
        lanes: dictionary with keys as lane indexes, values are Lane objects with call_downstream method.
            Used for downstream boundary.
        calibration_kwargs: keyword arguments passed to Calibration
    """
    if vehicle_class is None:
        vehicle_class = CalibrationVehicle
    if calibration_class is None:
        calibration_class = Calibration
    if event_maker is None:
        event_maker = make_lc_event
    if lc_event_fun is None:
        lc_event_fun = lc_event

    # initialize
    vehicle_list = []
    addevent_list = []
    lcevent_list = []
    id2obj = {}  # holds references to the CalibrationVehicle objects we create
    max_end = 0  # maximum possible time loss function can be calculated

    for veh in vehicles:
        # make vehicle objects
        vehdata = vehdict[veh]
        t0, t1 = vehdata.longest_lead_times
        y = np.array(vehdata.posmem[t0:t1+1])
        y_lc = np.array(vehdata.lanemem[t0:t1+1])
        initpos, initspd = vehdata.posmem[t0], vehdata.speedmem[t0]
        length, lane = vehdata.len, vehdata.lanemem[t1]

        needleads = set(vehdata.leads).difference(vehicles)
        # build the leadstatemem in all times [t0, t1], even if it is only needed for a portion of the times.
        if len(needleads)>0:
            leadstatemem = list(zip(vehdata.leadmem.pos[t0:t1+1], vehdata.leadmem.speed[t0:t1+1]))
            leadstart = t0
        else:
            leadstatemem = leadstart = 0

        if vehicle_class == CalibrationVehicle:
            newveh = vehicle_class(veh, y, initpos, initspd, t0, leadstatemem, leadstart, length=length,
                               lane=lanes[lane])
        else:
            newveh = vehicle_class(veh, y, y_lc, initpos, initspd, t0, length=length, lane=lanes[lane])

        vehicle_list.append(newveh)
        id2obj[veh] = newveh
        max_end = max(max_end, t1)

    # create events
    if event_maker == make_lc_event:
        addevent_list, lcevent_list = event_maker(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list)
    else:
        all_leadvehicles = make_leadvehicles(vehicles, id2obj, vehdict, dt)
        addevent_list, lcevent_list, leadveh_list = event_maker(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list, all_leadvehicles)

    addevent_list.sort(key = lambda x: x[0], reverse = True)
    lcevent_list.sort(key = lambda x: x[0], reverse = True)

    # make calibration object
    if calibration_class == Calibration:
        return calibration_class(vehicle_list, addevent_list, lcevent_list, dt, end=max_end,
                                 lc_event_fun=lc_event_fun, **calibration_kwargs)
    else:
        return calibration_class(vehicle_list, leadveh_list, addevent_list, lcevent_list, dt, end=max_end)


def make_lc_event(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list):
    """Makes lc and add events for default Calibration, which includes adding relaxation."""
    for veh in vehicles:
        curveh = id2obj[veh]
        t0, t1 = vehdict[veh].longest_lead_times
        leadinfo =  vehdict[veh].leadmem.intervals(t0, t1)
        for count, j in enumerate(leadinfo):
            # we have an event everytime a leader changes - make the data
            curlead, start, end = j
            curlead_in_vehicles = False  # initialize as False
            leaddata = vehdict[curlead]
            leadstart = leaddata.start

            # even though the change occurs at time start, we want to calculate the relaxation using
            # the differences in headway at time start - 1. This leads to 4 combinations, first, whether
            # the new leader is simulated or not, and second, whether the new lead is available at start-1
            if curlead in vehicles:  # curlead is simulated (in the same calibration object)
                if start-1 < leadstart:  # handle edge case where leadstart = start
                    leadstate = (leaddata.posmem[leadstart]-leaddata.speedmem[leadstart]*dt,
                                 leaddata.speedmem[leadstart])
                else:
                    leadstate = (None,)
                curlead, curlen = id2obj[curlead], None
                curlead_in_vehicles = True
            else:
                curlen = leaddata.len  # curlead is not simulated, it is stored in curveh.leadstatemem
                if start-1 < leadstart:  # handle edge case where leadstart = start
                    leadstate = (leaddata.posmem[leadstart]-leaddata.speedmem[leadstart]*dt,
                                 leaddata.speedmem[leadstart])
                else:
                    leadstate = (leaddata.posmem[start-1], leaddata.speedmem[start-1])

            # decide what case we are in
            if count == 0:  # first event is always an add event to add vehicle to simulation
                userelax = True if t0 > vehdict[veh].start else False
                # make the add event
                curevent = (start, 'lc', curveh, curlead, curlen, userelax, leadstate)
                curevent = (start, 'add', curveh, curevent)
                addevent_list.append(curevent)
            else:  # lc event changes leader, applies relax
                curevent = (start, 'lc', curveh, curlead, curlen, True, leadstate)
                lcevent_list.append(curevent)

            if count+1 == len(leadinfo) and not curlead_in_vehicles:  # set leader to None after leader exits
                curevent = (end, 'lc', curveh, None, 0, False, None)
                lcevent_list.append(curevent)

    return addevent_list, lcevent_list


def add_event(event, vehicles, timeind, dt, lc_event):
    """Adds a vehicle to the simulation and applies the first lead change event.

    Add events occur when a vehicle is added to the Calibration.
    Add events are a tuple of
        start (float) - time index of the event
        'add' (str) - identifies event as an add event
        curveh - CalibrationVehicle object to add to simulation
        lc_event - a lead change event which sets the first leader for curveh.

    Args:
        event: the add event
        vehicles: set of current vehicles in simulation which is modified in place
        timeind: time index
        dt: timestep
    """
    unused, unused, curveh, lcevent = event
    vehicles.add(curveh)
    lc_event(lcevent, timeind, dt)
    if curveh.in_leadveh:
        curveh.leadveh.update(timeind+1)
    if curveh.lead is not None:
        curveh.hd = get_headway(curveh, curveh.lead)


def lc_event(event, timeind, dt):
    """Applies lead change event, updating a CalibrationVehicle's leader.

    Lead change events occur when a CalibrationVehicle's leader changes. In a Calibration, it is
    assumed that vehicles have fixed lc times and fixed vehicle orders.
    Lead change events are a tuple of
        start (float) - time index of the event
        'lc' (str) - identifies event as a lane change event
        curveh - CalibrationVehicle object which has the leader change at time start
        newlead - The new leader for curveh. If the new leader is being simulated, curlead is a
            CalibrationVehicle, otherwise curlead is a float corresponding to the vehicle ID
        leadlen - If newlead is a float (i.e. the new leader is not simulated), curlen is the length of
            curlead. Otherwise, curlen is None (curlead.len gives the length for a CalibrationVehicle)
        userelax - bool, whether to apply relaxation
        leadstate - if the new leader is not simulated, leadstate is a tuple which gives the position/speed
            to use for computing the relaxation amount

    Args:
        event: lead change event
        timeind: time index
        dt: timestep
    """
    unused, unused, curveh, newlead, leadlen, userelax, leadstate = event

    # calculate relaxation amount
    if userelax:
        # get olds/oldv
        if curveh.lead is None:  # rule for merges
            olds, oldv = curveh.get_eql(curveh.speed), curveh.speed
        else:  # normal rule
            olds, oldv = curveh.hd, curveh.lead.speed

        # get news/newv
        uselen = newlead.len if leadlen is None else leadlen
        if leadstate[0] is None:
            newpos, newv = newlead.pos, newlead.speed
        else:
            newpos, newv = leadstate
        news = newpos - uselen - curveh.pos

        # apply relaxation
        relaxamounts = (olds - news, oldv - newv)
        curveh.set_relax(relaxamounts, timeind, dt)

    if not curveh.end:
        update_lead(curveh, newlead, leadlen, timeind)  # update leader


def update_lead(curveh, newlead, leadlen, timeind):
    """Updates leader for curveh.

    Args:
        curveh: Vehicle to update
        newlead: if a float, the new leader is the LeadVehicle for curveh. Otherwise, newlead is the
            Vehicle object of the new leader. newlead can be set to None, in which case the downstream
            boundary is used in place of the car following model
        leadlen: if newlead is a float, leadlen gives the length of the leader so LeadVehicle can be updated.
        timeind: time index of update (change happens at timeind+1)
    """
    if leadlen is None:  # newlead is simulated
        curveh.lead = newlead
        newlead.fol = curveh
        curveh.leadmem.append([newlead, timeind+1])
        curveh.in_leadveh = False
    elif newlead is None:
        curveh.lead = None
        curveh.in_leadveh = False
        curveh.leadmem.append([None, timeind+1])
    else:  # LeadVehicle
        curveh.lead = curveh.leadveh
        curveh.lead.set_len(leadlen)  # must set the length of LeadVehicle
        curveh.leadmem.append([newlead, timeind+1])
        curveh.in_leadveh = True
