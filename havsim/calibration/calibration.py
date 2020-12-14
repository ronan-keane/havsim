"""Calibration objects do simulation with fixed vehicle orders, compute loss on trajectory level.

The simulation module does an entire micro simulation. The calibration module supports simulations where
the lane changing times and vehicle orders fixed apriori to match trajectory data. This allows direct
comparison with the trajectory data, removing the need to only calibrate to the macroscopic, aggreagted data.
Either single vehicles, or strings (platoons) of vehicles can be simulated.
"""

import numpy as np
from havsim.simulation.road_networks import get_headway
from havsim.calibration.vehicles import LeadVehicle
import math


class CalibrationCF:
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
        lc_event_fun: function that can apply lc_events (default is apply_calibrationcf_lc_event)
    """
    def __init__(self, vehicles, add_events, lc_events, dt, end=None, lc_event_fun=None,
                 run_type='max endtime', parameter_dict=None, ending_position=math.inf):
        """Inits CalibrationCF.

        Args:
            vehicles: list of all Vehicle objects to be simulated.
            add_events: list of add events, sorted in time
            lc_events: list of lead change (lc) events, sorted in time
            dt: timestep, float
            end: last time index which is simulated. The start is inferred from add_events.
            lc_event_fun: function which applies lc events, or None if using default.
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
        self.lc_event_fun = apply_calibrationcf_lc_event if lc_event_fun is None else lc_event_fun

        self.start = min([add_events[i][0] for i in range(len(add_events))])
        self.end = end
        self.dt = dt

    def step(self):
        """Logic for a single simulation step. Main logics are in update_calibration_cf."""
        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)

        self.addtime, self.lctime = update_calibration_cf(self.vehicles, self.add_events, self.lc_events,
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
        # had to change these from + 1?
        self.addtime = self.add_events[-1][0] + 1
        self.lctime = self.lc_events[-1][0] + 1 if len(self.lc_events)>0 else math.inf
        self.timeind = self.start
        self.update_add_events(self.timeind-1, self.dt)


    def update_lc_events(self, timeind, dt):
        """Check if we need to apply the next lc event, apply it and update lctime if so.

        See function apply_calibrationcf_lc_event.
        """
        if self.lctime == timeind+1:
            self.lc_event_fun(self.lc_events.pop(), timeind, dt)
            self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
            self.update_lc_events(timeind, dt)

    def update_add_events(self, timeind, dt):
        """Check if we need to apply the next add event, apply it and update addtime if so.

        See function apply_calibrationcf_add_event.
        """
        if self.addtime == timeind+1:
            apply_calibrationcf_add_event(self.add_events.pop(), self.vehicles, timeind, dt,
                                          self.lc_event_fun)
            self.addtime = self.add_events[-1][0] if len(self.add_events)>0 else math.inf
            self.update_add_events(timeind, dt)


def update_calibration_cf(vehicles, update_lc_fun, update_add_fun, timeind, dt, ending_position):
    """Main logic for a single step of the CalibrationCF simulation.

    At the beginning of the timestep, vehicles/states/events are assumed to be fully updated. Then, in order,
        -call each vehicle's cf model (done in CalibrationCF.step).
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
        update_lc_fun: CalibrationCF.update_lc_events method, which applies lead change events
        update_add_fun: CalibrationCF.update_add_events method, which applies add events
        timeind: time index
        dt: timestep
    """
    update_lc_fun(timeind, dt)

    for veh in vehicles:
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None:
            veh.hd = get_headway(veh, veh.lead)

    #removing vehs that have an end position above the ending_position attribute
    remove_list = remove_vehicles(vehicles, ending_position, timeind)
    for remove_vec in remove_list:
        vehicles.remove(remove_vec)

    update_add_fun(timeind, dt)



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

def update_calibration(vehicles, leadvehicles, update_lc_fun, update_add_fun, timeind, dt, ending_position, dummy_vec):
    update_lc_fun(timeind, dt)

    for veh in vehicles:
        veh.update(timeind, dt)
    for veh in leadvehicles:
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None:
            veh.hd = get_headway(veh, veh.lead)

    update_add_fun(timeind, dt)

    remove_list = remove_vehicles_lc(vehicles, ending_position, dummy_vec, timeind)
    for remove_vec in remove_list:
        vehicles.remove(remove_vec)

def remove_vehicles_lc(vehicles, endpos, dummy_vec, timeind):
    remove_list = []
    for veh in vehicles:
        if veh.pos > endpos:
            veh.end = timeind+1
            lfolupdate(veh, veh.lfol, timeind)
            rfolupdate(veh, veh.rfol, timeind)
            lleadupdate(veh, veh.llead, dummy_vec, timeind)
            rleadupdate(veh, veh.rlead, dummy_vec, timeind)
            leadupdate(veh, veh.lead, dummy_vec, timeind)
            folupdate(veh, veh.fol, timeind)
            remove_list.append(veh)

    return remove_list

def lfolupdate(removed_veh, next_veh, timeind):
    if type(next_veh) is not LeadVehicle and next_veh is not None and next_veh.rlead is removed_veh:
        next_veh.rlead = None
        next_veh.rleadmem.append([None, timeind+1])
        lfolupdate(removed_veh, next_veh.fol, timeind)

def rfolupdate(removed_veh, next_veh, timeind):
    if type(next_veh) is not LeadVehicle and next_veh is not None and next_veh.llead is removed_veh:
        next_veh.llead = None
        next_veh.lleadmem.append([None, timeind+1])
        rfolupdate(removed_veh, next_veh.fol, timeind)

def folupdate(removed_veh, next_veh, timeind):
    if type(next_veh) is not LeadVehicle and next_veh is not None:
        next_veh.lead = None
        next_veh.leadmem.append([None, timeind+1])

def lleadupdate(removed_veh, next_veh, dummy_vec, timeind):
    if type(next_veh) is not LeadVehicle and next_veh is not None and next_veh.rfol is removed_veh:
        next_veh.rfol = dummy_vec
        next_veh.rfolmem.append([None, timeind+1])
        lleadupdate(removed_veh, next_veh.lead, dummy_vec, timeind)

def rleadupdate(removed_veh, next_veh, dummy_vec, timeind):
    if type(next_veh) is not LeadVehicle and next_veh is not None and next_veh.lfol is removed_veh:
        next_veh.lfol = dummy_vec
        next_veh.lfolmem.append([None, timeind+1])
        rleadupdate(removed_veh, next_veh.lead, dummy_vec, timeind)

def leadupdate(removed_veh, next_veh, dummy_vec, timeind):
    if type(next_veh) is not LeadVehicle and next_veh is not None:
        next_veh.fol = dummy_vec
        next_veh.folmem.append([None, timeind+1])


class Calibration(CalibrationCF):
    def __init__(self, vehicles, leadvehicles, add_events, lc_events, dt, lane_dict, end=None, run_type='max endtime',
                 parameter_dict=None, ending_position=math.inf):
        super().__init__(vehicles, add_events, lc_events, dt, end=end, run_type=run_type,
                         parameter_dict=parameter_dict, ending_position=1475)
        self.all_leadvehicles = leadvehicles
        self.leadvehicles = set()
        self.dummy_vec = LeadVehicle([], 0)
        self.dummy_vec.pos = 1
        self.dummy_vec.acc = 1
        self.dummy_vec.speed = 1
        self.dummy_vec.len = 0


    def step(self):



        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)
        for veh in self.vehicles:
            veh.set_lc(self.timeind, self.dt)

        # for veh in self.vehicles:
        #     print(veh)
        #     print(veh.pos)
        #     print(veh.acc)
        #     print(veh.speed)
        # print(self.timeind)
        # print(self.ending_position)
        # print('-------------------------')


        # we need to have seperate add events and lc events, but the order of updates is exactly the same.
        update_calibration(self.vehicles, self.leadvehicles, self.update_lc_events, self.update_add_events, self.timeind, self.dt, self.ending_position, self.dummy_vec)
        # only difference is that when we call veh.update for veh in vehicles, we also need to call
        # veh.update for veh in leadvehicles. Suggest to just write a new update_calibration_lc as it is only
        # like 15 lines of code.

        self.timeind += 1

    def initialize(self, parameters):

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
        for veh in self.all_leadvehicles:
            # do we need seprate parameters for leadvehicles?
            veh.initialize(parameters)
        # initialize events, add the first vehicle(s), initialize vehicles headways/LeadVehicles
        self.vehicles = set()
        self.add_events = self.all_add_events.copy()
        self.lc_events = self.all_lc_events.copy()
        # had to change these from + 1?
        self.addtime = self.add_events[-1][0]
        self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
        self.timeind = self.start
        self.update_add_events(self.timeind-1, self.dt)
        self.update_lc_events(self.timeind-1, self.dt)



    def simulate(self, parameters):
        self.initialize(parameters)
        # do we simulate until we done with all vehicles? No max_endtime?
        while (self.vehicles or self.add_events) and self.timeind < 30000:
            self.step()

        if self.timeind == 30000:
            print("THIS IS A PROBLEM")
        else:
            print("THIS ONE IS FINE")

        loss = 0
        for veh in self.all_vehicles:
            loss += veh.loss()
        return loss


    def update_lc_events(self, timeind, dt):
        """Check if we need to apply the next lc event, apply it and update lctime if so.

        See function apply_calibration_lc_event.
        """
        if self.lctime == timeind+1:
            apply_calibration_lc_event(self.lc_events.pop(), timeind, dt)
            self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
            self.update_lc_events(timeind, dt)


    def update_add_events(self, timeind, dt):
        """Check if we need to apply the next add event, apply it and update addtime if so.

        See function apply_calibration_add_event.
        """
        if self.addtime == timeind+1:
            apply_calibration_add_event(self.add_events.pop(), self.vehicles, self.leadvehicles, timeind, dt,
                                          apply_calibration_lc_event)
            self.addtime = self.add_events[-1][0] if len(self.add_events)>0 else math.inf
            self.update_add_events(timeind, dt)


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


def apply_calibration_add_event(event, vehicles, leadvehicles, timeind, dt, lc_event_fun):
    if not event[3]:
        unused, fol_lead_veh, is_add, unused = event
        if is_add:
            leadvehicles.add(fol_lead_veh)
        else:
            leadvehicles.remove(fol_lead_veh)
    else:
        unused, curveh, unused, unused, lcevent = event
        curveh.r_lc = "discretionary"
        curveh.l_lc = "discretionary"
        curveh.update_lc_state(timeind, None)
        vehicles.add(curveh)
        lc_event_fun(lcevent, timeind, dt)



def apply_calibrationcf_add_event(event, vehicles, timeind, dt, lc_event_fun):
    """Adds a vehicle to the simulation and applies the first lead change event.

    Add events occur when a vehicle is added to the CalibrationCF.
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
        lc_event_fun: function which applies lc events
    """
    unused, unused, curveh, lcevent = event
    vehicles.add(curveh)
    lc_event_fun(lcevent, timeind, dt)
    if curveh.in_leadveh:
        curveh.leadveh.update(timeind+1)
    if curveh.lead is not None:
        curveh.hd = get_headway(curveh, curveh.lead)

def apply_calibration_lc_event(event, timeind, dt):
    # Apply relaxation if leader changes, apply relax.
    if len(event) == 5:
        start, curveh, r_lc, l_lc, lc = event
        curveh.r_lc = r_lc
        curveh.l_lc = l_lc
        curveh.update_lc_state(timeind, lc)
        # problem with set_relax
        curveh.set_relax(timeind, dt)

    else:
        start, curveh, fol_lead_veh, fl_type = event
        setattr(curveh, fl_type, fol_lead_veh)
        vehmem = fl_type + 'mem'
        vehmem = getattr(curveh, vehmem)
        vehmem.append([fol_lead_veh, timeind + 1])



def apply_calibrationcf_lc_event(event, timeind, dt):
    """Applies lead change event, updating a CalibrationVehicle's leader.

    Lead change events occur when a CalibrationVehicle's leader changes. In a CalibrationCF, it is
    assumed that vehicles have fixed lc times and fixed vehicle orders.
    Besides updating the leader, this also possibly applies relaxation, and possibly updates the follower
    attribute.
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
        update_lead_calibrationcf(curveh, newlead, leadlen, timeind)  # update leader


def update_lead_calibrationcf(curveh, newlead, leadlen, timeind):
    """Updates leader for curveh. Possibly updates in_leadveh and fol attributes.

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
