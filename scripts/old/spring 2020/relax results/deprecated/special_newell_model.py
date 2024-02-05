"""
Implementation of the Modified Newell Model (for calibration) with Relaxation proposed in
Laval, Leclerq (2008). Only intended to be used for single vehicle calibration.
"""
#Implementation details -
# because it needs both the current as well as next speed of the leader, there is the LLLeadVehicle.
# special formulation of relaxation means that lc_events are different - this means make_calibration makes
# the new events, lc_event handles the new event, and update_lc_event, update_add_event, add_event all need
# to call the correct lc_event.
# There is a special LLCalibration to ensure that the DeltaN variable is exactly updated correctly.
import numpy as np
from havsim.simulation.simulation import get_headway
import havsim.calibration.helper as helper
import math
import havsim.simulation.calibration as hc

class LLRelaxVehicle(hc.CalibrationVehicle):
    def __init__(self, vehid, y, initpos, initspd, inittime, leadstatemem, leadinittime, length=3,
                 accbounds=None, maxspeed=1e4, hdbounds=None, eql_type='v'):
        self.vehid = vehid
        self.len = length
        self.y = y
        self.initpos = initpos
        self.initspd = initspd
        self.inittime = inittime

        self.road = None
        self.lane = None

        if accbounds is None:
            self.minacc, self.maxacc = -7, 3
        else:
            self.minacc, self.maxacc = accbounds[0], accbounds[1]
        self.maxspeed = maxspeed
        self.hdbounds = (0, 1e4) if hdbounds is None else hdbounds
        self.eql_type = eql_type

        if leadstatemem is not None:
            self.leadveh = hc.LeadVehicle(leadstatemem, leadinittime)
        self.in_leadveh = False

    def cf_model(self, p, state):
        #p = space shift (=1/\kappa), wave speed (\omega), max speed, \epsilon
        #state = [self position, lead position-lead.len, lead speed, dt, DeltaN]
        K = (p[1]/p[0])/(p[1]+state[1])
        return (state[0] + state[1]*state[2] - state[3]/K - state[4])/state[2]  # calculate update

    def set_cf(self, timeind, dt):
        self.speed = self.cf_model(self.cf_parameters, [self.lead.pos-self.lead.len, self.lead.speed, dt, self.DeltaN,
                                                        self.pos])

    def set_relax(self, relaxamounts, timeind, dt):
        """Applies relaxation given the relaxation amounts."""
        if True:
            self.DeltaN = relaxamounts
            self.in_relax = True
            self.first_index = True

    def update(self, timeind, dt):
        curspeed = self.speed
        if curspeed < 0:
            curspeed = 0
        elif curspeed > self.maxspeed:
            curspeed = self.maxspeed
        # update state
        self.pos += curspeed*dt
        self.speed = curspeed
        # update memory
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        self.DeltaNmem.append(self.DeltaN)

        if self.in_leadveh:
            self.leadveh.update(timeind+1)

    def loss(self):
        """Calculates loss."""
        return sum(np.square(np.array(self.posmem) - self.y))/len(self.posmem)

    def initialize(self, parameters):
        super().initialize(parameters)  # before the first cf call, the speed is initialized as initspd.
        # this handles the edge case for if a vehicle tries to access the speed before the first cf call.
        # after the first cf call, in this case the speed will simply be the speed from the previous timestep
        self.cf_parameters = parameters
        self.speedmem = []  # note that speedmem will be 1 len shorter than posmem for a 1st order model
        self.maxspeed = parameters[2]
        self.DeltaNmem = []
        self.DeltaN = 1
        self.in_relax = False
        self.first_index = False


# class LLLeadVehicle:
#     """Lead Vehicle which has access to the speed from the next timestep."""
#     def __init__(self, leadstatemem, inittime):
#         """
#         leadstatemem - list of tuples, each tuple is a pair of (position, speed)
#         inittime - leadstatemem[0] corresponds to time inittime
#         """
#         self.leadstatemem = leadstatemem
#         self.inittime = inittime
#         self.road = None

#     def update(self, timeind, *args):
#         """Update position/speed."""
#         self.pos, self.speed = self.leadstatemem[timeind - self.inittime]
#         try:
#             self.nextspeed = self.leadstatemem[timeind-self.inittime+1][1]
#         except:
#             self.nextspeed = self.leadstatemem[timeind-self.inittime][1]

#     def set_len(self, length):
#         """Set len so headway can be computed correctly."""
#         self.len = length


class LLCalibration(hc.Calibration):
    """Does a simulation of a single LLRelaxVehicle, and returns the loss."""

    def step(self):
        """Has a different order of updates than Calibration"""
        self.lctime = update_lc_event(self.lc_events, self.lctime, self.timeind, self.dt)

        for veh in self.vehicles:  # assume 1 vehicle at a time
        # update for \Delta N
            if veh.in_relax:
                if veh.first_index:
                    veh.first_index = False
                else:
                    p = veh.cf_parameters
                    vtilde = veh.lead.nextspeed*(1-veh.DeltaN)+veh.lead.speed*veh.DeltaN-p[3]
                    Kvj = (p[1]/p[0])/(p[1]+veh.lead.speed)
                    Kvjp1 = (p[1]/p[0])/(p[1]+veh.lead.nextspeed)
                    veh.DeltaN = (veh.DeltaN/Kvj + (veh.lead.nextspeed-vtilde)*self.dt)*Kvjp1

            veh.set_cf(self.timeind, self.dt)
            veh.update(self.timeind, self.dt)
            veh.hd = get_headway(veh, veh.lead)

        self.timeind += 1

    def simulate(self, parameters):
        """The same, but we need to put it here so it calls the right event_updates."""
        # reset all vehicles, and assign their parameters
        for veh in self.all_vehicles:
            veh.initialize(parameters)

        # initialize events, add the first vehicle(s), initialize vehicles headways/LeadVehicles
        self.vehicles = set()
        self.add_events = self.all_add_events.copy()
        self.lc_events = self.all_lc_events.copy()
        self.addtime = self.add_events[-1][0]
        self.lctime = self.lc_events[-1][0] if len(self.lc_events)>0 else math.inf
        self.timeind = self.starttime
        self.addtime = update_add_event(self.vehicles, self.add_events, self.addtime, self.timeind-1, self.dt)
        for veh in self.vehicles:
            if veh.in_leadveh:
                veh.leadveh.update(self.timeind)
            veh.hd = get_headway(veh, veh.lead)


        # do simulation by calling step repeatedly
        for i in range(self.endtime - self.starttime):
            self.step()

        # calculate and return loss
        loss = 0
        for veh in self.all_vehicles:
            loss += veh.loss()

        return loss

def make_calibration(vehicles, meas, platooninfo, dt):
    """Sets up a Calibration object.

    Extracts the relevant quantities (e.g. LeadVehicle, initial conditions, loss) from the data
    and creates the add/lc event.

    Args:
        vehicles: list/set of vehicles to add to the Calibration
        meas: from havsim.calibration.algs.makeplatoonlist
        platooninfo: from havsim.calibration.algs.makeplatoonlist
        dt: timestep
        vehicle_class: subclassed Vehicle to use
    """
    vehicle_list = []
    addevent_list = []
    lcevent_list = []
    id2obj = {}
    if type(vehicles) == list:
        vehicles = set(vehicles)

    for veh in vehicles:
        # make lead memory - a list of position/speed tuples for any leaders which are not simulated
        leads = set(platooninfo[veh][4])
        needleads = leads.difference(vehicles)
        if len(needleads) > 0:
            leadstatemem = []
            leadinittime = None  # initial time lead is used
            leadinfo = helper.makeleadinfo([veh],platooninfo,meas)
            for j in leadinfo[0]:
                curlead, start, end = j
                if curlead in needleads:
                    if not leadinittime:
                        leadinittime = start
                    leadt_n = platooninfo[curlead][0]
                    poslist = list(meas[curlead][start-leadt_n:end-leadt_n+1,2])
                    spdlist = list(meas[curlead][start-leadt_n:end-leadt_n+1,3])
                    leadstatemem.extend(zip(poslist,spdlist))
                elif leadinittime is not None:
                    temp = [0]*(end-start+1)
                    leadstatemem.extend(temp)
        else:
            leadstatemem, leadinittime = None, None

        # get initial values, y for veh
        t_nstar, inittime, endtime = platooninfo[veh][0:3]
        initlead, initpos, initspd, length = meas[veh][inittime-t_nstar,[4,2,3,6]]
        y = meas[veh][inittime-t_nstar:endtime+1-t_nstar,2]

        # create vehicle object
        newveh = LLRelaxVehicle(veh, y, initpos, initspd, inittime, leadstatemem, leadinittime,
                                    length=length)
        vehicle_list.append(newveh)
        id2obj[veh] = newveh

    # create events
    for veh in vehicles:
        curveh = id2obj[veh]
        leadinfo = helper.makeleadinfo([veh],platooninfo,meas)[0]
        t_nstar, t_n = platooninfo[veh][:2]
        for count, j in enumerate(leadinfo):
            curlead, start, end = j
            # make lc_event - supports single vehicle simulation only
            curlen = meas[curlead][0,6]
            if start > t_n or (t_n > t_nstar and count == 0):  # LC or merge
                prevlane, newlane = meas[veh][start-t_nstar-1, 7], meas[veh][start-t_nstar,7]
                if newlane != prevlane:  # veh = 'i' vehicle in LL notation
                    userelax = True
                    ip1 = False
                    x1veh, x2veh = meas[veh][start-t_nstar,4], meas[veh][start-t_nstar,5]
                    if x2veh == 0:  # handle edge case
                        # print('warning - follower missing for '+str(veh)+' change at time '+str(start))
                        userelax = False
                        x1, x2 = None, None
                    else:
                        x1t_nstar, x2t_nstar = platooninfo[x1veh][0], platooninfo[x2veh][0]
                        x1, x2 = meas[x1veh][start-x1t_nstar,2], meas[x2veh][start-x2t_nstar,2]
                else:
                    temp = meas[veh][start-t_nstar,4]
                    tempt_nstar = platooninfo[temp][0]
                    if meas[temp][start-1-tempt_nstar,7] != newlane:  # veh = 'i+1' in LL notation
                        userelax = True
                        ip1 = True
                        x1veh, x2veh = meas[veh][start-1-t_nstar,4], meas[veh][start-t_nstar,4]
                        x1t_nstar, x2t_nstar = platooninfo[x1veh][0], platooninfo[x2veh][0]
                        x1, x2 = meas[x1veh][start-x1t_nstar,2], meas[x2veh][start-x2t_nstar,2]
                    else:
                        userelax = False
                        x1, x2, ip1 = None, None, None
            else:
                userelax = False  # no relax
                x1, x2, ip1 = None, None, None

            leadstate = (x1, x2, ip1)
            curevent = (start, 'lc', curveh, curlead, curlen, userelax, leadstate)

            if count == 0:
                curevent = (start, 'add', curveh, curevent)
                addevent_list.append(curevent)
            else:
                lcevent_list.append(curevent)

    addevent_list.sort(key = lambda x: x[0], reverse = True)  # sort events in time
    lcevent_list.sort(key = lambda x: x[0], reverse = True)

    # make calibration object
    return LLCalibration(vehicle_list, addevent_list, lcevent_list, dt, endtime=endtime)


def update_lc_event(lc_events, lctime, timeind, dt):
    """Check if we need to apply the next lc event, apply it and update lctime if so.

    Args:
        lc_events: sorted list of current lead change events
        lctime: next time a lead change event occurs
        timeind: time index
        dt: timestep
    """
    if lctime == timeind:
        lc_event(lc_events.pop(), timeind, dt)
        lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
        if lctime == timeind+1:
            while lctime == timeind+1:
                lc_event(lc_events.pop(), timeind, dt)
                lctime = lc_events[-1][0] if len(lc_events)>0 else math.inf
    return lctime


def update_add_event(vehicles, add_events, addtime, timeind, dt):
    """Check if we need to apply the next add event, apply it and update addtime if so.

    Args:
        add_events: sorted list of current add events
        addtime: next time an add event occurs
        timeind: time index
        dt: timestep
    """
    if addtime == timeind+1:
        add_event(add_events.pop(), vehicles, timeind, dt)
        addtime = add_events[-1][0] if len(add_events)>0 else math.inf
        if addtime == timeind+1:
            while addtime == timeind+1:
                add_event(add_events.pop(), vehicles, timeind, dt)
                addtime = add_events[-1][0] if len(add_events)>0 else math.inf
    return addtime

def add_event(event, vehicles, timeind, dt):
    """Adds a vehicle to the simulation and applies the first lead change event.

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


def lc_event(event, timeind, dt):
    """Applies lead change event, updating a CalibrationVehicle's leader.

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
        x1, x2, ip1 = leadstate
        if ip1:  # vehicle is i+1 -> x1 = xi-1, x2 = xi
            relaxamount = (x2 - curveh.pos)/(x1 - curveh.pos)
        else:  # vehicle is i -> x1 = xi-1, x2 = i+1
            relaxamount = (x1 - curveh.pos)/(x1 - x2)
        curveh.set_relax(max(relaxamount,.01), timeind, dt)

    hc.update_lead(curveh, newlead, leadlen, timeind)  # update leader