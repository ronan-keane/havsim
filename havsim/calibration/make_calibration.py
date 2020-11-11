""" Making calibration object and event functions"""
import numpy as np
from havsim.simulation.road_networks import get_headway
from havsim.calibration.vehicles import LeadVehicle, CalibrationVehicle
from havsim.calibration.calibration import CalibrationCF, Calibration
import math

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
        (vehdict[veh].rfolmem.intervals(t0, t1), "rl"),
        ]

        # add/lc events for lead veh relationships
        for tup_index in range(len(info_list)):
            tup = info_list[tup_index]
            info, fl_type = tup[0], tup[1]
            if info == 0:
                create_add_events(info, id2obj, curveh, vehdict, vehicles, dt, addevent_list, lcevent_list, all_leadvehicles, fl_type, True)
            else:
                create_add_events(info, id2obj, curveh, vehdict, vehicles, dt, addevent_list, lcevent_list, all_leadvehicles, fl_type)

        #lc events for CalibrationVeh
        lc_intervals = vehdict[veh].lanemem.intervals(t0,t1)
        if len(lc_intervals) > 1:
            for i in range(1,len(lc_intervals)):
                last_lane = lc_intervals[i-1][0]
                start = lc_intervals[i][1]
                new_lane = lc_intervals[i][0]
                lc_event = (start, 'lc', curveh, last_lane, new_lane, "currveh_lc")
                print(lc_event)
                lcevent_list.append(lc_event)


    #Remove and add events for lead_vehicles
    for leadveh_id in all_leadvehicles:
        start = all_leadvehicles[leadveh_id][0]
        end = all_leadvehicles[leadveh_id][1]
        fol_lead_veh = id2obj[leadveh_id]

        curevent = (start, 'add', fol_lead_veh)
        addevent_list.append(curevent)

        curevent = (end, 'remove', fol_lead_veh)
        addevent_list.append(curevent)


    return addevent_list, lcevent_list


def make_leadvehicles(vehicles, id2obj, vehdict, dt):
    # find all LeadVehicles and their start/end times
    all_leadvehicles = {}  # keys are LeadVehicles id, values are tuples of (start, end)
    vehmem_list = ['leadmem', 'lleadmem', 'rleadmem', 'folmem', 'lfolmem', 'rfolmem']
    for veh in vehicles:
        for curvehmem in vehmem_list:
            vehmem = getattr(vehdict[veh], curvehmem)
            for i in vehmem.intervals():
                curveh, curstart, curend = i
                if curveh:
                    if curveh in all_leadvehicles:
                        start, end = all_leadvehicles[curveh]
                        all_leadvehicles[curveh] = (min(curstart, start), max(curend, end))
                    else:
                        all_leadvehicles[curveh] = (curstart, curend)

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

        if not fol_lead_veh:
            # if we are looking at followers
            if fl_type == "f" or fl_type == "lf" or fl_type == "rf":
                # how should we reuse this for all cases?
                dummy_vec = LeadVehicle([], 0)
                curevent = (start, 'lc', curveh, dummy_vec, fl_type)
                lcevent_list.append(curevent)
            # if we are looking at leader relationships
            else:
                fol_lead_veh = None
                sorted_lc_event = sorted(lcevent_list, key = lambda x:x[0], reverse=True)
                for event in sorted_lc_event:
                    if event[1] == fl_type:
                        if not isinstance(event[2], CalibrationVehicle):
                            curevent = (start, "lc", curveh, fol_lead_veh, fl_type)
                            lcevent_list.append(curevent)
                            break

        else:
            fol_lead_veh, curlen = id2obj[fol_lead_veh], None


            if count == 0:
                # lc event for first tims
                curevent = (start, 'lc', curveh, fol_lead_veh, fl_type)
                # only needed for the first time to create the add cur_veh event
                if first_call:
                    curevent = (start, 'add', curveh, curevent)
                addevent_list.append(curevent)

            else:
                # lc event for fol_lead_veh with respect to curveh
                curevent = (start, 'lc', curveh, fol_lead_veh, fl_type)
                lcevent_list.append(curevent)

# TODO Refactor this version into seperate functions for Calibration/CalibrationCF
# put the make_calibration/make_event functions into a seperate file (make_calibration.py)
def make_calibration(vehicles, vehdict, dt, event_maker=None, lc_event_fun=None, lanes={}, calibration_kwargs={}):

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


        newveh = CalibrationVehicle(veh, y, y_lc, initpos, initspd, t0, length=length, lane=lanes[lane])

        vehicle_list.append(newveh)
        id2obj[veh] = newveh
        max_end = max(max_end, t1)

    # create events
    all_leadvehicles = make_leadvehicles(vehicles, id2obj, vehdict, dt)
    addevent_list, lcevent_list = event_maker(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list, all_leadvehicles)

    addevent_list.sort(key = lambda x: x[0], reverse = True)
    lcevent_list.sort(key = lambda x: x[0], reverse = True)

    return Calibration(vehicle_list, all_leadvehicles, addevent_list, lcevent_list, dt, end=max_end)


def make_calibration_CF(vehicles, vehdict, dt, vehicle_class=None, calibration_class=None, event_maker=None, lc_event_fun=None, lanes={}, calibration_kwargs={}):

    """Sets up a CalibrationCF object.

    Extracts the relevant quantities (e.g. LeadVehicle, initial conditions, loss) from the data
    and creates the add/lc event.

    Args:
        vehicles: list of vehicles to add to the CalibrationCF
        vehdict: dictionary of all VehicleData
        dt: timestep
        vehicle_class: subclassed Vehicle to use - if None defaults to CalibrationVehicle
        calibration_class: subclassed CalibrationCF to use - if None defaults to CalibrationCF
        event_maker: specify a function to create custom (lc) events
        lc_event_fun: specify function which handles custom lc events
        lanes: dictionary with keys as lane indexes, values are Lane objects with call_downstream method.
            Used for downstream boundary.
        calibration_kwargs: keyword arguments passed to CalibrationCF
    """

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
        initpos, initspd = vehdata.posmem[t0], vehdata.speedmem[t0]
        length, lane = vehdata.len, vehdata.lanemem[t1]

        needleads = set(vehdata.leads).difference(vehicles)
        # build the leadstatemem in all times [t0, t1], even if it is only needed for a portion of the times.
        if len(needleads)>0:
            leadstatemem = list(zip(vehdata.leadmem.pos[t0:t1+1], vehdata.leadmem.speed[t0:t1+1]))
            leadstart = t0
        else:
            leadstatemem = leadstart = 0

        newveh = CalibrationVehicleCF(veh, y, initpos, initspd, t0, leadstatemem, leadstart, length=length,
                               lane=lanes[lane])

        vehicle_list.append(newveh)
        id2obj[veh] = newveh
        max_end = max(max_end, t1)

    addevent_list, lcevent_list = event_maker(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list)

    addevent_list.sort(key = lambda x: x[0], reverse = True)
    lcevent_list.sort(key = lambda x: x[0], reverse = True)

    return CalibrationCF(vehicle_list, addevent_list, lcevent_list, dt, end=max_end,
                             lc_event_fun=lc_event_fun, **calibration_kwargs)


def make_lc_event(vehicles, id2obj, vehdict, dt, addevent_list, lcevent_list):
    """Makes lc and add events for default CalibrationCF, which includes adding relaxation."""
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
