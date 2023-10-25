
"""
Functions for setting/updating lane events, route events, and code for changing lanes.
"""
from havsim.simulation import vehicle_orders
from havsim.simulation.road_networks import get_headway


def update_veh_after_lc(lc_actions, veh, timeind):
    """When a vehicle changes lanes, this function does all the necessary updates.

    When a vehicle changes lanes, we need to update it's lane, road, llane/rlane, r/l_lc, lanemem,
    and the lane changing model internal state.
    More importantly, we need to update all the leader/follower relationships.
    ***Naming conventions***
    Every vehicle has its leader (lead) and follower (fol). Putting l or r in front of lead/fol indicates
    that it is the left/right leader/follower. Consider some vehicle, the 'ego vehicle'. The ego vehicle's
    lfol is the vehicle in the left lane closest to the ego vehicle, without going past the position of the
    ego vehicle. llead has two possible meanings. The llead attribute is the set of all vehicles which
    have the ego vehicle as a rfol. In the context of a lane changing model, we use llead to refer to the
    leader of lfol. Note that the leader of lfol is not even necessarily in the set which defines
    the llead attribute.
    The same definitions apply to rfol and rlead as to lfol and llead.
    The other naming conventions are lcside, newlcside, and opside. If a vehicle changes to the left,
    lcside (lane change side) refers to the left lane, the opside (opposite lane change side) refers to
    the right lane. The newlcside (new lane change side) is the new lcside after changing lanes, so if the
    side is left, it refers to two lanes to the left.
    Note in this case we are using 'new' to refer to the situation after the lane change. This is another
    convention used for lane changing models.

    Args:
        lc_actions: dictionary with keys as vehicles which request lane changes in the current timestep,
            values are a string either 'l' or 'r' which indicates the side of the change
        veh: Vehicle object which changes lanes, and has a key/value in lc_actions
        timeind: int giving the timestep of the simulation (0 indexed)

    Returns:
        None. Modifies veh, and all vehicles which have a relationship with veh, in place.
    """
    lc = lc_actions[veh]
    # updates to lanes, road
    lcsidelane, newlcsidelane, lc = update_lane_after_lc(veh, lc, timeind+1)
    veh.update_lc_state(timeind, lc=True)

    # updates to vehicle orders
    vehicle_orders.update_leadfol_after_lc(veh, lcsidelane, newlcsidelane, lc, timeind)

    return


def update_lane_after_lc(veh, lc, timeind):
    """After a lane change, this updates the lane, road, pos, lanemem, r/llane, l/r_lc attributes.

    Args:
        veh: Vehicle object to update.
        lc: side of lane change; either 'l' or 'r'
        timeind: time index of change (+1 higher than the current time index in simulation)

    Returns:
        lcsidelane, newlcsidelane, lc
    """
    if lc == 'l':
        veh.rlane = veh.lane
        lcsidelane = veh.llane
        newroadname = lcsidelane.roadname
        if newroadname != veh.road:
            veh.pos -= veh.lane.roadlen[newroadname]
            veh.road = newroadname
            veh.r_lc = None
        else:
            veh.r_lc = 'd'
        veh.lane = lcsidelane
        veh.lanemem.append((lcsidelane, timeind))
        newlcsidelane = lcsidelane.get_connect_left(veh.pos)
        veh.llane = newlcsidelane
        if newlcsidelane is not None and newlcsidelane.roadname == veh.road:
            veh.l_lc = 'd'
        else:
            veh.l_lc = None
    else:
        veh.llane = veh.lane
        lcsidelane = veh.rlane
        newroadname = lcsidelane.roadname
        if newroadname != veh.road:
            veh.pos -= veh.lane.roadlen[newroadname]
            veh.road = newroadname
            veh.l_lc = None
        else:
            veh.l_lc = 'd'
        veh.lane = lcsidelane
        veh.lanemem.append((lcsidelane, timeind))
        newlcsidelane = lcsidelane.get_connect_right(veh.pos)
        veh.rlane = newlcsidelane
        if newlcsidelane is not None and newlcsidelane.roadname == veh.road:
            veh.r_lc = 'd'
        else:
            veh.r_lc = None

    return lcsidelane, newlcsidelane, lc


def update_lane_events(veh, timeind, remove_vehicles):
    """Check if the next event from a Vehicle's lane_events should be applied, and apply it if so.

    lane_events are a list of events which handle anything related to the network topology,
    i.e. when the current lane ends, or when the current lane's left or right connections change.
    Each event is a dictionary with the keys of
    'pos': the float position the event occurs (relative to the vehicle's current lane)
    'event': one of
        'new lane' - occurs when a vehicle reaches the end of its current lane and transitions to a new lane
        'update lr' - occurs when the current lane's left or right connections change
        'exit' - occurs when a vehicle reaches the end of its current lane and exits the road network
    'left': for 'new lane' or 'update lr', if the left connection changes, 'left' has a value of either
        'add' - Used when the current left lane changes from None to a new lane, or when the current left
            lane changes tracks.
        'remove' -  if the current left connection is no longer possible
        'update' - if there is still a left lane in the same track, but now it refers to a new lane object
    'left anchor': if 'left' is 'add', 'left anchor' is an index giving the merge anchor for the
        new left lane
    'right': same as left, for right side
    'right anchor': same as left anchor, for right side

    Args:
        veh: Vehicle object to update
        timeind: int giving the timestep of the simulation (0 indexed)
        remove_vehicles: set of vehicles which will be removed from simulation at current timestep

    Returns:
        None. (Modifies Vehicle attributes in place, adds to remove_vehicles in place.)
    """
    if not veh.lane_events:
        return
    curevent = veh.lane_events[0]
    if veh.pos > curevent['pos']:
        if curevent['event'] == 'new lane':
            # update lane/road/position
            newlane = veh.lane.connect_to
            update_new_lane(veh, veh.lane, newlane, timeind+1)

            # updates left and right connections
            update_lane_lr(veh, newlane, curevent)
            veh.update_lc_state(timeind)

            # enter new road/lane -> need new lane/route events
            set_lane_events(veh)
            set_route_events(veh, timeind)

        elif curevent['event'] == 'update lr':
            update_lane_lr(veh, veh.lane, curevent)
            veh.update_lc_state(timeind)
            veh.lane_events.pop(0)  # event is over, so we shouldn't check it in the future

        elif curevent['event'] == 'exit':
            fol = veh.fol
            for i in veh.llead:
                i.rfol = fol
                fol.llead.append(i)
            for i in veh.rlead:
                i.lfol = fol
                fol.rlead.append(i)

            # update vehicle orders
            lead = veh.lead
            fol.lead = lead
            fol.leadmem.append((lead, timeind+1))
            if lead is not None:
                lead.fol = fol
                fol.hd = get_headway(fol, lead)
            if veh.lfol is not None:
                veh.lfol.rlead.remove(veh)
            if veh.rfol is not None:
                veh.rfol.llead.remove(veh)

            # to remove the vehicle set its end and put it in the remove_vehicles
            veh.end = timeind+1
            remove_vehicles.append(veh)
    return


def update_lane_lr(veh, curlane, curevent):
    """Updates a vehicle's attributes when its lane changes its left/right connections.

    For a Vehicle veh which reaches a point where its curlane.get_connect_left or get_connect_right
    go from None to some Lane, or some Lane to None, there needs to be 'add' or 'remove' events for the
    corresponding sides. This handles those events.
    Updates the vehicle orders and defaults the lane change states to the correct behavior (by default,
    enter discretionary only if the left/right lane is in the same road as the current lane).
    This updates the left/right followers, l/r lane, and l/r_lc attributes.

    Args:
        veh: Vehicle object to update
        curlane: the Lane veh is currently on, curlane has the new/ending connections
        curevent: The event (dictionary) triggering the update

    Returns:
        None (Modifies veh attributes in place.)
    """
    if curevent['left'] == 'remove':
        # update lead/fol order
        veh.lfol.rlead.remove(veh)
        veh.lfol = None
        veh.l_lc = None
        veh.llane = None

    elif curevent['left'] == 'add':
        newllane = curlane.get_connect_left(curevent['pos'])

        # get the new follower in the new track
        merge_anchor = newllane.merge_anchors[curevent['left anchor']][0]
        unused, newfol = curlane.leadfol_find(veh, merge_anchor, 'l')
        if veh.lfol is not None:
            veh.lfol.rlead.remove(veh)
        veh.lfol = newfol
        newfol.rlead.append(veh)

        if newllane.roadname == curlane.roadname:
            veh.l_lc = 'd'
        else:
            veh.l_lc = None
        veh.llane = newllane

    elif curevent['left'] == 'update':
        newllane = curlane.get_connect_left(curevent['pos'])

        if newllane.roadname == curlane.roadname:
            veh.l_lc = 'd'
        else:
            veh.l_lc = None
        veh.llane = newllane

    # same thing for right
    if curevent['right'] == 'remove':

        veh.rfol.llead.remove(veh)
        veh.rfol = None
        veh.r_lc = None
        veh.rlane = None

    elif curevent['right'] == 'add':
        newrlane = curlane.get_connect_right(curevent['pos'])

        merge_anchor = newrlane.merge_anchors[curevent['right anchor']][0]
        unused, newfol = curlane.leadfol_find(veh, merge_anchor, 'r')
        if veh.rfol is not None:
            veh.rfol.llead.remove(veh)
        veh.rfol = newfol
        newfol.llead.append(veh)

        if newrlane.roadname == curlane.roadname:
            veh.r_lc = 'd'
        else:
            veh.r_lc = None
        veh.rlane = newrlane

    elif curevent['right'] == 'update':
        newrlane = curlane.get_connect_right(curevent['pos'])

        if newrlane.roadname == curlane.roadname:
            veh.r_lc = 'd'
        else:
            veh.r_lc = None
        veh.rlane = newrlane


def update_new_lane(veh, oldlane, newlane, timeind):
    """When a vehicle enters a new lane, this updates the lane, road, pos, and lanemem attributes.

    Args:
        veh: Vehicle object to update.
        oldlane: current Lane veh is on.
        newlane: The new Lane the vehicle is changing to.
        timeind: int giving the timestep of the simulation (0 indexed)
    Returns:
        None. Modifies veh in place
    """
    newroadname = newlane.roadname
    if newroadname != veh.road:
        veh.pos -= oldlane.roadlen[newroadname]
        veh.road = newroadname
    veh.lane = newlane
    veh.lanemem.append((newlane, timeind))


def set_lane_events(veh):
    """Creates lane_events attribute for Vehicle after entering a new lane.

    Refer to update_lane_events for description of lane events. Note that we only need to add upcoming
    lane events, and past lane events are not applied (this is in contrast to route_events, where past
    events ARE applied.)

    Args:
        veh: Vehicle to be updated

    Returns:
        None (Modifies veh in place.)
    """
    veh.lane_events = []
    for i in veh.lane.events:
        if i['pos'] > veh.pos:
            veh.lane_events.append(i)


def update_route_events(veh, timeind):
    """Check if the next event from a vehicle's route_events should be applied, and apply it if so. Called recursively.

    route_events are a list of events which handles any lane changing behavior related to
    a vehicle's route, i.e. route events ensure that the vehicle follows its route.
    Each event is a dictionary with the keys of
    'pos': the float position the event occurs (relative to the vehicle's current lane).
    'event': 'end discretionary' or 'mandatory', which end discretionary or start mandatory
        lane changing states
    'side': 'l_lc' or 'r_lc' the side which is updated by the event
    'lc_urgency': only for a 'mandatory' event, a tuple giving the position for 0% and 100% forced cooperation
    'endpos': only for a 'mandatory' event, last float position to complete the change

    Args:
        veh: Vehicle object to update

    Returns:
        None.
    """
    if not veh.route_events:
        return
    curevent = veh.route_events[0]
    if veh.pos > curevent['pos']:

        if curevent['event'] == 'end discretionary':
            side = curevent['side']
            setattr(veh, side, None)
            veh.update_lc_state(timeind)

        elif curevent['event'] == 'mandatory':
            if veh.pos > curevent['endpos']:  # case where the mandatory LC can no longer be completed
                # just reset to default state. The route will not be followed.
                if veh.llane is not None:
                    veh.l_lc = 'd' if veh.llane.roadname == veh.road else None
                if veh.rlane is not None:
                    veh.r_lc = 'd' if veh.rlane.roadname == veh.road else None
            else:  # normal update
                side = curevent['side']
                setattr(veh, side, 'm')
                veh.lc_urgency = curevent['lc_urgency']  # must always set urgency for mandatory changes

            veh.update_lc_state(timeind)

        veh.route_events.pop(0)
        update_route_events(veh, timeind)
    return


def make_cur_route(p, curlane, nextroadname):
    """Creates cur_route attribute (stores route events) for Vehicle after entering a new lane.

    Refer to update_route_events for a description of route events.
    Upon entering a new road, we create a cur_route which stores the list of route events for several lanes,
    specifically the lanes we will ultimately end up on, as well as all lanes which we will need to cross
    to reach those lanes we want to be on. We do not create the routes for every single lane on a road.
    Roads have a connections attribute whose keys are road names and value is a tuple of:
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

    Explanation of current route model -
    suppose you need to be in lane '2' by position 'x' and start in lane '1', then starting:
        at x - 2*p[0] - 2*p[1] you will end discretionary changing into lane '0'
        at x - p[0] - p[1] you wil begin mandatory changing into lane '2'
        at x - p[0] your mandatory change will have urgency of 100% which will always force
            cooperation of your l/rfol
    for lane changing with a merge/diverse (e.g. on/off-ramp) which begins at 'x' and ends at 'y',
    you will start mandatory at 'x' always, reaching 100% cooperation by 'y' - p[0]

    Args:
        p: parameters, length 2 list of floats, where p[0] controls how quickly a vehicle with mandatory LC can
            force cooperation of the lfol/rfol, larger is faster to force cooperation. p[0] + p[1] is a comfortable
            distance for completing a mandatory lane change
        curlane: Lane object to create route events for
        nextroadname: str name of the next road in the route (the next road you want to be on after leaving
            curlane's road)

    Returns:
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle with parameters p needs to take on that lane
    """
    # TODO we only get the route for the current road - no look ahead to take into account future roads

    curroad = curlane.road
    curlaneind = curlane.laneind
    pos, change_type, laneind, side, nextroad = curlane.connections[nextroadname][:]
    # if nextroadname in curlane.connections:
    #     pos, change_type, laneind, side, nextroad = curlane.connections[nextroadname][:]
    # else:
    #     print(' vehicle on ' + str(curlane) + ' missed route which planned for going to ' + nextroadname)
    #     return {i: [] for i in curroad.lanes}

    cur_route = {}
    if change_type == 'continue':  # -> vehicle needs to reach end of lane
        # initialize for lanes which vehicle needs to continue on
        leftind, rightind = laneind[:] if len(laneind) == 2 else (laneind[0], laneind[0])
        for i in range(leftind, rightind+1):
            cur_route[curroad[i]] = []

        if leftind > 0:
            templane = curroad[leftind]
            curpos = min(templane.end, curroad[leftind-1].end)
            cur_route[templane].append({'pos': curpos - 2*(p[0] + p[1]),
                                        'event': 'end discretionary', 'side': 'l_lc'})

        if rightind < curroad.num_lanes-1:
            templane = curroad[rightind]
            curpos = min(templane.end, curroad[rightind+1].end)
            cur_route[templane].append({'pos': curpos - 2*(p[0] + p[1]),
                                        'event': 'end discretionary', 'side': 'r_lc'})

        if leftind <= curlaneind <= rightind:  # if on correct lane already, do no more work
            return cur_route
        elif curlaneind < leftind:  # need to change right possibly multiple times
            uselaneind = leftind
        else:
            uselaneind = rightind

        pos = min(curroad[uselaneind].end, pos)
        cur_route = make_route_helper(p, cur_route, curroad, curlaneind, uselaneind, pos, pos)

    elif change_type == 'merge':  # logic is similar and also uses make_route_helper
        templane = curroad[laneind]
        pos, endpos = pos[:]
        cur_route[templane] = []

        # determine end discretionary event if necessary
        if side == 'l_lc':
            if laneind < curroad.num_lanes-1:
                enddisc = min(pos, curroad[laneind+1].end)
                cur_route[templane].append({'pos': enddisc - 2*(p[0] + p[1]),
                                            'event': 'end discretionary', 'side': 'r_lc'})
        else:
            if laneind > 0:
                enddisc = min(pos, curroad[laneind-1].end)
                cur_route[templane].append({'pos': enddisc - 2*(p[0] + p[1]),
                                            'event': 'end discretionary', 'side': 'l_lc'})

        cur_route[templane].append({'pos': pos, 'event': 'mandatory', 'side': side,
                                    'lc_urgency': [pos, max(endpos - p[0], pos)],
                                    'endpos': endpos})

        if curlaneind != laneind:
            cur_route = make_route_helper(p, cur_route, curroad, curlaneind, laneind, pos, endpos)

    return cur_route


def make_route_helper(p, cur_route, curroad, curlaneind, laneind, curpos, endpos):
    """Generates list of route events for all lanes with indexes [curlaneind, laneind).

    Starting on curroad in lane with index curlaneind, wanting to be in lane index laneind by position curpos,
    generates route events for all lanes in [curlaneind, laneind). If curlaneind < laneind, starts at
    laneind -1, moving to the left until routes on all lanes are defined. Similarly for curlaneind > laneind.
    Assumes we already have the route for laneind in cur_route.
    Edge cases where lanes have different lengths are handled, but we assume that all lanes are connected
    when they exist. E.g. for a road with 2 lanes, lane0 and lane1, you could have:
    lane0.start = 0, lane0.end = 1000
    lane1.start = 500, lane1.end = 1500,
    lane0.connect_right = [(0, None), (lane1.start, lane1)]
    lane1.connect_left = [(lane1.start, lane0)]
    This configuration would work.
    But if:
    lane1.connect_left = [(lane1.start, None), (800, lane0)]
    his case is not currently handled, because the current code does not look at the connect_left/right,
    it just uses the .start, .end

    Args:
        p: parameters, length 2 list of floats, where p[0] is a safety buffer for merging and p[1]
            is a comfortable distance for merging
        cur_route: dictionary where keys are lanes, value is a list of route event dictionaries which
            defines the route a vehicle with parameters p needs to take on that lane
        curroad: road that the route is being generated for
        curlaneind: index of the lane that the vehicle starts in
        laneind: index of the lane that we want to be in by position curpos
        curpos: we want to be in lane with index laneind by curpos
        endpos: if in lane laneind by endpos, we've missed the route

    Returns:
        cur_route: Updates cur_route in place
    """
    nexttemplane = None
    if curlaneind < laneind:  # populate route events in lanes left of target
        curind = laneind - 1
        templane = curroad[curind]
        while not curind < curlaneind:
            cur_route[templane] = []
            # determine curpos, the latest position the veh wants to change by
            if templane.end < curpos:  # in case templane ends before the curpos
                curpos = templane.end
            curpos -= p[0] + p[1]

            # determine enddiscpos = where the discretionary ends
            # only necessary if there is something to end the discretionary into
            if curind > 0:
                nexttemplane = curroad[curind-1]
                enddiscpos = min(curpos, nexttemplane.end)
                enddiscpos = enddiscpos - 2*(p[0] + p[1])
                cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'l_lc'})

            # there is always a mandatory event
            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'r_lc',
                                        'lc_urgency': [curpos, curpos + p[0]],
                                        'endpos': endpos})

            # update iteration
            curind += -1
            templane = nexttemplane

    # same code but for opposite side
    elif curlaneind > laneind:  # populate route events in lanes right of target
        curind = laneind+1
        templane = curroad[curind]
        while not curind > curlaneind:
            cur_route[templane] = []
            # determine curpos = where the mandatory change starts
            if templane.end < curpos:
                curpos = templane.end
            curpos -= p[0] + p[1]

            if curind < curroad.num_lanes - 1:
                nexttemplane = curroad[curind + 1]
                enddiscpos = min(curpos, nexttemplane.end)
                enddiscpos = enddiscpos - 2*(p[0] + p[1])
                cur_route[templane].append({'pos': enddiscpos, 'event': 'end discretionary', 'side': 'r_lc'})

            cur_route[templane].append({'pos': curpos, 'event': 'mandatory', 'side': 'l_lc',
                                        'lc_urgency': [curpos, curpos + p[0]],
                                        'endpos': endpos})

            # update iteration
            curind += 1
            templane = nexttemplane

    return cur_route


def set_route_events(veh, timeind):
    """When a vehicle enters a new lane, this function generates all its route events for that lane.

    Every Lane has a list of 'route events' defined for it, which ensure that the Vehicle follows its
    specified route. Refer to update_route_events for a description of route events, and make_cur_route for
    a description of the route model.
    If a vehicle enters a new road, this function will generate the cur_route for that road and a subset
    of its lanes. This function will pop from the vehicle's route when that occurs. The exception to this
    is when vehicles are first initialized, the initialize method of Vehicle creates the first
    cur_route, and therefore pops from the route the first time.
    If a vehicle enters a new lane on the same road, it will either get the existing route
    from cur_route, or if the route for the new lane does not exist, it will create it add the key/value
    to cur_route. When creating a route for a new lane on the same road, it uses make_route_helper.

    Args:
        veh: Vehicle object which we will set its current route_events for.

    Returns:
        None. Modifies veh attributes in place (route_events, cur_route, possibly applies route events).
    """

    # get new route events if they are stored in memory already
    newlane = veh.lane
    if newlane in veh.cur_route:
        veh.route_events = veh.cur_route[newlane].copy()  # route_events store current route events, cur_route
        # stores all route events for subset of lanes on current road
    else:
        p = veh.route_parameters
        prevlane = veh.lanemem[-2][0]
        if prevlane.road is newlane.road:  # on same road - use helper function to update cur_route
            # need to figure out what situation we are in to give make route helper right call
            prevlane_events = veh.cur_route[prevlane]
            if not prevlane_events:  # this can only happen for continue event => curpos = end of lane
                curpos = prevlane.end
                make_route_helper(p, veh.cur_route, veh.lane.road, newlane.laneind, prevlane.laneind, curpos, curpos)
            elif prevlane_events[0]['event'] == 'end discretionary':
                curpos = prevlane_events[0]['pos'] + 2*(p[0] + p[1])
                make_route_helper(p, veh.cur_route, veh.lane.road, newlane.laneind, prevlane.laneind, curpos, curpos)
            else:  # mandatory event
                curpos, endpos = prevlane_events[0]['pos'], prevlane_events[0]['endpos']
                make_route_helper(p, veh.cur_route, veh.lane.road, newlane.laneind, prevlane.laneind, curpos, endpos)
        else:  # on new road - we need to generate new cur_route and update the vehicle's route
            veh.cur_route = make_cur_route(p, newlane, veh.route.pop(0))

        veh.route_events = veh.cur_route[newlane].copy()

    # for route events, past events need to be applied.
    update_route_events(veh, timeind)


def update_merge_anchors(curlane, lc_actions):
    """Updates merge_anchors attribute for curlane.

    Lanes have lists of merge anchors, they are used as guesses for leadfol_find for 'new lane' or
    'update lanes' events when a left or right lane is added. Thus, merge anchors are used to ensure
    the leader/follower relationships are updated correctly when the network topology changes.
    A merge anchor is defined as a (vehicle, position) tuple. vehicle can be either an anchor or normal
    vehicle. position can be either None or a float position. If position is None, vehicle is an anchor,
    and does not need to be updated. Otherwise, position is a float of the position on curlane,
    and the merge anchor is the vehicle on the same track as curlane which is closest to position without
    yet passing position.
    Merge anchors have fixed index.
    position being None corresponds to the situation where a new lane starts.
    position being a float corresponds to the situation where two lanes initially meet.
    Unlike lfol/rfol, merge anchors do not need to be completely updated. They should be kept
    in the same track as curlane however.

    Args:
        curlane: Lane object to update
        lc_actions: dictionary with keys as vehicles which request lane changes in the current timestep,
            values are a string either 'l' or 'r' which indicates the side of the change

    Returns:
        None. Modifies merge_anchors attribute for curlane
    """
    for i in range(len(curlane.merge_anchors)):
        veh, pos = curlane.merge_anchors[i][:]
        if pos is None:  # merge anchor is always an anchor we do nothing
            # update_lrfol(veh)  # no need to update lrfol for anchors
            pass
        else:
            # veh is an anchor -> we see if we can make its leader the new merge anchor
            if veh.cf_parameters is None:
                lead = veh.lead
                if lead is not None:
                    temp = curlane.roadlen[lead.road] + lead.pos
                    if temp - pos < 0:
                        curlane.merge_anchors[i][0] = lead

            elif veh in lc_actions:
                if lc_actions[veh] == 'l':
                    curlane.merge_anchors[i][0] = veh.rfol
                else:
                    curlane.merge_anchors[i][0] = veh.lfol

            elif curlane.roadlen[veh.road]+veh.pos - pos > 0:
                curlane.merge_anchors[i][0] = veh.fol