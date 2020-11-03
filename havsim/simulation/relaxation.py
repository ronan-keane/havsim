
"""
Functions for applying relaxation after a lane change.
"""
import math
from havsim.simulation.road_networks import get_headway

def new_relaxation(veh, timeind, dt, relax_speed=False):
    """Generates relaxation for a vehicle after it experiences a lane change.

    This is called directly after a vehicle changes it lane, while it still has the old value for its
    headway, and its position has not yet been updated.
    See (https://arxiv.org/abs/1904.08395) for an explanation of the relaxation model.
    This implements single parameter relaxation. The Vehicle attributes associated with relaxation are
        relax_parameters: float which gives the relaxation constant (units of time)
        in_relax: bool of whether or not vehicle is experiencing relaxation currently
        relax: list of floats which stores the relaxation values
        relax_start: time index (timeind) of the 0 index of relax
        relaxmem: memory of past relaxation, list of tuples of (starttime, endtime, relax)

    Args:
        veh: Vehicle to add relaxation to
        timeind: int giving the timestep of the simulation (0 indexed)
        dt: float of time unit that passes in each timestep
        relax_speed: If True, we relax leader's speed as well.

    Returns:
        None. Modifies relaxation attributes for vehicle in place.
    """
    # TODO can add seperate parameters for positive/negative relaxation
    rp = veh.relax_parameters
    if rp is None:  # no relax -> do nothing
        return
    if veh.lead is None:  
        if veh.in_relax:  # new lead is None -> reset relaxation
            veh.in_relax = False
            veh.relax = veh.relax[:timeind-veh.relax_start]
            veh.relaxmem.append((veh.relax, veh.relax_start))
        return
        

    prevlead = veh.leadmem[-2][0]
    if prevlead is None:
        olds = veh.get_eql(veh.speed)
        # olds = veh.get_eql(veh.lead.speed)
    else:
        olds = veh.hd
    news = get_headway(veh, veh.lead)

    if relax_speed:  # relax speed + headway so we have a list of tuples
        if prevlead is None:
            oldv = veh.speed
        else:
            oldv = prevlead.speed
        newv = veh.lead.speed

        relaxamount_s = olds-news
        relaxamount_v = oldv-newv
        relax_helper_vhd(rp, relaxamount_s, relaxamount_v, veh, timeind, dt)


    else:  # relax headway only = list of float of relax values
        relaxamount = olds-news  # edge case error here, see update_lane_routes.update_veh_after_lc
        relax_helper(rp, relaxamount, veh, timeind, dt)


def relax_helper_vhd(rp, relaxamount_s, relaxamount_v, veh, timeind, dt):
    """Helper function for headway + speed relaxation."""
    #rp = parameter, relaxamount_s = headway relaxation, _v = velocity relaxation
    relaxlen = math.ceil(rp/dt) - 1
    if relaxlen == 0:
        return
    tempdt = -dt/rp*relaxamount_s
    tempdt2 = -dt/rp*relaxamount_v
    temp = [relaxamount_s + tempdt*i for i in range(1,relaxlen+1)]
    temp2 = [relaxamount_v + tempdt2*i for i in range(1, relaxlen+1)]
    curr = list(zip(temp,temp2))

    if veh.in_relax:  # add to existing relax
        # find indexes with overlap - need to combine relax values for those
        overlap_end = min(veh.relax_end, timeind+relaxlen)
        prevr_indoffset = timeind - veh.relax_start+1
        prevr = veh.relax
        overlap_len = max(overlap_end-timeind, 0)
        for i in range(overlap_len):
            curtime = prevr_indoffset+i
            prevrelax, currelax = prevr[curtime], curr[i]
            prevr[curtime] = (prevrelax[0]+currelax[0], prevrelax[1]+currelax[1])
        prevr.extend(curr[overlap_len:])
        veh.relax_end = max(veh.relax_end, timeind+relaxlen)
    else:
        veh.in_relax = True
        veh.relax_start = timeind + 1  # add relax
        veh.relax = curr
        veh.relax_end = timeind + relaxlen


def relax_helper(rp, relaxamount, veh, timeind, dt):
    """Helper function for headway only relaxation."""
    #rp = parameter, relaxamount = float relaxation amount
    relaxlen = math.ceil(rp/dt) - 1
    if relaxlen == 0:
        return
    tempdt = -dt/rp*relaxamount
    curr = [relaxamount + tempdt*i for i in range(1,relaxlen+1)]

    if veh.in_relax:  # add to existing relax
        overlap_end = min(veh.relax_end, timeind+relaxlen)
        prevr_indoffset = timeind - veh.relax_start+1
        prevr = veh.relax
        overlap_len = max(overlap_end-timeind, 0)
        for i in range(overlap_len):
            prevr[prevr_indoffset+i] += curr[i]
        prevr.extend(curr[overlap_len:])
        veh.relax_end = max(veh.relax_end, timeind+relaxlen)
    else:
        veh.in_relax = True
        veh.relax_start = timeind + 1  # add relax
        veh.relax = curr
        veh.relax_end = timeind + relaxlen

##### previous numpy based code for relax - using python lists/list comprehension now
# relaxlen = math.ceil(rp/dt) - 1
# curr = np.zeros((relaxlen,2))
# curr[:,0] = np.linspace((1 - dt/rp)*relaxamount_s, (1 - dt/rp*relaxlen)*relaxamount_s, relaxlen)
# curr[:,1] = np.linspace((1 - dt/rp)*relaxamount_v, (1 - dt/rp*relaxlen)*relaxamount_v, relaxlen)
# if veh.in_relax:
#     curlen = len(veh.relax)
#     newend = timeind + relaxlen  # time index when relax ends
#     newrelax = np.zeros((newend - veh.relax_start+1, 2))
#     newrelax[0:curlen,:] = veh.relax
#     newrelax[timeind-veh.relax_start+1:,:] += curr
#     veh.relax = newrelax
# else:
#     veh.in_relax = True
#     veh.relax_start = timeind + 1
#     veh.relax = curr

#### def new_relaxation_acc(veh, timeind, dt):
#     """Relaxation for acceleration. Recommended to use new_relaxation instead."""
#     # This formulation is not as robust as relaxing speed/headway instead.
#     rp = veh.relax_parameters
#     lead = veh.lead
#     if lead is None or rp is None:
#         return
#     oldacc = veh.acc
#     newhd = get_headway(veh, lead)
#     newacc = veh.get_cf(newhd, veh.speed, lead, veh.lane, timeind, dt, veh.in_relax)
#     relax_helper(rp, oldacc-newacc, veh, timeind, dt)