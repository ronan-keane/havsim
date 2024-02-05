
"""
Functions for updating vehicle orders.
"""
from havsim import get_dist
import numpy as np
import math

def update_lrfol(veh):
    """After a vehicle's state has been updated, this updates its left and right followers.

    It is recommended to use update_all_lrfol_multiple instead of this.
    We keep each vehicle's left/right followers updated by doing a single distance compute each timestep.
    The current way of dealing with l/r followers is designed for timesteps which are relatively small.
    (e.g. .1 or .25 seconds) where we also keep l/rfol updated at all timesteps. Other strategies would be
    better for larger timesteps or if l/rfol don't need to always be updated. See block comment
    above update_change to discuss alternative strategies for defining leader/follower relationships.
    See update_change documentation for explanation of naming conventions.
    Called in simulation by update_all_lrfol.
    update_all_lrfol_multiple can be used to handle edge cases so that the vehicle order is always correct,
    but is slightly more computationally expensive.
    The edge cases occur when vehicles overtake multiple vehicles in a single timestep.

    Args:
        veh: Vehicle to check its lfol/rfol to be updated.

    Returns:
        None. Modifies veh attributes, attributes of its lfol/rfol, in place.
    """
    lfol, rfol = veh.lfol, veh.rfol
    if lfol is None:
        pass
    elif get_dist(veh, lfol) > 0:
        # update for veh
        veh.lfol = lfol.fol
        veh.lfol.rlead.append(veh)
        lfol.rlead.remove(veh)
        # update for lfol
        lfol.rfol.llead.remove(lfol)
        lfol.rfol = veh
        veh.llead.append(lfol)

    # similarly for right
    if rfol is None:
        pass
    elif get_dist(veh, rfol) > 0:
        veh.rfol = rfol.fol
        veh.rfol.llead.append(veh)
        rfol.llead.remove(veh)

        rfol.lfol.rlead.remove(rfol)
        rfol.lfol = veh
        veh.rlead.append(rfol)


def update_all_lrfol(vehicles):
    """Updates all vehicles left and right followers, without handling multiple overtakes edge cases."""
    for veh in vehicles:
        update_lrfol(veh)


def update_all_lrfol_multiple(vehicles):
    """Handles edge cases where a single vehicle overtakes 2 or more vehicles in a timestep.

    In update_lrfol, when there are multiple overtakes in a single timestep (i.e. an l/rfol passes 2 or more
    l/rlead vehicles, OR a vehicle is overtaken by its lfol, lfol.fol, lfol.fol.fol, etc.) the vehicle order
    can be wrong for the next timestep. For every overtaken vehicle, 1 additional timestep with no overtakes
    is sufficient but not necessary to correct the vehicle order. So if an ego vehicle overtakes 2 vehicles
    on timestep 10, timestep 11 will have an incorrect order. More detail in comments.
    This version is slightly slower than update_all_lrfol but handles the edge cases.
    """
    # edge case 1 - a vehicle veh overtakes 2 llead, llead and llead.lead, in a timestep. If llead.lead
    # has its update_lrfol called first, and then llead, the vehicle order will be wrong because veh will
    # have llead as a rfol. If llead has its update_lrfol called first, the vehicle order will be correct.
    # for the wrong order to be correct, there will need to be 1 timestep where veh does not overtake, so that
    # the llead.lead will see veh overtook it and the order will be corrected. If, instead, veh overtakes
    # llead.lead.lead, then again whether the order will be correct depends on the update order (it will
    # be correct if llead.lead updates first, otherwise it will be behind 1 vehicle like before).

    # edge case 2 - a vehicle veh is overtaken by lfol and lfol.fol in the same timestep. Here veh will have
    # lfol.fol as its lfol. The order will always be corrected in the next timestep, unless another
    # edge case (either 1 or 2) occurs.
    # note edge case 2 is rarer and can potentially not be checked for - remove the update_recursive call
    # and we won't check for it.

    lovertaken = {}  # key with overtaking vehicles as keys, values are a list of vehicles the key overtook
    # lovertaken = left overtaken meaning a vehicles lfol overtook
    rovertaken = {}  # same as lovertaken but for right side
    # first loop we update all vehicles l/rfol and keep track of overtaking vehicles
    for veh in vehicles:
        lfol, rfol = veh.lfol, veh.rfol
        if lfol is None:
            pass
        else:
            update_lfol_recursive(veh, lfol, lovertaken)

        # same for right side
        if rfol is None:
            pass
        else:
            update_rfol_recursive(veh, rfol, rovertaken)

    #now to finish the order we have to update all vehicles which overtook
    for lfol, overtook in lovertaken.items():
        if len(overtook) == 1:  # we know what lfol new rfol is - it can only be one thing
            veh = overtook[0]
        else:
            distlist = [get_dist(veh, lfol) for veh in overtook]
            ind = np.argmin(distlist)
            veh = overtook[ind]
        # update for lfol
        # try/except is for rare edge case when lfol.rfol is None - occurs when you overtake and get new rlane
        # in the same timestep
        try:
            lfol.rfol.llead.remove(lfol)
        except:
            pass
        lfol.rfol = veh
        veh.llead.append(lfol)

    # same for right side
    for rfol, overtook in rovertaken.items():
        if len(overtook) == 1:  # we know what lfol new rfol is - it can only be one thing
            veh = overtook[0]
        else:
            distlist = [get_dist(veh, rfol) for veh in overtook]
            ind = np.argmin(distlist)
            veh = overtook[ind]
        # update for rfol
        try:
            rfol.lfol.rlead.remove(rfol)
        except:
            pass
        rfol.lfol = veh
        veh.rlead.append(rfol)


def update_lfol_recursive(veh, lfol, lovertaken):
    """Handles edge case 2 for update_all_lrfol_multiple by allowing lfol to update multiple times."""
    if get_dist(veh, lfol) > 0:
        # update for veh
        veh.lfol = lfol.fol
        veh.lfol.rlead.append(veh)
        lfol.rlead.remove(veh)
        # updating lovertaken so we can handle edge case 1
        if lfol in lovertaken:
            lovertaken[lfol].append(veh)
        else:
            lovertaken[lfol] = [veh]
        # to handle edge case 2 we update recursively if lfol overtakes
        update_lfol_recursive(veh, lfol.fol, lovertaken)


def update_rfol_recursive(veh, rfol, rovertaken):
    """Handles edge case 2 for update_all_lrfol_multiple by allowing rfol to update multiple times."""
    if get_dist(veh, rfol) > 0:

        veh.rfol = rfol.fol
        veh.rfol.llead.append(veh)
        rfol.llead.remove(veh)

        if rfol in rovertaken:
            rovertaken[rfol].append(veh)
        else:
            rovertaken[rfol] = [veh]
        update_rfol_recursive(veh, rfol.fol, rovertaken)

######### lfol/rfol/llead/rlead explanation
# explanation of why leader follower relationships are designed this way (i.e. why we have lfol/llead etc.)
# in current logic, main cost per timestep is just one distance compute in update_lrfol
# whenever there is a lane change, there are a fair number of extra updates we have to do to keep all
# of the rlead/llead updated. Also, whenever an lfol/rfol changes, there are two rlead/lead attributes
# that need to be changed as well. Thus this strategy is very efficient assuming we want to keep lfol/rfol
# updated (call lc every timestep), lane changes aren't super common, and all vehicles travel at around
# the same speed. (which are all reasonable assumptions)

# naive way would be having to do something like keep a sorted list, every time we want lfol/rfol
# we have to do log(n) dist computes, where n is the number of vehicles in the current lane.
# whenever a vehicle changes lanes, you need to remove from the current list and add to the new,
# so it is log(n) dist computations + 2n for searching/updating the 2 lists.
# Thus the current implementation is definitely much better than the naive way.

# Another option you could do is to only store lfol/rfol, to keep it updated you would have to
# do 2 dist calculations per side per timestep (do a call of leadfol find where we already have either
# a follower or leader as guess. To explain why you need to do 2 dist calculations - consider how to
# update the opsidefol of the new lcside leaders).
# When there is a lane change store a dict which has the lane changing
# vehicle as a key, and store as the value the new guess to use. You would need to use this dict to update
# guesses whenever it is time to update the lfol/rfol. Updating guesses efficiently is the challenge here.
# This strategy would have higher costs per timestep to keep lfol/rfol updated, but would be simpler to
# update when there is a lane change. Thus it might be more efficient if the number of timesteps is small
# relative to the number of lane changes.
# Because you don't need the llead/rlead in this option, you also can avoid fully updating l/rfol every
# timestep, although you would need to check if your l/rfol had a lane change, but that operation
# just checks for a hash value or compares anchor vehicles, which is less expensive than a get_dist call.

# More generally, the different strategies (always updating lead/fol relationships, having l/rlead or not)
# can be combined. If you don't need l/rlead, then there is no reason to keep it updated (or even have it).
# likewise, if you don't need to always have the lead/fol updated every timestep, there is no reason to
# keep it updated. On the other hand, if you need more than just the lfol/rfol, perhaps those should
# kept updated as well.
# TODO optimizing/refactoring the vehicle orders can be considered
# ######

def update_leadfol_after_lc(veh, lcsidelane, newlcsidelane, side, timeind):
    """Logic for updating all the leader/follower relationships for veh following a lane change.

    Args:
        veh: Vehicle object which changed lanes.
        lcsidelane: The new lane veh changed onto.
        newlcsidelane: The new lane on the lane change side for veh.
        side: side of lane change, either left ('l') or right ('r').
        timeind: int giving the timestep of the simulation (0 indexed)

    Returns:
        None. Modifies veh and any vehicles which have leader/follower relationships with veh in place.
    """
    if side == 'l':
        # define lcside/opside
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'lfol', 'rfol', 'llead', 'rlead'
    else:
        lcsidefol, opsidefol, lcsidelead, opsidelead = 'rfol', 'lfol', 'rlead', 'llead'

    # update current leader
    lead = veh.lead
    fol = veh.fol
    if lead is None:
        pass
    else:
        lead.fol = fol

    # update cur opposite/lc side leaders
    for j in getattr(veh, opsidelead):
        setattr(j, lcsidefol, fol)
    for j in getattr(veh, lcsidelead):
        setattr(j, opsidefol, fol)

    # update cur follower
    getattr(fol, lcsidelead).extend(getattr(veh, lcsidelead))
    getattr(fol, opsidelead).extend(getattr(veh, opsidelead))
    fol.lead = lead
    fol.leadmem.append((lead, timeind+1))

    # update cur opposite side follower
    vehopsidefol = getattr(veh, opsidefol)
    if vehopsidefol is not None:
        getattr(vehopsidefol, lcsidelead).remove(veh)
    setattr(veh, opsidefol, fol)
    getattr(fol, lcsidelead).append(veh)
    # update cur lc side follower for vehicle
    lcfol = getattr(veh, lcsidefol)
    lclead = lcfol.lead
    lcfol.lead = veh
    lcfol.leadmem.append((veh, timeind+1))
    getattr(lcfol, opsidelead).remove(veh)
    veh.fol = lcfol
    # update cur lc side leader
    veh.lead = lclead
    veh.leadmem.append((lclead, timeind+1))
    if lclead is not None:
        lclead.fol = veh

    # new opside leaders
    newleads = []
    oldleads = getattr(lcfol, opsidelead)
    for j in oldleads.copy():
        curdist = get_dist(veh, j)
        if curdist > 0:
            setattr(j, lcsidefol, veh)
            newleads.append(j)
            oldleads.remove(j)
    setattr(veh, opsidelead, newleads)
    # new lcside leaders
    newleads = []
    oldleads = getattr(lcfol, lcsidelead)
    mindist = math.inf
    minveh = None
    newlcsidelane_anchor = newlcsidelane.anchor if newlcsidelane is not None else None
    for j in oldleads.copy():
        curdist = get_dist(veh, j)
        if curdist > 0:
            setattr(j, opsidefol, veh)
            newleads.append(j)
            oldleads.remove(j)
            if curdist < mindist and j.lane.anchor is newlcsidelane_anchor:
                mindist = curdist
                minveh = j  # minveh is the leader of new lc side follower
    setattr(veh, lcsidelead, newleads)

    # update new lcside follower
    if newlcsidelane is None:
        setattr(veh, lcsidefol, None)
    elif minveh is not None:
        setattr(veh, lcsidefol, minveh.fol)
        getattr(minveh.fol, opsidelead).append(veh)
    else:
        guess = get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane)
        unused, newlcsidefol = lcsidelane.leadfol_find(veh, guess, side)
        setattr(veh, lcsidefol, newlcsidefol)
        getattr(newlcsidefol, opsidelead).append(veh)


def get_guess(lcfol, lclead, veh, lcsidefol, newlcsidelane):
    """Generates a guess to use for finding the newlcside follower for a vehicle.

    Since we keep the lfol/rfol updated, after a lane change we need to find the newlcside follower if
    applicable. This function will generate a guess by using the lcside fol of the lcfol/lclead.
    If such a guess cannot be found, we use the AnchorVehicle for the newlcside lane.


    Args:
        lcfol: lcfol for veh (fol after lane change)
        lclead: lclead for veh (lead after lane change)
        veh: want to generate a guess to find veh's new lcside follower.
        lcsidefol: str, either lfol if vehicle changes left or rfol otherwise.
        newlcsidelane: new lane change side lane after veh changes lanes.

    Returns:
        guess: Vehicle/AnchorVehicle to use as guess for leadfol_find

    """
    # need to find new lcside follower for veh
    guess = getattr(lcfol, lcsidefol)
    anchor = newlcsidelane.anchor
    if guess is None or guess.lane.anchor is not anchor:
        if lclead is not None:
            guess = getattr(lclead, lcsidefol)
            if guess is None or guess.lane.anchor is not anchor:
                guess = anchor
        else:
            guess = anchor
    return guess