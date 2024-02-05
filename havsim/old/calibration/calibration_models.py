"""Example of subclassed models for Calibration"""

import math
import havsim.models as hm
import havsim.old.calibration as hc
import havsim.simulation as hs
from havsim import helper

class OVMCalibrationVehicle(hc.CalibrationVehicle):
    """Optimal Velocity Model Implementation."""
    def cf_model(self, p, state):
        return hm.OVM(p, state)

    def get_cf(self, hd, spd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if self.in_relax:
                # accident free formulation of relaxation
                # ttc = hd / (self.speed - lead.speed)
                # if ttc < 1.5 and ttc > 0:
                if False:  # disable accident free
                    temp = (ttc/1.5)**2
                    # currelax, currelax_v = self.relax[timeind-self.relax_start]  # hd + v relax
                    # currelax, currelax_v = currelax*temp, currelax_v*temp
                    currelax = self.relax[timeind - self.relax_start]*temp
                else:
                    # currelax, currelax_v = self.relax[timeind-self.relax_start]
                    currelax = self.relax[timeind - self.relax_start]

                # acc = self.cf_model(self.cf_parameters, [hd + currelax, spd, lead.speed + currelax_v])
                acc = self.cf_model(self.cf_parameters, [hd + currelax, spd, lead.speed])
            else:
                acc = self.cf_model(self.cf_parameters, [hd, spd, lead.speed])

        return acc

    def eqlfun(self, p, s):
        return hm.OVM_eql(p, s)

    def set_relax(self, relaxamounts, timeind, dt):
        rp = self.relax_parameters
        if rp is None:
            return
        relaxamount_s, relaxamount_v = relaxamounts
        hs.relaxation.relax_helper(rp, relaxamount_s, self, timeind, dt)

    def initialize(self, parameters):
        super().initialize(parameters)
        self.maxspeed = parameters[0]*(1-math.tanh(-parameters[2]))-.1
        self.eql_type = 's'  # you are supposed to set this in __init__


# for Newell
class NewellCalibrationVehicle(hc.CalibrationVehicle):
    """Implementation of Newell model in Differential form, example of 1st order ODE implementation."""
    def cf_model(self, p, state):
        """p = parameters (time shift, space shift), state = headway"""
        return (state - p[1])/p[0]

    def get_cf(self, hd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if self.in_relax:
                currelax = self.relax[timeind - self.relax_start]
                spd = self.cf_model(self.cf_parameters, hd+currelax)
            else:
                spd = self.cf_model(self.cf_parameters, hd)
        return spd

    def set_cf(self, timeind, dt):
        self.speed = self.get_cf(self.hd, self.lead, self.lane, timeind, dt, self.in_relax)

    def eqlfun(self, p, v):
        return p[0]*v+p[1]

    def set_relax(self, relaxamounts, timeind, dt):
        rp = self.relax_parameters
        if rp is None:
            return
        relaxamount_s, relaxamount_v = relaxamounts
        hs.relaxation.relax_helper(rp, relaxamount_s, self, timeind, dt)

    def update(self, timeind, dt):
        # bounds on speed must be applied for 1st order model
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
        if self.in_relax:
            if timeind == self.relax_end:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, self.relax))

        if self.in_leadveh:  # only difference is we update the LeadVehicle if applicable
            self.leadveh.update(timeind+1)


    def initialize(self, parameters):
        super().initialize(parameters)  # before the first cf call, the speed is initialized as initspd.
        # this handles the edge case for if a vehicle tries to access the speed before the first cf call.
        # after the first cf call, in this case the speed will simply be the speed from the previous timestep
        self.speedmem = []  # note that speedmem will be 1 len shorter than posmem for a 1st order model
        self.maxspeed = parameters[2]


class SKA_IDM(hc.CalibrationVehicle):
    """IDM with a relaxation model based on Schakel, Knoop, van Arem (2012).

    In the original paper, they give a full microsimulation model, and the relaxation is integrated in the
    sense that the 'desire' parameter controls both the gap acceptance as well as the relaxation amount.
    In this implementation, the relaxation amount is its own parameter, thus it has two relax parameters,
    the first being the desire which controls the relaxation amount, and the second being the rate
    of change for the relaxation.
    """
    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters[:-2].copy()
        self.relax_parameters = parameters[-2:].copy()
        self.relax_end = math.inf
        self.max_relax = parameters[1]

    def set_relax(self, relaxamounts, timeind, dt):
        # in_relax is always False, and we implement the relaxation by just
        # changing the time headway (cf_parameter[1]) appropriately
        self.relax_start = 'r'  # give special value 'r' in case we need to be adjusting the time headway
        temp = dt/self.relax_parameters[1]
        self.cf_parameters[1] = (self.relax_parameters[0] - self.max_relax*temp)/(1-temp)  # handle first
        if self.relax_parameters[0] >= self.max_relax: # don't apply relaxation for larger changes
            self.cf_parameters[1] = self.max_relax
            self.relax_start = None
        # relaxation value correctly (because it will be updated once before being used)

    def update(self, timeind, dt):
        super().update(timeind, dt)

        if self.relax_start == 'r':
            temp = dt/self.relax_parameters[1]
            self.cf_parameters[1] += (self.max_relax-self.cf_parameters[1])*temp

class RelaxExpIDM(hc.CalibrationVehicle):
    """Implements relaxation with an exponential adjustment rate."""
    def get_cf(self, hd, spd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if self.in_relax:
                currelax, currelax_v = self.currelax
                acc = self.cf_model(self.cf_parameters, [hd + currelax, spd, lead.speed + currelax_v])
            else:
                acc = self.cf_model(self.cf_parameters, [hd, spd, lead.speed])

        return acc

    def set_relax(self, relaxamounts, timeind, dt):
        if self.in_relax:
           self.currelax = (self.currelax[0] + relaxamounts[0], self.currelax[1] + relaxamounts[1])
        else:
            self.in_relax = True
            self.relax_end = math.inf
            self.currelax = relaxamounts

    def update(self, timeind, dt):
        super().update(timeind, dt)

        if self.in_relax:
            temp = 1 - dt/self.relax_parameters
            self.currelax = (self.currelax[0]*temp, self.currelax[1]*temp)

class Relax2IDM(hc.CalibrationVehicle):
    """Implements relaxation with 2 seperate parameters for positive/negative relaxation amounts."""
    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters[:-2]
        self.relax_parameters = parameters[-2:]

    def set_relax(self, relaxamounts, timeind, dt):
        """2 parameter positive/negative relaxation."""
        relaxamount_s, relaxamount_v = relaxamounts
        # make headway relax
        rp = self.relax_parameters[0] if relaxamount_s > 0 else self.relax_parameters[1]
        relaxlen = math.ceil(rp/dt) - 1
        tempdt = -dt/rp*relaxamount_s
        temp = [relaxamount_s + tempdt*i for i in range(1,relaxlen+1)]
        # make velocity relax
        rp2 = self.relax_parameters[0] if relaxamount_v > 0 else self.relax_parameters[1]
        relaxlen2 = math.ceil(rp2/dt) - 1
        tempdt = -dt/rp2*relaxamount_v
        temp2 = [relaxamount_v + tempdt*i for i in range(1,relaxlen2+1)]
        if max(relaxlen, relaxlen2) == 0:
            return
        # pad relax if necessary
        if relaxlen < relaxlen2:
            temp.extend([0]*(relaxlen2-relaxlen))
            relaxlen = relaxlen2
        elif relaxlen2 < relaxlen:
            temp2.extend([0]*(relaxlen-relaxlen2))
        # rest of code is the same as relax_helper_vhd
        curr = list(zip(temp, temp2))
        if self.in_relax:  # add to existing relax
            # find indexes with overlap - need to combine relax values for those
            overlap_end = min(self.relax_end, timeind+relaxlen)
            prevr_indoffset = timeind - self.relax_start+1
            prevr = self.relax
            overlap_len = max(overlap_end-timeind, 0)
            for i in range(overlap_len):
                curtime = prevr_indoffset+i
                prevrelax, currelax = prevr[curtime], curr[i]
                prevr[curtime] = (prevrelax[0]+currelax[0], prevrelax[1]+currelax[1])
            prevr.extend(curr[overlap_len:])
            self.relax_end = max(self.relax_end, timeind+relaxlen)
        else:
            self.in_relax = True
            self.relax_start = timeind + 1  # add relax
            self.relax = curr
            self.relax_end = timeind + relaxlen

class Relax2vhdIDM(hc.CalibrationVehicle):
    """Implements relaxation with 2 seperate parameters for headway/velocity relaxation amounts."""
    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters[:-2]
        self.relax_parameters = parameters[-2:]

    def set_relax(self, relaxamounts, timeind, dt):
        """2 parameter positive/negative relaxation."""
        relaxamount_s, relaxamount_v = relaxamounts
        # make headway relax
        rp = self.relax_parameters[0]
        relaxlen = math.ceil(rp/dt) - 1
        tempdt = -dt/rp*relaxamount_s
        temp = [relaxamount_s + tempdt*i for i in range(1,relaxlen+1)]
        # make velocity relax
        rp2 = self.relax_parameters[1]
        relaxlen2 = math.ceil(rp2/dt) - 1
        tempdt = -dt/rp2*relaxamount_v
        temp2 = [relaxamount_v + tempdt*i for i in range(1,relaxlen2+1)]
        if max(relaxlen, relaxlen2) == 0:
            return
        # pad relax if necessary
        if relaxlen < relaxlen2:
            temp.extend([0]*(relaxlen2-relaxlen))
            relaxlen = relaxlen2
        elif relaxlen2 < relaxlen:
            temp2.extend([0]*(relaxlen-relaxlen2))
        # rest of code is the same as relax_helper_vhd
        curr = list(zip(temp, temp2))
        if self.in_relax:  # add to existing relax
            # find indexes with overlap - need to combine relax values for those
            overlap_end = min(self.relax_end, timeind+relaxlen)
            prevr_indoffset = timeind - self.relax_start+1
            prevr = self.relax
            overlap_len = max(overlap_end-timeind, 0)
            for i in range(overlap_len):
                curtime = prevr_indoffset+i
                prevrelax, currelax = prevr[curtime], curr[i]
                prevr[curtime] = (prevrelax[0]+currelax[0], prevrelax[1]+currelax[1])
            prevr.extend(curr[overlap_len:])
            self.relax_end = max(self.relax_end, timeind+relaxlen)
        else:
            self.in_relax = True
            self.relax_start = timeind + 1  # add relax
            self.relax = curr
            self.relax_end = timeind + relaxlen


class RelaxShapeIDM(hc.CalibrationVehicle):
    """Implements 2 parameter relaxation where the second parameter controls the shape."""
    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters[:-2]
        self.relax_parameters = parameters[-2:]

    def set_relax(self, relaxamounts, timeind, dt):
        relaxamount_s, relaxamount_v = relaxamounts
        # parametrized by class of monotonically decreasing second order polynomials
        rp = self.relax_parameters[0]
        p = self.relax_parameters[-1]
        p1 = -p-1
        tempdt = dt/rp
        relaxlen = math.ceil(rp/dt) - 1
        if relaxlen == 0:
            return
        temp = [relaxamount_s*(p*(i*tempdt)**2+p1*i*tempdt+1) for i in range(1,relaxlen+1)]
        temp2 = [relaxamount_v*(p*(i*tempdt)**2+p1*i*tempdt+1) for i in range(1,relaxlen+1)]
        # rest of code is the same as relax_helper_vhd
        curr = list(zip(temp, temp2))
        if self.in_relax:  # add to existing relax
            # find indexes with overlap - need to combine relax values for those
            overlap_end = min(self.relax_end, timeind+relaxlen)
            prevr_indoffset = timeind - self.relax_start+1
            prevr = self.relax
            overlap_len = max(overlap_end-timeind, 0)
            for i in range(overlap_len):
                curtime = prevr_indoffset+i
                prevrelax, currelax = prevr[curtime], curr[i]
                prevr[curtime] = (prevrelax[0]+currelax[0], prevrelax[1]+currelax[1])
            prevr.extend(curr[overlap_len:])
            self.relax_end = max(self.relax_end, timeind+relaxlen)
        else:
            self.in_relax = True
            self.relax_start = timeind + 1  # add relax
            self.relax = curr
            self.relax_end = timeind + relaxlen


class NewellTT(hc.CalibrationVehicle):
    """Implements Newell model in trajectory translation form, based on Laval, Leclerq (2008).

    3 parameters (4 with relaxation) like the model in differential form, but this form is different. It
    should be equivalent to NewellCalibrationVehicle up to floating point error unless the maximum
    acceleration bound is active.
    """
    def cf_model(self, p, state):
        """
        p - [space shift (1/\kappa), wave speed (\omega), max speed]
        state - [headway, leader speed]
        """
        K = (p[1]/p[0])/(p[1]+state[1])
        tau = p[0]/p[1]
        return (state[0] + state[1]*tau - 1/K)/tau


    def get_cf(self, hd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)

        else:
            if self.in_relax:
                currelax, currelax_v = self.relax[timeind - self.relax_start]
                spd = self.cf_model(self.cf_parameters, [self.hd + currelax, lead.speed+currelax_v])
            else:
                spd = self.cf_model(self.cf_parameters, [self.hd, lead.speed])
        return spd

    def set_cf(self, timeind, dt):
        self.speed = self.get_cf(self.hd, self.lead, self.lane, timeind, dt, self.in_relax)

    def eqlfun(self, p, v):
        pass

    def update(self, timeind, dt):
        #calculate which regime we are in (congested, free flow, or free acceleration)
        freespeed = min(self.prevspeed+self.maxacc*dt, self.maxspeed)
        curspeed = min(self.speed, freespeed)

        if curspeed < 0:
            curspeed = 0
        elif curspeed > self.maxspeed:
            curspeed = self.maxspeed
        # update state
        self.pos += curspeed*dt
        self.speed = curspeed
        self.prevspeed = curspeed
        # update memory
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        if self.in_relax:
            if timeind == self.relax_end:
                self.in_relax = False
                self.relaxmem.append((self.relax_start, self.relax))

        if self.in_leadveh:
            self.leadveh.update(timeind+1)

    def initialize(self, parameters):
        super().initialize(parameters)

        self.speedmem = []
        self.maxspeed = parameters[2]
        self.maxacc = 3.4*3.28
        self.prevspeed = self.speed


class NewellLL(NewellTT):
    """Relaxation model proposed in Laval, Leclerq (2008). Use with the correct lc_event function."""
    # only designed to work for 1 vehicle at a time because of the way the DeltaN updates are handled.
    def cf_model(self, p, state):
        """
        p - [space shift (1/\kappa), wave speed (\omega), max speed, epsilon]
        state - [headway, leader speed, dt, DeltaN]
        """
        K = (p[1]/p[0])/(p[1]+state[1])
        tau = p[0]/p[1]
        return (state[0] + state[1]*tau - state[2]/K)/tau


    def get_cf(self, hd, lead, curlane, timeind, dt, userelax):
        if lead is None:
            acc = curlane.call_downstream(self, timeind, dt)
        else:
            spd = self.cf_model(self.cf_parameters, [self.hd, lead.speed, self.DeltaN])
        return spd

    def set_relax(self, relaxamounts, timeind, dt):
        self.DeltaN = relaxamounts
        self.in_relax = True
        self.first_index = True

    def update(self, timeind, dt):
        #calculate which regime we are in (congested, free flow, or free acceleration)
        freespeed = min(self.prevspeed+self.maxacc*dt, self.maxspeed)
        curspeed = min(self.speed, freespeed)

        if curspeed < 0:
            curspeed = 0
        elif curspeed > self.maxspeed:
            curspeed = self.maxspeed
        # update state
        self.pos += curspeed*dt
        self.speed = curspeed
        self.prevspeed = curspeed
        # update memory
        self.posmem.append(self.pos)
        self.speedmem.append(self.speed)
        self.DeltaNmem.append(self.DeltaN)

        self.prevleadspeed = self.lead.speed  # this is why this would only work for 1 vehicle at a time -
        # for multiple vehicles you would need to modify the calibration.step so the prevleadspeed is updated
        # correctly.
        if self.in_leadveh:
            self.leadveh.update(timeind+1)

        if self.in_relax:
            if self.first_index:
                self.first_index = False
            else:
                p = self.cf_parameters
                vtilde = self.lead.speed*(1-self.DeltaN)+self.prevleadspeed*self.DeltaN - p[3]
                Kvj = (p[1]/p[0])/(p[1]+self.prevleadspeed)
                Kvjp1 = (p[1]/p[0])/(p[1]+self.lead.speed)
                self.DeltaN = (self.DeltaN/Kvj + (self.lead.speed-vtilde)*dt)*Kvjp1
                if self.DeltaN >= 1:
                    self.DeltaN = 1
                    self._in_relax = False


    def initialize(self, parameters):
        super().initialize(parameters)

        self.cf_parameters = parameters
        #attributes for the LL relaxation formulation
        self.prevleadspeed = None
        self.DeltaN = 1
        self.DeltaNmem = []
        self.first_index = False


def make_ll_lc_event(vehicles, id2obj, meas, platooninfo, dt, addevent_list, lcevent_list):
    """Makes the lc_events for NewellLL model."""
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

    return addevent_list, lcevent_list

def ll_lc_event(event, timeind, dt):
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
        leadstate - tuple of (x1, x2, ip1) which follows notation by LL

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


class NoRelaxIDM(hc.CalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):  # just need to set parameters correctly
        super().initialize(parameters)
        self.cf_parameters = parameters


class NoRelaxOVM(OVMCalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters


class NoRelaxNewell(NewellCalibrationVehicle):
    def set_relax(self, *args):
        pass

    def initialize(self, parameters):
        super().initialize(parameters)
        self.cf_parameters = parameters
        
