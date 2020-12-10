"""More efficient implementation of infiniteroad_simulation which only partially simulates vehicles."""

import havsim.simulation as hs
import havsim.plotting as hp
import time

#%%
IDM_parameters = [35, 2, 2, 1.1, 1.5]  # in order: max speed, time headway, jam spacing, comfortable acceleration,
# comfortable deceleration. Units are in meters.
eql_speed = 20  # define the equilibrium speed you want to perturb around
nveh = 5000  # number of vehicles
dt = .25  # timestep in seconds
acc_tolerance = 1e-3  # acc tolerance for adding new vehicles
speed_tolerance = 1e-1  # speed tolerance for subtracting vehicles


# define speed profile of initial vehicle
def downstream(timeind, *args):
    if timeind < 50:
        return eql_speed-15
    else:
        return eql_speed
# define initial headway of the first following vehicle
init_hd = hs.Vehicle(-1, None, IDM_parameters, None).get_eql(eql_speed)
#%%

mainroad_len= 1e10
mainroad = hs.Road(1,mainroad_len, 'main road')
mainroad.connect('exit', is_exit=True)
mainroad.set_downstream({'method':'speed', 'time_series':downstream})
def newveh(vehid, *args):
    cf_p = IDM_parameters
    unused, lc_p = hs.models.IDM_parameters()
    kwargs = {'route': ['exit'], 'maxspeed':cf_p[0]}
    return hs.Vehicle(vehid, mainroad[0], cf_p, lc_p, **kwargs)

def make_waves():
    """Simulates the evolution of a traffic wave initiated by downstream over nveh vehicles.

    We assume that all vehicles start in the equilibrium defined by eql_speed. The first vehicle follows
    the speed profile defined by the downstream function. This simulates how this initial speed profile
    evolves as it propagates through the vehicles. This implements an efficient algorithm that scales to large
    numbers of vehicles/timesteps. The infiniteroad_simulation.py script has code which does the full
    simulation; the full simulation was used to validate this algorithm.

    At every timestep, we first evaluate whether to add a new vehicle to the simulation. We can calculate
    in closed form the trajectory of the next following vehicle, assuming it stays in equilibrium. This
    is used to evaluate the acceleration of this potential new vehicle, if it is added to the simulation.
    If the acceleration is greater than the acc_tolerance threshold, the new vehicle is added and will
    begin to be fully simulated. Otherwise, we approximate its trajectory by the equilibrium solution.
    To evaluate whether to remove the most downstream vehicles from the simulation, we calculate
    the difference between its current speed and the equilibrium speed. If the difference is less than
    speed_tolerance, then this most downstream vehicle is no longer fully simulated, and we approximate
    its trajectory by the equilibrium solution.
    """
    # initialization
    next_initpos = 1e5
    testveh = newveh(-1)
    testveh.speed = eql_speed
    leadveh = newveh(-2)
    leadveh.speed = eql_speed
    leadveh.pos = 0
    eql_hd = testveh.get_eql(eql_speed)
    cur_vehicles = []
    all_vehicles = []
    prev_veh = None
    counter = 0
    curtime = 0
    while counter < nveh and (len(cur_vehicles) > 0 or counter==0):
        # check if we need to add a new vehicle
        testveh.pos = next_initpos + curtime*dt*eql_speed
        testhd = hs.get_headway(testveh, prev_veh) if prev_veh is not None else None
        acc = testveh.get_cf(testhd, eql_speed, prev_veh, mainroad[0], curtime, dt, False)
        if abs(acc*dt) > acc_tolerance:
            veh = newveh(counter)
            veh.lead = prev_veh
            if counter > 0:
                hd = eql_hd
            else:
                hd = init_hd
            veh.initialize(testveh.pos, eql_speed, testhd, curtime)
            next_initpos += -hd - veh.len
            cur_vehicles.append(veh)
            prev_veh = veh
            counter += 1

        # update simulation
        for veh in cur_vehicles:
            veh.set_cf(curtime, dt)
        for veh in cur_vehicles:
            veh.update(curtime, dt)
        leadveh.pos += leadveh.speed*dt
        leadveh.speed = eql_speed
        for veh in cur_vehicles:
            veh.hd = hs.get_headway(veh, veh.lead) if veh.lead is not None else None
        curtime += 1

        # check if we need to remove vehicles
        # if len(cur_vehicles)>0:
        veh = cur_vehicles[0]
        if abs(veh.speed-eql_speed)<speed_tolerance:
            all_vehicles.append(veh)
            veh.end = curtime
            leadveh.pos = veh.pos
            leadveh.speed = veh.speed
            cur_vehicles.pop(0)
            if len(cur_vehicles) > 0:
                cur_vehicles[0].lead = leadveh

    return all_vehicles, cur_vehicles, curtime

#%%  simulation and plotting
start = time.time()
all_vehicles, cur_vehicles, timesteps = make_waves()
end = time.time()
print('simulation time is '+str(end-start)+' over '+str(sum([timesteps - veh.start+1 if veh.end is None else veh.end - veh.start+1
                                                         for veh in all_vehicles]))+' timesteps')
all_vehicles.extend(cur_vehicles)

laneinds = {mainroad[0]:0}
sim, siminfo = hp.plot_format(all_vehicles, laneinds)

hp.platoonplot(sim,None, siminfo, lane=0, opacity=0)


