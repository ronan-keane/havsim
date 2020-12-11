"""Full simulation of vehicles on a single lane, straight, infinite road."""

import havsim.simulation as hs
import matplotlib.pyplot as plt
import numpy as np
import havsim.plotting as hp
import time


#%% Set up

IDM_parameters = [30, 1, 4, 1.3, 2]  # in order: max speed, time headway, jam spacing, comfortable acceleration,
# comfortable deceleration. Units are in meters.
eql_speed = 28  # define the equilibrium speed you want to perturb around
nveh = 1000  # number of vehicles
nt = 3000  # number of timesteps
dt = .25  # timestep in seconds
# define speed profile of lead vehicle
def downstream(timeind, *args):
    if timeind < 50:
        return eql_speed-25
    else:
        return eql_speed

#%% Plot the equiliibrum solution FD and selected equiliibrum point

tempveh = hs.Vehicle(-1, None, IDM_parameters, None, maxspeed = IDM_parameters[0], length=5)
spds = np.arange(0,IDM_parameters[0],.01)
flows = np.array([tempveh.get_flow(i) for i in spds])
density = np.divide(flows,spds)
plt.figure()
plt.plot( density*1000, flows*3600,)  # convert to units of km, hours
plt.plot(tempveh.get_flow(eql_speed)/eql_speed*1000, tempveh.get_flow(eql_speed)*3600, 'ko')



#%% Define and run simulation
mainroad_len= 1e10
mainroad = hs.Road(1,mainroad_len, 'main road')
mainroad.connect('exit', is_exit=True)
mainroad.set_downstream({'method':'speed', 'time_series':downstream})
def newveh(vehid, *args):
    cf_p = IDM_parameters
    unused, lc_p = hs.models.IDM_parameters()
    kwargs = {'route': ['exit'], 'maxspeed':cf_p[0], 'length':5}
    return hs.Vehicle(vehid, mainroad[0], cf_p, lc_p, **kwargs)

vehicles = set()
eql_hd = tempveh.get_eql(eql_speed)
curpos = 1e5
prev_veh = None
for i in range(nveh):
    veh = newveh(i)
    veh.lead = prev_veh
    veh.initialize(curpos, eql_speed, eql_hd, 0)
    vehicles.add(veh)
    curpos += -veh.len-eql_hd
    prev_veh = veh

simulation = hs.Simulation(vehicles=vehicles, dt=dt)
start = time.time()
simulation.simulate(nt)  # number of timesteps
end = time.time()
all_vehicles = simulation.vehicles
print('simulation time is '+str(end-start)+' over '+str(sum([nt - veh.start+1 if veh.end is None else veh.end - veh.start+1
                                                         for veh in all_vehicles]))+' timesteps')

#%% basic plotting
laneinds = {mainroad[0]:0}
sim, siminfo = hp.plot_format(all_vehicles, laneinds)
hp.platoonplot(sim,None, siminfo, lane=0, opacity=0)







