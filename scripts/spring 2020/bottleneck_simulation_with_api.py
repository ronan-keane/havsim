
"""
Bottleneck simulation
"""
import havsim.simulation as hs
from havsim.simulation.road_networks import downstream_wrapper, AnchorVehicle, get_inflow_wrapper, Lane, increment_inflow_wrapper
from havsim.simulation.road import Road
from havsim.helper import boundaryspeeds, getentryflows, calculateflows
from havsim.plotting import plot_format, platoonplot, plotvhd
import numpy as np
import matplotlib.pyplot as plt
from havsim.simulation.models import IDM_parameters
import time
#%%get boundary conditions (careful with units)
# #option 1 -
# #could get them directly from data
# entryflows, unused = getentryflows(meas, [3],.1,.25)
# unused, unused, exitspeeds, unused = boundaryspeeds(meas, [], [3],.1,.1)

# #option 2 - use calculateflows, which has some aggregation in it and uses a different method to compute flows
# q,k = calculateflows(meas, [[200,600],[1000,1400]], [0, 9900], 30*10, lane = 6)

#option 3 - can also just make boudnary conditions based on what the FD looks like
# tempveh = hs.Vehicle(-1, None, [30, 1.1, 3, 1.1, 1.5], None, maxspeed = 30-1e-6)
# spds = np.arange(0,33.3,.01)
# flows = np.array([tempveh.get_flow(i) for i in spds])
# density = np.divide(flows,spds)
# plt.figure()
# plt.plot(density,flows)

#%%
# done with - accident free relax, no acceleration bounds, max speed bounds
#vehicle parameters
def onramp_newveh(self, vehid, *args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route':['main road', 'exit'], 'maxspeed': cf_p[0]-1e-6, 'relax_parameters':15,
              'shift_parameters': [-1.5, 1.5]}
    self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)

def mainroad_newveh(self, vehid, *args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route':['exit'], 'maxspeed': cf_p[0]-1e-6, 'relax_parameters':15, 'shift_parameters': [-1.5, 1.5]}
    self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)
#inflow amounts
def onramp_inflow(timeind, *args):
    # return .06 + np.random.rand()/25
    return .1
def mainroad_inflow(*args):
    # return .43 + np.random.rand()*24/100
    return .48

#outflow using speed series
tempveh = hs.Vehicle(-1, None, [30, 1.1, 3, 1.1, 1.5], None, maxspeed = 30-1e-6)
outspeed = tempveh.inv_flow(.48, congested = False)
inspeed, inhd = tempveh.inv_flow(.48, output_type = 'both', congested = True)
inspeedramp, inhd = tempveh.inv_flow(.07, output_type = 'both', congested = True)
def mainroad_outflow(*args):
    return outspeed

def speed_inflow(*args):
    return inspeed

def speed_inflow_ramp(*args):
    return inspeedramp

#define boundary conditions
get_inflow1 = {'time_series':onramp_inflow}
get_inflow2 = {'time_series':mainroad_inflow}
# increment_inflow = {'method': 'ceql'}
increment_inflow = {'method': 'seql', 'kwargs':{'c':.8}}
# increment_inflow = {'method': 'shifted', 'accel_bound':-.3, 'shift':1.5}
# increment_inflow = {'method': 'speed', 'accel_bound':-.1, 'speed_series':speed_inflow}
# increment_inflow_ramp = {'method': 'speed', 'accel_bound':-.1, 'speed_series':speed_inflow_ramp}
increment_inflow_ramp=increment_inflow
downstream1 ={'method':'free', }
# downstream1 = {'method': 'speed', 'time_series':mainroad_outflow}

#make road network with boundary conditions - want to make an api for this in the future
#main road has len mainroadlen, on ramp connects to right lane of main road on (startmerge, endmerge),
#onramplen has onramplen before reaching the merge section
mainroadlen = 2000
startmerge = 1100
endmerge = 1300
onramplen = 200

main_road = Road(num_lanes=2, length=mainroadlen, name='main road')
main_road.connects('exit', is_exit=True)
onramp_road = Road(num_lanes=1, length=[(startmerge - 100, endmerge)], name='on ramp')
onramp_road[0].connects(main_road[1], connect_type='merge',
                        self_pos=(startmerge, endmerge),
                        new_lane_pos=(startmerge, endmerge))
for i in range(2):
    lane = main_road[i]
    lane.call_downstream = downstream_wrapper(**downstream1).__get__(lane, Lane)
    lane.get_inflow = get_inflow_wrapper(**get_inflow2).__get__(lane, Lane)
    lane.inflow_buffer = 0
    lane.newveh = None
    lane.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(lane, Lane)
    lane.new_vehicle = mainroad_newveh.__get__(lane, Lane)

onramp_lane = onramp_road[0]
onramp_lane.get_inflow = get_inflow_wrapper(**get_inflow1).__get__(onramp_lane, Lane)
onramp_lane.inflow_buffer = 0
onramp_lane.newveh = None
onramp_lane.increment_inflow = increment_inflow_wrapper(**increment_inflow_ramp).__get__(onramp_lane, Lane)
onramp_lane.new_vehicle = onramp_newveh.__get__(onramp_lane, Lane)
# downstream2 = {'method':'merge', 'merge_anchor_ind':0, 'target_lane': lane1, 'self_lane':lane2, 'stopping':'ballistic'}
downstream2 = {'method': 'free merge', 'self_lane':onramp_lane, 'stopping':'car following'}
onramp_lane.call_downstream = downstream_wrapper(**downstream2).__get__(onramp_lane, Lane)

#make simulation
merge_lanes = [main_road[1], onramp_road[0]]
inflow_lanes = [main_road[0], main_road[1], onramp_road[0]]
simulation = hs.Simulation(inflow_lanes, merge_lanes, dt = .25)

#call
timesteps = 10000
start = time.time()
simulation.simulate(timesteps)
end = time.time()

all_vehicles = simulation.prev_vehicles.copy()
all_vehicles.extend(simulation.vehicles)

print('simulation time is '+str(end-start)+' over '+str(sum([timesteps - veh.start+1 if veh.end is None else veh.end - veh.start+1
                                                         for veh in all_vehicles]))+' timesteps')

#%%
laneinds = {main_road[0]:0, main_road[1]:1, onramp_road[0]:2}
sim, siminfo = plot_format(all_vehicles, laneinds)

mylane2list = []
for veh in sim.keys():
    if 2 in sim[veh][:,7]:
        mylane2list.append(veh)
#%%
platoonplot(sim, None, siminfo, lane = 2, opacity = 0)
platoonplot(sim, None, siminfo, lane = 1, opacity = 0)
# platoonplot(sim, None, siminfo, lane = 0, opacity = 0)
# platoonplot(sim, None, siminfo, lane = 2, colorcode = False)
# platoonplot(sim, None, siminfo, lane = 1, colorcode = False)
# %%
# plotspacetime(sim, siminfo, lane = 2)
# plotspacetime(sim, siminfo, lane = 1)
# plotspacetime(sim, siminfo, lane = 0)