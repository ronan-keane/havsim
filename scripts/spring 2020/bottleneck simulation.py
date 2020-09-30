
"""
Bottleneck simulation
"""
import havsim.simulation as hs
from havsim.simulation.road_networks import downstream_wrapper, AnchorVehicle
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
# tempveh = hs.Vehicle(-1, None, [30, 1.5, 2, 1.1, 1.5], None, maxspeed = 30-1e-6)
# spds = np.arange(0,30,.01)
# flows = np.array([tempveh.get_flow(i) for i in spds])
# density = np.divide(flows,spds)
# plt.figure()
# plt.plot(density,flows)

#%%
# done with - accident free relax, no acceleration bounds, max speed bounds
#vehicle parameters
def onramp_newveh(self, vehid, *args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route':['main road', 'exit'], 'maxspeed': cf_p[0]-1e-6, 'relax_parameters':None,
              'shift_parameters': [-1.5, 1.5]}
    self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)

def mainroad_newveh(self, vehid, *args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route':['exit'], 'maxspeed': cf_p[0]-1e-6, 'relax_parameters':None, 'shift_parameters': [-1.5, 1.5]}
    self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)
#inflow amounts
def onramp_inflow(timeind, *args):
    # return .06 + np.random.rand()/25
    return .07
def mainroad_inflow(*args):
    # return .43 + np.random.rand()*24/100
    return .48

#outflow using speed series
tempveh = hs.Vehicle(-1, None, [30, 1.5, 2, 1.1, 1.5], None, maxspeed = 30-1e-6)
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

road = {'name': 'main road', 'len': mainroadlen, 'laneinds':2, 0: None, 1: None}
road['connect to'] = {'exit': (mainroadlen, 'continue', (0,1), None, None)}
onramp = {'name': 'on ramp', 'len': endmerge-startmerge+100, 'laneinds':1, 0: None}
onramp['connect to'] = {'main road': ((startmerge,endmerge), 'merge', 0, 'l_lc', road)}
lane0 = hs.Lane(0,mainroadlen, road, 0, downstream = downstream1, increment_inflow = increment_inflow, get_inflow = get_inflow2, new_vehicle = mainroad_newveh)
lane1 = hs.Lane(0,mainroadlen, road, 1, downstream = downstream1, increment_inflow = increment_inflow, get_inflow = get_inflow2, new_vehicle = mainroad_newveh)
road[0] = lane0
road[1] = lane1
lane2 = hs.Lane(startmerge-100,endmerge,onramp,0, increment_inflow = increment_inflow_ramp, get_inflow = get_inflow1, new_vehicle = onramp_newveh)
# downstream2 = {'method':'merge', 'merge_anchor_ind':0, 'target_lane': lane1, 'self_lane':lane2, 'stopping':'ballistic'}
downstream2 = {'method': 'free merge', 'self_lane':lane2, 'stopping':'car following'}
# downstream2 = {'method': 'free merge', 'time_series':mainroad_outflow, 'stopping':'ballistic', 'self_lane':lane2}
lane2.call_downstream = downstream_wrapper(**downstream2).__get__(lane2, hs.Lane)
onramp[0] = lane2

#road 1 connect left/right and roadlen
roadlenmain = {'on ramp':0, 'main road':0}
lane0.roadlen = roadlenmain
lane1.roadlen = roadlenmain
lane0.connect_right = [(0, lane1)]
lane1.connect_left = [(0, lane0)]
lane1.connect_right.append((startmerge,lane2))
lane1.connect_right.append((endmerge,None))
#road 2 connect left/right and roadlen
roadlenonramp = {'main road':0, 'on ramp':0}
lane2.roadlen = roadlenonramp
lane2.connect_left.append((startmerge, lane1))
#anchors
lane0.anchor = AnchorVehicle(lane0, 0)
lane1.anchor = AnchorVehicle(lane1,0)
lane2.anchor = AnchorVehicle(lane2,0)
lane1.merge_anchors = [[lane1.anchor, startmerge]]
lane2.merge_anchors = [[lane2.anchor,startmerge]]
#add lane events
lane0.events = [{'event':'exit', 'pos':mainroadlen}]
lane1.events = [{'event':'update lr', 'left': None, 'right':'add','right anchor':0, 'pos':startmerge}, {'event':'update lr', 'left':None, 'right':'remove','pos':endmerge},
                {'event':'exit','pos':mainroadlen}]
lane2.events = [{'event':'update lr', 'left':'add', 'left anchor':0, 'right': None, 'pos':startmerge}]

#make simulation
merge_lanes = [lane1, lane2]
inflow_lanes = [lane0, lane1, lane2]
simulation = hs.Simulation(inflow_lanes, merge_lanes, dt = .25)

#call
timesteps = 10000
start = time.time()
simulation.simulate(timesteps)
end = time.time()

all_vehicles = simulation.prev_vehicles.copy()
all_vehicles.extend(simulation.vehicles)

print('simulation time is '+str(end-start)+' over '+str(sum([timesteps - veh.starttime+1 if veh.endtime is None else veh.endtime - veh.starttime+1
                                                         for veh in all_vehicles]))+' timesteps')

#%%
laneinds = {lane0:0, lane1:1, lane2:2}
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