
"""
Diverge network simulation
"""

import havsim.simulation as hs
from havsim.simulation.road import Road
from havsim.plotting import plot_format, platoonplot, plotvhd
from havsim.simulation.models import IDM_parameters
import time


def road1_inflow(*args):
    # return .43 + np.random.rand()*24/100
    return .48

#outflow using speed series
tempveh = hs.Vehicle(-1, None, [30, 1.1, 3, 1.1, 1.5], None, maxspeed = 30-1e-6)
outspeed = tempveh.inv_flow(.48, congested = False)
inspeed, inhd = tempveh.inv_flow(.48, output_type = 'both', congested = True)

def road1_outflow(*args):
    return outspeed

#define boundary conditions
get_inflow = {'time_series':road1_inflow}
increment_inflow = {'method': 'seql', 'kwargs':{'c':.8}}

road1len = 2000
road2len = 1500
road3len = 1500
mergelen = 200

# Construct the road network
road1 = Road(num_lanes=3, length=[(0, road1len), (0, road1len), (0, road1len - mergelen)], name='road 1')
road2 = Road(num_lanes=2, length=road2len, name='road 2')
road1.connect(road2, self_indices=[0, 1], new_road_indices=[0, 1])
road3 = Road(num_lanes=2, length=road3len, name='road 3')
road3.connect('exit', is_exit=True)
road1.merge(road3, self_index=1, new_lane_index=0,
            self_pos=(road1len - mergelen, road1len), new_lane_pos=(0, mergelen))
road1.connect(road3, self_indices=[2], new_road_indices=[1])


def road1_newveh(self, vehid, *args):
    cf_p, lc_p  = IDM_parameters()
    kwargs = {'route': ['road 3','exit'],
              'maxspeed': cf_p[0]-1e-6, 'relax_parameters':15, 'shift_parameters': [-1.5, 1.5]}
    self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)

# Set boundary conditions
downstream1 ={'method':'free', }
road1.set_downstream(downstream1)
road1.set_upstream(increment_inflow=increment_inflow, get_inflow=get_inflow, new_vehicle=road1_newveh)

# Make simulation
simulation = hs.Simulation(roads=[road1, road2, road3], dt = .25)

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
# laneinds = {road1[0]:0, road1[1]:1, road3[0]:2}
# sim, siminfo = plot_format(all_vehicles, laneinds)
#
# mylane2list = []
# for veh in sim.keys():
#     if 2 in sim[veh][:,7]:
#         mylane2list.append(veh)
# #%%
# platoonplot(sim, None, siminfo, lane = 2, opacity = 0)
# platoonplot(sim, None, siminfo, lane = 1, opacity = 0)
# platoonplot(sim, None, siminfo, lane = 0, opacity = 0)
# platoonplot(sim, None, siminfo, lane = 2, colorcode = False)
# platoonplot(sim, None, siminfo, lane = 1, colorcode = False)
# %%
# plotspacetime(sim, siminfo, lane = 2)
# plotspacetime(sim, siminfo, lane = 1)
# plotspacetime(sim, siminfo, lane = 0)
