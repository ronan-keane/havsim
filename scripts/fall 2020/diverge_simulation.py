"""
Diverge network simulation
"""

import havsim.simulation as hs
import havsim.plotting as hp
import time
import numpy as np

# create road network
road1len = 1000
road2len = 1000
road3len = 1000
mergelen = 200
road1 = hs.Road(num_lanes=3, length=[(0, road1len), (0, road1len), (0, road1len - mergelen)], name='road 1')
road2 = hs.Road(num_lanes=2, length=road2len, name='road 2')
road3 = hs.Road(num_lanes=2, length=road3len+mergelen, name='road 3')
road1.connect(road2, self_indices=[0, 1], new_road_indices=[0, 1])
road1.merge(road3, self_index=1, new_lane_index=0,
            self_pos=(road1len - mergelen, road1len), new_lane_pos=(0, mergelen))
road1.connect(road3, self_indices=[2], new_road_indices=[1])
road2.connect('exit', is_exit=True)
road3.connect('exit 2', is_exit=True)

road2.set_downstream({'method':'free'})
road3.set_downstream({'method':'free'})

# manually fixing bugs in road network
# events
road1[1].events = [{'event': 'update lr', 'left': None, 'right': 'add', 'right anchor': 0, 'pos': 800},
                   {'event': 'new lane', 'pos': 1000, 'left': 'update', 'right': 'remove'}]
road1[2].events = [{'event': 'new lane', 'pos': 800, 'left': 'add', 'left anchor': 0, 'right': None}]
# roadlen
road1[0].roadlen = {'road 1': 0, 'road 2': 1000, 'road 3': 800}
road1[1].roadlen = {'road 1': 0, 'road 2': 1000, 'road 3': 800}
road1[2].roadlen = {'road 1': 0, 'road 2': 1000, 'road 3': 800}
road2[0].roadlen = {'road 1': -1000, 'road 2': 0, 'road 3': -200}
road2[1].roadlen = {'road 1': -1000, 'road 2': 0, 'road 3': -200}
road3[0].roadlen = {'road 1': -800, 'road 2': 200, 'road 3': 0}
road3[1].roadlen = {'road 1': -800, 'road 2': 200, 'road 3': 0}
# connect_left/right
road1[1].connect_right = [(0, road1[2]), (800, road3[0])]


def newveh_wrapper(split_ratio):  # split ratio is what % of vehicles go left
    def mainroad_newveh(self, vehid, *args):
        if np.random.rand() < split_ratio:
            route = ['road 2', 'exit']
        else:
            route = ['road 3', 'exit 2']
        cf_p = [35, 1.3, 2, 1.1, 1.5]
        lc_p = [-8, -20, .6, .1, 0, .2, .1, 20, 20]
        kwargs = {'route': route, 'maxspeed': cf_p[0]-1e-6, 'relax_parameters':8.7,
                  'shift_parameters': [-2, 2], 'hdbounds':(cf_p[2]+1e-6, 1e4)}
        self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)
    return mainroad_newveh
# boundary conditions
increment_inflow = {'method': 'seql2', 'kwargs':{'c':.8, 'eql_speed':True, 'transition':19}}
mainroad_inflow = lambda *args: .65
road1.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': mainroad_inflow}, new_vehicle=newveh_wrapper(.95), self_indices=[0])
road1.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': mainroad_inflow}, new_vehicle=newveh_wrapper(.5), self_indices=[1])
road1.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': mainroad_inflow}, new_vehicle=newveh_wrapper(.05), self_indices=[2])

# Make simulation
simulation = hs.Simulation(roads=[road1, road2, road3], dt = .25)
#%%
timesteps=10000
start = time.time()
simulation.simulate(timesteps)
end = time.time()

all_vehicles = simulation.prev_vehicles.copy()
all_vehicles.extend(simulation.vehicles)
print('simulation time is '+str(end-start)+' over '+str(sum([timesteps - veh.start+1 if veh.end is None else veh.end - veh.start+1
                                                         for veh in all_vehicles]))+' timesteps')

#%%
# laneinds = {road1[0]:0, road1[1]:1, road1[2]:2, road2[0]:3, road2[1]:4, road3[0]:5, road3[1]:6}
# sim, siminfo = hp.plot_format(all_vehicles, laneinds)
# hp.platoonplot(sim, None, siminfo, lane = 5, opacity = 0, colorcode=False)
