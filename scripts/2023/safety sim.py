"""Simulation of 12km length of E94 in Ann Arbor area"""

import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt
import numpy as np
import time

# specify vehicle parameters
def veh_parameters():
    cf_p = [35, 1.3, 2, 1.1, 1.5]
    lc_p = [-8, -20, .6, .1, 0, .2, .1, 20, 20]
    kwargs = {'relax_parameters': 8.7, 'shift_parameters': [-2, 2], 'coop_parameters': 0.2,
              'route_parameters': [200, 200],
              'accbounds': [-8, None], 'maxspeed': cf_p[0]-1e-6, 'hdbounds': (cf_p[2]+1e-6, 1e4)}
    return cf_p, lc_p, kwargs

# road network
main_road = hs.Road(num_lanes=2, length=12000, name='E94')
main_road.connect('exit', is_exit=True)
offramp1 = hs.Road(num_lanes=1, length=100, name='jackson off ramp')
offramp1.merge(main_road, self_index=0, new_lane_index=1, self_pos=(0, 45), new_lane_pos=(175, 220))
offramp1.connect('offramp 1', is_exit=True)
onramp1 = hs.Road(num_lanes=1, length=250, name='jackson on ramp')
onramp1.merge(main_road, self_index=0, new_lane_index=1, self_pos=(100, 250), new_lane_pos=(1200, 1350))
offramp2 = hs.Road(num_lanes=1, length=200, name='ann arbor saline off ramp')
offramp2.merge(main_road, self_index=0, new_lane_index=1, self_pos=(0, 150), new_lane_pos=(5330, 5480))
offramp2.connect('offramp 2', is_exit=True)
onramp2 = hs.Road(num_lanes=1, length=280, name='ann arbor saline on ramp SW')
onramp2.merge(main_road, self_index=0, new_lane_index=1, self_pos=(100, 280), new_lane_pos=(6150, 6330))
onramp3 = hs.Road(num_lanes=1, length=300, name='ann arbor saline on ramp NE')
onramp3.merge(main_road, self_index=0, new_lane_index=1, self_pos=(200, 300), new_lane_pos=(6710, 6810))
offramp3 = hs.Road(num_lanes=1, length=180, name='state off ramp')
offramp3.merge(main_road, self_index=0, new_lane_index=1, self_pos=(0, 130), new_lane_pos=(7810, 7940))
offramp3.connect('offramp 3', is_exit=True)
onramp4 = hs.Road(num_lanes=1, length=300, name='state on ramp S')
onramp4.merge(main_road, self_index=0, new_lane_index=1, self_pos=(100, 300), new_lane_pos=(8510, 8710))
onramp5 = hs.Road(num_lanes=1, length=300, name='state on ramp N')
onramp5.merge(main_road, self_index=0, new_lane_index=1, self_pos=(200, 300), new_lane_pos=(9130, 9230))

# downstream boundary conditions
main_road.set_downstream({'method': 'free'})
onramp1.set_downstream({'method': 'free merge', 'self_lane': onramp1[0], 'minacc': -2.5})
onramp2.set_downstream({'method': 'free merge', 'self_lane': onramp2[0], 'minacc': -2.5})
onramp3.set_downstream({'method': 'free merge', 'self_lane': onramp3[0], 'minacc': -2.5})
onramp4.set_downstream({'method': 'free merge', 'self_lane': onramp4[0], 'minacc': -2.5})
onramp5.set_downstream({'method': 'free merge', 'self_lane': onramp5[0], 'minacc': -2.5})

# upstream boundary conditions
# inflow amounts and entering speeds
main_inflow = lambda *args: 1530/3600/2, None
onramp1_inflow = lambda *args: 529/3600, 13
onramp2_inflow = lambda *args: 261/3600, 13
onramp3_inflow = lambda *args: 414/3600, 25
onramp4_inflow = lambda *args: 1261/3600, 13
onramp5_inflow = lambda *args: 1146/3600, 25
# define the routes of vehicles
def select_route(routes, probabilities):
    p = np.cumsum(probabilities)
    rng = np.random.default_rng()
    def make_route():
        rand = rng.random()
        ind = (rand < p).nonzero()[0][0]
        return routes[ind].copy()
    return make_route

def make_newveh(route_picker):
    def newveh(self, vehid):
        route = route_picker()
        cf_p, lc_p, kwargs = veh_parameters()
        self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, route=route, **kwargs)
    return newveh

main_routes = [['jackson off ramp', 'offramp 1'], ['ann arbor saline off ramp', 'offramp 2'], ['state off ramp', 'offramp 3'], ['exit']]
main_probabilities = [.2170, .2054, .0682, .5095]
main_newveh = make_newveh(select_route(main_routes, main_probabilities))
onramp1_routes = [['E94', 'ann arbor saline off ramp', 'offramp 2'], ['E94', 'state off ramp', 'offramp 3'], ['E94', 'exit']]
onramp1_probabilities = [.2623, .0871, .651]
onramp1_newveh = make_newveh(select_route(onramp1_routes, onramp1_probabilities))
onramp2_routes = [['E94', 'state off ramp', 'offramp 3'], ['E94', 'exit']]
onramp2_probabilities = [.118, .882]
onramp2_newveh = make_newveh(select_route(onramp2_routes, onramp2_probabilities))
onramp3_routes = [['E94', 'state off ramp', 'offramp 3'], ['E94', 'exit']]
onramp3_probabilities = [.118, .882]
onramp3_newveh = make_newveh(select_route(onramp3_routes, onramp3_probabilities))
onramp4_newveh = make_newveh(lambda: ['E94', 'exit'])
onramp5_newveh = make_newveh(lambda: ['E94', 'exit'])
# define set_upstream method
main_road.set_upstream(increment_inflow={'method': 'seql'}, get_inflow={'time_series': main_inflow, 'inflow_type': 'flow speed'}, new_vehicle=main_newveh)
onramp1.set_upstream(increment_inflow={'method': 'seql'}, get_inflow={'time_series': onramp1_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp1_newveh)
onramp2.set_upstream(increment_inflow={'method': 'seql'}, get_inflow={'time_series': onramp2_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp2_newveh)
onramp3.set_upstream(increment_inflow={'method': 'seql'}, get_inflow={'time_series': onramp3_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp3_newveh)
onramp4.set_upstream(increment_inflow={'method': 'seql'}, get_inflow={'time_series': onramp4_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp4_newveh)
onramp5.set_upstream(increment_inflow={'method': 'seql'}, get_inflow={'time_series': onramp5_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp5_newveh)

simulation = hs.Simulation(roads=[main_road, onramp1, onramp2, onramp3, onramp4, onramp5, offramp1, offramp2, offramp3], dt=.25)

start = time.time()
simulation.simulate(4000)
end = time.time()



