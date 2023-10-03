"""Simulation of 12km length of E94 in Ann Arbor area"""

import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt
import numpy as np
import time

# specify vehicle parameters
def veh_parameters():
    s1 = np.random.rand()*6-4
    s2 = np.random.rand()*.3-.2
    cf_p = [35+s1, 1.3+s2, 2, 1.1, 1.5]
    lc_p = [-4, -8, .3, .15, 0., 0., .45, .2, 10, 42]
    kwargs = {'relax_parameters': 8.7, 'relaxs_parameters': [0.1, 1.5],
              'shift_parameters': [-3, 2, -3], 'coop_parameters': 0.2, 'route_parameters': [300, 500],
              'accbounds': [-10, None], 'maxspeed': cf_p[0]-1e-6, 'hdbounds': (cf_p[2]+1e-6, 1e4)}
    return cf_p, lc_p, kwargs

# road network
main_road = hs.Road(num_lanes=2, length=12000, name='E94')
main_road.connect('exit', is_exit=True)
offramp1 = hs.Road(num_lanes=1, length=[(175, 275)], name='jackson off ramp')
main_road.merge(offramp1, self_index=1, new_lane_index=0, self_pos=(175, 260), new_lane_pos=(175, 260))
offramp1.connect('offramp 1', is_exit=True)
onramp1 = hs.Road(num_lanes=1, length=[(1000, 1350)], name='jackson on ramp')
onramp1.merge(main_road, self_index=0, new_lane_index=1, self_pos=(1100, 1350), new_lane_pos=(1100, 1350))
offramp2 = hs.Road(num_lanes=1, length=[(5330, 5530)], name='ann arbor saline off ramp')
main_road.merge(offramp2, self_index=1, new_lane_index=0, self_pos=(5330, 5480), new_lane_pos=(5330, 5480))
offramp2.connect('offramp 2', is_exit=True)
onramp2 = hs.Road(num_lanes=1, length=[(5950, 6330)], name='ann arbor saline on ramp SW')
onramp2.merge(main_road, self_index=0, new_lane_index=1, self_pos=(6050, 6330), new_lane_pos=(6050, 6330))
onramp3 = hs.Road(num_lanes=1, length=[(6410, 6810)], name='ann arbor saline on ramp NE')
onramp3.merge(main_road, self_index=0, new_lane_index=1, self_pos=(6610, 6810), new_lane_pos=(6610, 6810))
offramp3 = hs.Road(num_lanes=1, length=[(7810, 7990)], name='state off ramp')
main_road.merge(offramp3, self_index=1, new_lane_index=0, self_pos=(7810, 7940), new_lane_pos=(7810, 7940))
offramp3.connect('offramp 3', is_exit=True)
onramp4 = hs.Road(num_lanes=1, length=[(8310, 8710)], name='state on ramp S')
onramp4.merge(main_road, self_index=0, new_lane_index=1, self_pos=(8410, 8710), new_lane_pos=(8410, 8710))
onramp5 = hs.Road(num_lanes=1, length=[(8830, 9230)], name='state on ramp N')
onramp5.merge(main_road, self_index=0, new_lane_index=1, self_pos=(8980, 9230), new_lane_pos=(8980, 9230))

# downstream boundary conditions
main_road.set_downstream({'method': 'free'})
offramp1.set_downstream({'method': 'free'})
offramp2.set_downstream({'method': 'free'})
offramp3.set_downstream({'method': 'free'})
onramp1.set_downstream({'method': 'free merge', 'self_lane': onramp1[0], 'minacc': -5})
onramp2.set_downstream({'method': 'free merge', 'self_lane': onramp2[0], 'minacc': -5})
onramp3.set_downstream({'method': 'free merge', 'self_lane': onramp3[0], 'minacc': -5})
onramp4.set_downstream({'method': 'free merge', 'self_lane': onramp4[0], 'minacc': -5})
onramp5.set_downstream({'method': 'free merge', 'self_lane': onramp5[0], 'minacc': -5})

# upstream boundary conditions
# inflow amounts and entering speeds
# inflow = [1530/3600/2, 529/3600, 261/3600, 414/3600, 1261/3600, 1146/3600]  # (4pm-6pm)
inflow = [2060/3600/2, 529/3600, 261/3600, 414/3600, 1260/3600, 1146/3600]  # (4pm-6pm)
# inflow = np.array(inflow)
# inflow[0] = inflow[0] * .863
# inflow[1:] = inflow[1:] * .382
main_inflow = lambda *args: (inflow[0], None)
onramp1_inflow = lambda *args: (inflow[1], 10)
onramp2_inflow = lambda *args: (inflow[2], 10)
onramp3_inflow = lambda *args: (inflow[3], 10)
onramp4_inflow = lambda *args: (inflow[4], 10)
onramp5_inflow = lambda *args: (inflow[5], 10)
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
    MyVeh = hs.vehicles.add_crash_behavior(hs.Vehicle)
    def newveh(self, vehid):
        route = route_picker()
        cf_p, lc_p, kwargs = veh_parameters()
        self.newveh = MyVeh(vehid, self, cf_p, lc_p, route=route, **kwargs)
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
main_road.set_upstream(increment_inflow={'method': 'seql', 'kwargs': {'c': .8}}, get_inflow={'time_series': main_inflow, 'inflow_type': 'flow speed'}, new_vehicle=main_newveh)
increment_inflow = {'method': 'speed', 'kwargs': {'speed_series': lambda *args: 5., 'accel_bound': -2}}
# increment_inflow = {'method': 'seql', 'kwargs': {'c': .8, 'eql_speed': True}}
onramp1.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': onramp1_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp1_newveh)
onramp2.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': onramp2_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp2_newveh)
onramp3.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': onramp3_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp3_newveh)
onramp4.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': onramp4_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp4_newveh)
onramp5.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': onramp5_inflow, 'inflow_type': 'flow speed'}, new_vehicle=onramp5_newveh)


simulation = hs.simulation.CrashesSimulation(roads=[main_road, onramp1, onramp2, onramp3, onramp4, onramp5, offramp1, offramp2, offramp3], dt=.2)

timesteps = 3600*10
replications = 1
near_miss = 0
rear_end = 0
sideswipe = 0
vmt = 0
for i in range(replications):
    start = time.time()
    simulation.simulate(timesteps)
    end = time.time()

    all_vehicles = simulation.prev_vehicles
    all_vehicles.extend(simulation.vehicles)
    print('simulation time is '+str(end-start)+' over '+str(sum([timesteps - veh.start+1 if veh.end is None else veh.end - veh.start+1
                                                                 for veh in all_vehicles]))+' timesteps')
    print('there were {:n} crashes involving {:n} vehicles'.format(len(simulation.crashes), len(simulation.crashed_veh)))
    print('there were roughly {:n} near misses'.format(len(simulation.near_miss_veh)-len(simulation.crashes)))
    for crash in simulation.crashes:  # determine whether it's due to sideswipe or rear end
        # check the first two vehicles only
        crash_time = crash[0].crash_time
        if len(crash[0].lanemem) > 1:
            lc_times = [lc[1] for lc in crash[0].lanemem[1:]]
        else:
            lc_times = []
        if len(crash[1].lanemem) > 1:
            lc_times.extend([lc[1] for lc in crash[1].lanemem[1:]])
        if crash_time - 6 in lc_times:
            sideswipe += 1
        else:
            rear_end += 1
    for veh in simulation.crashed_veh:  # count near misses
        if veh in simulation.near_miss_veh:
            simulation.near_miss_veh.remove(veh)
    near_miss += len(simulation.near_miss_veh)
    for veh in all_vehicles:  # vmt
        vmt += veh.posmem[-1] - veh.posmem[0]

    if i < replications - 1:
        simulation.reset()
print('\n-----------SUMMARY-----------')
print('average near misses: {:n}'.format(near_miss/replications))
print('average rear end crashes: {:n}'.format(rear_end/replications))
print('average sideswipe crashes: {:n}'.format(sideswipe/replications))
print('average vmt (miles): {:.0f}'.format(vmt/replications/1609.34))
