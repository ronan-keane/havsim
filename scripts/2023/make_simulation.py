"""Configure simulations."""
import havsim.simulation as hs
import numpy as np

def e94():
    """Simulation of 12km length of E94 in Ann Arbor area"""
    # specify vehicle parameters
    def veh_parameters():
        s1 = min(max(np.random.normal() * 2, -1), 4)
        s2 = np.random.rand() * .12 - .03
        s3 = np.random.rand() * .4 - .1
        kwargs = {'cf_parameters': [34 + s1, 1.3 + s2, 4, 1.3 + s3, 1.6],
                  'lc_parameters': [-8, -8, .4, .05, .1, 0, .2, 10, 30], 'lc2_parameters': [-2, 2, 1, -1, 1, .2],
                  'relax_parameters': [9., 4.5, .6, 2.], 'route_parameters': [300, 500], 'accbounds': [-10, None]}
        return kwargs

    # road network
    main_road = hs.Road(num_lanes=2, length=12000, name='E94')
    main_road.connect('exit', is_exit=True)
    offramp1 = hs.Road(num_lanes=1, length=[(475, 675)], name='jackson off ramp')
    main_road.merge(offramp1, self_index=1, new_lane_index=0, self_pos=(475, 660), new_lane_pos=(475, 660))
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
    onramp1.set_downstream({'method': 'free merge', 'self_lane': onramp1[0], 'minacc': -2})
    onramp2.set_downstream({'method': 'free merge', 'self_lane': onramp2[0], 'minacc': -2})
    onramp3.set_downstream({'method': 'free merge', 'self_lane': onramp3[0], 'minacc': -2})
    onramp4.set_downstream({'method': 'free merge', 'self_lane': onramp4[0], 'minacc': -2})
    onramp5.set_downstream({'method': 'free merge', 'self_lane': onramp5[0], 'minacc': -2})

    # upstream boundary conditions
    # inflow amounts and entering speeds
    # inflow = [1530/3600/2, 529/3600, 261/3600, 414/3600, 1261/3600, 1146/3600]  # (4pm-6pm)
    inflow = [1930 / 3600 / 2, 529 / 3600, 261 / 3600, 414 / 3600, 1100 / 3600, 1100 / 3600]  # (4pm-6pm)
    main_inflow = lambda *args: (inflow[0], None)
    onramp1_inflow = lambda *args: (inflow[1], None)
    onramp2_inflow = lambda *args: (inflow[2], None)
    onramp3_inflow = lambda *args: (inflow[3], None)
    onramp4_inflow = lambda *args: (inflow[4], None)
    onramp5_inflow = lambda *args: (inflow[5], None)

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
        # MyVeh = hs.vehicles.CrashesVehicle
        MyVeh = hs.vehicles.CrashesStochasticVehicle

        def newveh(self, vehid, timeind):
            route = route_picker()
            kwargs = veh_parameters()
            self.newveh = MyVeh(vehid, self, route=route, **kwargs)

        return newveh

    main_routes = [['jackson off ramp', 'offramp 1'], ['ann arbor saline off ramp', 'offramp 2'],
                   ['state off ramp', 'offramp 3'], ['exit']]
    # main_probabilities = [.2170, .2054, .0682, .5095]
    main_probabilities = [.172, .176, .0638, .589]
    main_newveh = make_newveh(select_route(main_routes, main_probabilities))
    onramp1_routes = [['E94', 'ann arbor saline off ramp', 'offramp 2'], ['E94', 'state off ramp', 'offramp 3'],
                      ['E94', 'exit']]
    onramp1_probabilities = [.213, .077, .71]
    onramp1_newveh = make_newveh(select_route(onramp1_routes, onramp1_probabilities))
    onramp2_routes = [['E94', 'state off ramp', 'offramp 3'], ['E94', 'exit']]
    onramp2_probabilities = [.098, .9025]
    onramp2_newveh = make_newveh(select_route(onramp2_routes, onramp2_probabilities))
    onramp3_routes = [['E94', 'state off ramp', 'offramp 3'], ['E94', 'exit']]
    onramp3_probabilities = [.098, .9025]
    onramp3_newveh = make_newveh(select_route(onramp3_routes, onramp3_probabilities))
    onramp4_newveh = make_newveh(lambda: ['E94', 'exit'])
    onramp5_newveh = make_newveh(lambda: ['E94', 'exit'])
    # define set_upstream method
    main_road.set_upstream(increment_inflow={'method': 'seql', 'kwargs': {'c': .8}},
                           get_inflow={'time_series': main_inflow, 'inflow_type': 'flow speed'},
                           new_vehicle=main_newveh)
    # increment_inflow = {'method': 'speed', 'kwargs': {'speed_series': lambda *args: 15., 'accel_bound': -1}}
    increment_inflow = {'method': 'seql', 'kwargs': {'c': .9, 'eql_speed': True}}
    onramp1.set_upstream(increment_inflow=increment_inflow,
                         get_inflow={'time_series': onramp1_inflow, 'inflow_type': 'flow speed'},
                         new_vehicle=onramp1_newveh)
    onramp2.set_upstream(increment_inflow=increment_inflow,
                         get_inflow={'time_series': onramp2_inflow, 'inflow_type': 'flow speed'},
                         new_vehicle=onramp2_newveh)
    onramp3.set_upstream(increment_inflow=increment_inflow,
                         get_inflow={'time_series': onramp3_inflow, 'inflow_type': 'flow speed'},
                         new_vehicle=onramp3_newveh)
    onramp4.set_upstream(increment_inflow=increment_inflow,
                         get_inflow={'time_series': onramp4_inflow, 'inflow_type': 'flow speed'},
                         new_vehicle=onramp4_newveh)
    onramp5.set_upstream(increment_inflow=increment_inflow,
                         get_inflow={'time_series': onramp5_inflow, 'inflow_type': 'flow speed'},
                         new_vehicle=onramp5_newveh)

    simulation = hs.simulation.CrashesSimulation(
        roads=[main_road, onramp1, onramp2, onramp3, onramp4, onramp5, offramp1, offramp2, offramp3], dt=.2)
    laneinds = {main_road[0]: 0, main_road[1]: 1, onramp1[0]: 2, onramp2[0]: 2, onramp3[0]: 2, onramp4[0]: 2,
                onramp5[0]: 2, offramp1[0]: 2, offramp2[0]: 2, offramp3[0]: 2}

    return simulation, laneinds


def merge_bottleneck(main_inflow=None, onramp_inflow=None):
    """Test simulation of merge bottleneck."""
    main_road = hs.Road(num_lanes=2, length=2000, name='main road')
    main_road.connect('exit', is_exit=True)
    onramp = hs.Road(num_lanes=1, length=[(950, 1300)], name='on ramp')
    onramp.merge(main_road, self_index=0, new_lane_index=1, self_pos=(1050, 1300), new_lane_pos=(1050, 1300))

    main_road.set_downstream({'method': 'free'})
    onramp.set_downstream({'method': 'free merge', 'self_lane': onramp[0]})

    def veh_parameters(route):
        def newveh(self, vehid, timeind):
            s1 = min(max(np.random.normal()*1.5 + 1, -4), 3)
            s2 = np.random.rand() * .12 - .03
            s3 = np.random.rand()*.3-.2
            kwargs = {'cf_parameters': [34, 1.2, 3, 1.4, 1.6],
                      'lc_parameters': [-8, -8, .4, .05, .1, 0, .2, 10, 30], 'lc2_parameters': [-2, 2, 1, -1.5, 1, .2],
                      'relax_parameters': [9., 4.5, .6, 2.], 'route_parameters': [300, 500], 'accbounds': [-10, None],
                      'route': route.copy()}
            self.newveh = hs.Vehicle(vehid, self, **kwargs)
        return newveh

    mainroad_newveh = veh_parameters(['exit'])
    onramp_newveh = veh_parameters(['main road', 'exit'])
    increment_inflow = {'method': 'seql2', 'kwargs': {'c': .8, 'eql_speed': True, 'transition': 20}}
    if main_inflow is None:
        main_inflow = lambda *args: .56
    if onramp_inflow is None:
        onramp_inflow = lambda *args: .11111

    main_road.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': main_inflow},
                           new_vehicle=mainroad_newveh)
    onramp.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series': onramp_inflow},
                        new_vehicle=onramp_newveh)
    simulation = hs.Simulation(roads=[main_road, onramp], dt=.2)
    laneinds = {main_road[0]: 0, main_road[1]: 1, onramp[0]: 2}

    return simulation, laneinds
