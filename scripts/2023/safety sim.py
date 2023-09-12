"""Simulation of 12km length of E94 in Ann Arbor area"""

import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt
import time

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

# onramp 1, 2, 4 need lower speed for initial vehicles
main_road.set_downstream({'method': 'free'})
onramp1.set_downstream({'method': 'free merge', 'self_lane': onramp1[0], 'minacc': -2})
onramp2.set_downstream({'method': 'free merge', 'self_lane': onramp2[0], 'minacc': -2})
onramp3.set_downstream({'method': 'free merge', 'self_lane': onramp3[0], 'minacc': -2})
onramp4.set_downstream({'method': 'free merge', 'self_lane': onramp4[0], 'minacc': -2})
onramp5.set_downstream({'method': 'free merge', 'self_lane': onramp5[0], 'minacc': -2})

