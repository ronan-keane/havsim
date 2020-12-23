import havsim.simulation as hs
import havsim.plotting as hp
import time


main_road = hs.Road(num_lanes=2, length=2000, name='main road')
main_road.connect('exit', is_exit=True)
onramp = hs.Road(num_lanes=1, length=[(0,300)], name='on ramp')
onramp.merge(main_road, self_index=0, new_lane_index=1,
             self_pos=(100, 300), new_lane_pos=(1100, 1300))


main_road.set_downstream({'method':'free'})
onramp.set_downstream({'method':'free merge', 'self_lane': onramp[0]})


def veh_parameters(route):
    def newveh(self, vehid, *args):
        cf_p = [35, 1.3, 2, 1.1, 1.5]
        lc_p = [-8, -20, .6, .1, 0, .2, .1, 20, 20]
        kwargs = {'route': route.copy(), 'maxspeed': cf_p[0]-1e-6, 'relax_parameters':8.7,
                  'shift_parameters': [-2, 2], 'hdbounds':(cf_p[2]+1e-6, 1e4)}
        self.newveh = hs.Vehicle(vehid, self, cf_p, lc_p, **kwargs)
    return newveh
mainroad_newveh = veh_parameters(['exit'])
onramp_newveh = veh_parameters(['main road', 'exit'])
increment_inflow = {'method': 'seql2', 'kwargs':{'c':.8, 'eql_speed':True, 'transition':20}}
mainroad_inflow = lambda *args: .56
onramp_inflow = lambda *args: .11111

main_road.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series':mainroad_inflow}, new_vehicle=mainroad_newveh)
onramp.set_upstream(increment_inflow=increment_inflow, get_inflow={'time_series':onramp_inflow}, new_vehicle=onramp_newveh)

# fix road bugs
main_road[0].roadlen = {'main road': 0, 'on ramp': 1000}
main_road[1].roadlen = {'main road': 0, 'on ramp': 1000}
onramp[0].roadlen = {'main road': -1000, 'on ramp': 0}

simulation = hs.Simulation(roads=[main_road, onramp], dt=.25)


#%%
start = time.time()
simulation.simulate(10000)
end = time.time()

all_vehicles = simulation.prev_vehicles.copy()
all_vehicles.extend(simulation.vehicles)
print('simulation time is '+str(end-start)+' over '+str(sum([10000 - veh.start+1 if veh.end is None else veh.end - veh.start+1
                                                         for veh in all_vehicles]))+' timesteps')

#%%
laneinds = {main_road[0]:0, main_road[1]:1, onramp[0]:2}
sim, siminfo = hp.plot_format(all_vehicles, laneinds)
hp.platoonplot(sim, None, siminfo, lane = 1, opacity = 0)