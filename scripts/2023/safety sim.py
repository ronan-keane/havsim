from make_simulation import e94
import havsim.plotting as hp
import matplotlib.pyplot as plt
import time
import pickle

simulation, laneinds = e94()
timesteps = 3600*6
replications = 1
make_plots = True
save_output = True
save_name = 'e94_sim_0'

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
        if crash_time - 8 in lc_times:
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

if make_plots or save_output:
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (8000, 10000))
    if save_output:
        with open(save_name+'.pkl', 'wb') as f:
            pickle.dump([sim, siminfo, sim2, siminfo2], f)
    if make_plots:
        hp.platoonplot(sim2, None, siminfo2, lane=1, opacity=0, timerange=[8000, 10000])
        # hp.platoonplot(sim2, None, siminfo2, lane=2, opacity=0)

        hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))
        hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=0, speed_bounds=(0, 35))

        hp.plotflows(sim, [[7000, 7100], [9230, 9330], [11000, 11100]], [0, timesteps], 300, h=.2)

        ani = hp.animatetraj(sim2, siminfo2, usetime=list(range(2000, 4000)), show_id=False, spacelim=(8000, 10000), lanelim=(3, -1))
        ani2 = hp.animatetraj(sim2, siminfo2, usetime=list(range(8000, 10000)), show_id=False, spacelim=(8000, 10000),
                             lanelim=(3, -1))
        plt.show()

