"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim.plotting as hp
import matplotlib.pyplot as plt
import time
import pickle
import multiprocessing

n_processes = 10
replications = 1
make_plots = False
save_output = True
save_crashes_only = True
save_name = 'pickle files/e94_sim_1'

simulation, laneinds = e94()
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
    print('there were roughly {:n} near misses'.format(len(simulation.near_misses) - len(simulation.crashes)))
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
        if veh in simulation.near_misses:
            simulation.near_misses.remove(veh)
    near_miss += len(simulation.near_misses)
    for veh in all_vehicles:  # vmt
        vmt += veh.posmem[-1] - veh.posmem[0]

    if i < replications - 1:
        simulation.reset()
print('\n-----------SUMMARY-----------')
print('average near misses: {:n}'.format(near_miss/replications))
print('average rear end crashes: {:n}'.format(rear_end/replications))
print('average sideswipe crashes: {:n}'.format(sideswipe/replications))
print('average vmt (miles): {:.0f}'.format(vmt/replications/1609.34))

if save_output:
    with open(save_name+'.pkl', 'wb') as f:
        pickle.dump([all_vehicles, laneinds], f)
if make_plots:
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (8000, 10000))

    hp.platoonplot(sim2, None, siminfo2, lane=1, opacity=0, timerange=[6000, 10000])
    # hp.platoonplot(sim2, None, siminfo2, lane=2, opacity=0)

    hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))

    hp.plotflows(sim, [[7000, 7100], [9230, 9330], [11000, 11100]], [0, timesteps], 300, h=.2)

    ani = hp.animatetraj(sim2, siminfo2, usetime=list(range(2000, 4000)), spacelim=(8000, 10000), lanelim=(3, -1))
    plt.show()

