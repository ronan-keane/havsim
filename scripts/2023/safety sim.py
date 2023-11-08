"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim
import havsim.plotting as hp
import matplotlib.pyplot as plt
import pickle
import multiprocessing

n_processes = 1
replications = 1
make_plots = False
save_output = True
save_crashes_only = True
save_name = 'pickle files/e94_crashes_0'


def do_simulation(verbose=False):
    simulation, my_laneinds = e94()
    my_rear_end, my_sideswipe, my_near_miss, my_vmt = 0, 0, 0, 0
    my_vehicle_lists = []
    for i in range(replications):
        all_vehicles = simulation.simulate(verbose=verbose, timesteps=100)

        for crash in simulation.crashes:
            if crash[0].crashed == 'rear end':
                my_rear_end += 1
            else:
                my_sideswipe += 1
        for veh in all_vehicles:
            my_near_miss += len(veh.near_misses)
            my_vmt += veh.posmem[-1] - veh.posmem[0]

        if save_crashes_only:
            cur = []
            for crash in simulation.crashes:
                crash_times = [veh.crash_time for veh in crash]
                t_start, t_end = min(crash_times) - 100, max(crash_times) + 5
                cur.extend(havsim.helper.add_leaders(crash, t_start, t_end))
            cur = list(set(cur))
            my_vehicle_lists.append(cur)
        else:
            my_vehicle_lists.append(all_vehicles)
        if i < replications - 1:
            simulation.reset()
    return my_rear_end, my_sideswipe, my_near_miss, my_vmt, my_vehicle_lists, my_laneinds


if __name__ == '__main__':
    args = [False for i in range(n_processes)]
    args[0] = True
    pool = multiprocessing.Pool(n_processes)
    out = pool.map(do_simulation, args)
    all_rear_end, all_sideswipe, all_near_miss, all_vmt = 0, 0, 0, 0
    all_lists = []
    for output in out:
        rear_end, sideswipe, near_miss, vmt, all_vehicle_lists, laneinds = output
        all_rear_end += rear_end
        all_sideswipe += sideswipe
        all_near_miss += near_miss
        all_vmt += vmt
        all_lists.extend(all_vehicle_lists)
    pool.close()

    print('\n-----------SUMMARY-----------')
    print('near misses: {:n}'.format(all_near_miss/replications))
    print('rear end crashes: {:n}'.format(all_rear_end/replications))
    print('sideswipe crashes: {:n}'.format(all_sideswipe/replications))
    print('vmt (miles): {:.0f}'.format(all_vmt/1609.34))

    if save_output:
        if len(all_lists) == 1:
            with open(save_name + '.pkl', 'wb') as f:
                pickle.dump([all_lists[-1], laneinds], f)
        else:
            with open(save_name+'.pkl', 'wb') as f:
                pickle.dump([all_lists, laneinds], f)
    if make_plots:
        sim, siminfo = hp.plot_format(all_lists[-1], laneinds)
        sim2, siminfo2 = hp.clip_distance(all_lists[-1], sim, (8000, 10000))

        hp.platoonplot(sim2, None, siminfo2, lane=1, opacity=0, timerange=[6000, 10000])
        # hp.platoonplot(sim2, None, siminfo2, lane=2, opacity=0)

        hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))

        hp.plotflows(sim, [[7000, 7100], [9230, 9330], [11000, 11100]], [0, 18000], 300, h=.2)

        ani = hp.animatetraj(sim2, siminfo2, usetime=list(range(2000, 4000)), spacelim=(8000, 10000), lanelim=(3, -1))
        plt.show()

