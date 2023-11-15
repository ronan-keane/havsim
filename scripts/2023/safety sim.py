"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim
import pickle
import multiprocessing

n_processes = 40
replications = 4
save_output = True
save_crashes_only = True
save_name = 'pickle files/e94_crashes_3'


def do_simulation(verbose=False):
    simulation, my_laneinds = e94()
    my_rear_end, my_sideswipe, my_near_miss, my_vmt = 0, 0, 0, 0
    my_out_lists = []
    for i in range(replications):
        all_vehicles = simulation.simulate(verbose=verbose)

        # count statistics
        for crash in simulation.crashes:
            if crash[0].crashed[0] == 'rear end':
                my_rear_end += 1
            else:
                my_sideswipe += 1
        for veh in all_vehicles:
            my_near_miss += len(veh.near_misses)
            my_vmt += veh.posmem[-1] - veh.posmem[0]

        # save vehicles
        if save_crashes_only:
            cur = []
            for crash in simulation.crashes:
                crash_times = [veh.crash_time for veh in crash]
                t_start, t_end = min(crash_times) - 100, max(crash_times) + 5
                cur.extend(havsim.helper.add_leaders(crash, t_start, t_end))
            cur = list(set(cur))
            for veh in all_vehicles:
                if len(veh.near_misses) > 0:
                    for times in veh.near_misses:
                        t_start, t_end = times[0] - 100, times[1] + 5
                        cur.extend(havsim.helper.add_leaders([veh], t_start, t_end))
            cur = list(set(cur))
            my_out_lists.append(pickle.dumps(cur))
        else:
            my_out_lists.append(pickle.dumps(all_vehicles))

        if i < replications - 1:
            simulation.reset()
            del all_vehicles
            del veh
            if save_crashes_only:
                del cur
    return my_rear_end, my_sideswipe, my_near_miss, my_vmt, my_out_lists, my_laneinds


if __name__ == '__main__':
    args = [False for i in range(n_processes)]
    args[0] = True
    pool = multiprocessing.Pool(n_processes)
    out = pool.map(do_simulation, args)
    all_rear_end, all_sideswipe, all_near_miss, all_vmt = 0, 0, 0, 0
    all_lists = []
    for output in out:
        rear_end, sideswipe, near_miss, vmt, all_out_lists, laneinds = output
        all_rear_end += rear_end
        all_sideswipe += sideswipe
        all_near_miss += near_miss
        all_vmt += vmt
        all_lists.extend([pickle.loads(out) for out in all_out_lists])
    pool.close()

    print('\n-----------SUMMARY-----------')
    print('near misses: {:n}'.format(all_near_miss))
    print('rear end crashes: {:n}'.format(all_rear_end))
    print('sideswipe crashes: {:n}'.format(all_sideswipe))
    print('vmt (miles): {:.0f}'.format(all_vmt/1609.34))

    if save_output:
        if len(all_lists) == 1:
            with open(save_name + '.pkl', 'wb') as f:
                pickle.dump([all_lists[-1], laneinds], f)
        else:
            with open(save_name+'.pkl', 'wb') as f:
                pickle.dump([all_lists, laneinds], f)

