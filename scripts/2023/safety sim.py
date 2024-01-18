"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim
import pickle
import multiprocessing
import tqdm
from datetime import datetime
from time import sleep

n_processes = 35
replications = 4
replications_batch_size = 2
save_crashes_only = True
save_name = 'pickle files/e94_test_crash_7'

use_times = [16, 17]
gamma_parameters = [-.1, .28, .5, 2., 2.]
xi_parameters = [.2, 4]


def do_simulation(my_args):
    my_pbar, iter = my_args
    simulation, my_lanes = e94(use_times, gamma_parameters, xi_parameters)
    my_rear_end, my_sideswipe, my_near_miss, my_vmt, my_time, my_timesteps = 0, 0, 0, 0, 0, 0
    my_out_lists = []
    for i in range(replications):
        all_vehicles, elapsed_time, total_timesteps = simulation.simulate(return_times=True)

        # count statistics
        for crash in simulation.crashes:
            if crash[0].crashed[0] == 'rear end':
                my_rear_end += 1
            else:
                my_sideswipe += 1
        for veh in all_vehicles:
            my_near_miss += len(veh.near_misses)
            my_vmt += veh.posmem[-1] - veh.posmem[0]
        my_time += elapsed_time
        my_timesteps += total_timesteps

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
            [veh.__remove_veh_references() for veh in cur]
            my_out_lists.append(cur)
        else:
            [veh.__remove_veh_references() for veh in all_vehicles]
            my_out_lists.append(all_vehicles)

        if i < replications - 1:
            if len(simulation.crashes) > 0:
                del crash
            simulation.reset()
            del all_vehicles
            del veh
            if save_crashes_only:
                del cur
    return my_rear_end, my_sideswipe, my_near_miss, my_vmt, my_time, my_timesteps, my_out_lists, my_lanes


if __name__ == '__main__':
    now = datetime.now()
    print('Starting at '+now.strftime("%H:%M:%S")+', simulating times '+str(use_times)+' for '+str(replications) +
          ' replications ('+str(n_processes)+' workers)')

    batch_iters = max(int(replications // replications_batch_size), 1)
    my_pbar, cur_iters = tqdm.tqdm(range(replications)), 0
    all_rear_end, all_sideswipe, all_near_miss, all_vmt = 0, 0, 0, 0
    initial_update_rate, cur_update_rate = 0, 0
    args = [(None, None) for i in range(n_processes)]
    args[0] = (my_pbar, cur_iters)

    for i in range(batch_iters):
        pool = multiprocessing.Pool(n_processes)
        out = pool.map(do_simulation, args)
        all_lists = []
        for output in out:
            rear_end, sideswipe, near_miss, vmt, time_used, timesteps, all_out_lists, lanes = output
            all_rear_end += rear_end
            all_sideswipe += sideswipe
            all_near_miss += near_miss
            all_vmt += vmt
            all_lists.extend(all_out_lists)
        pool.close()
        cur_iters += 1
        args[0] = (my_pbar, cur_iters)

    print('\n-----------SUMMARY-----------')
    print('near misses: {:n}'.format(all_near_miss))
    print('rear end crashes: {:n}'.format(all_rear_end))
    print('sideswipe crashes: {:n}'.format(all_sideswipe))
    print('vmt (miles): {:.0f}'.format(all_vmt/1609.34))

    with open(save_name+'.pkl', 'wb') as f:
        pickle.dump([all_lists, lanes], f)
    now = datetime.now()
    print('\nFinished  at ' + now.strftime("%H:%M:%S"))
