"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim
import pickle
import multiprocessing
import tqdm
import os
from datetime import datetime
from time import sleep

n_processes = 35
replications = 4
replications_batch_size = 2
save_crashes_only = True
save_name = 'e94_test_crash_1'

use_times = [16, 17]
gamma_parameters = [-.1, .28, .5, 2., 2.]
xi_parameters = [.2, 4]


def do_simulation(my_args):
    my_pbar, iters, printout = my_args
    simulation, my_lanes = e94(use_times, gamma_parameters, xi_parameters)
    my_rear_end, my_sideswipe, my_near_miss, my_vmt, my_time, my_timesteps = 0, 0, 0, 0, 0, 0
    my_out_lists = []
    for i in range(replications_batch_size):
        all_vehicles, elapsed_time, total_timesteps = simulation.simulate(return_times=True)

        # count statistics + do printout
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
        if my_pbar:
            my_pbar.set_description('Simulation {:n}'.format(iters+n_processes*i))
            my_pbar.set_postfix_str('Rear end/Sideswipe/Near miss: {:n}/{:n}/{:n}, VMT: {:n}, Steps/Sec: {:n}'.format(
                *printout[:4])+'Secs/Sim:{:n}'.format(printout[-1]))

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

        # reset simulation, del to reduce memory usage
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

    all_rear_end, all_sideswipe, all_near_miss, all_vmt = 0, 0, 0, 0
    initial_update_rate, cur_update_rate, cur_time_used, cur_timesteps = 0, 0, 0, 0
    args = [(None, None, None) for i in range(n_processes)]
    batch_iters = max(int(replications // replications_batch_size), 1)
    pbar, cur_iters = tqdm.tqdm(range(replications*n_processes)), 0
    args[0] = (pbar, cur_iters, (all_rear_end, all_sideswipe, all_near_miss, all_vmt, cur_update_rate))

    # do parallel simulations in batches
    for i in range(batch_iters):
        pool = multiprocessing.Pool(n_processes)
        out = pool.map(do_simulation, args)
        all_lists = []
        for output in out:
            rear_end, sideswipe, near_miss, vmt, time_used, timesteps, all_out_lists, lanes = output
            all_rear_end, all_sideswipe, all_near_miss, all_vmt = \
                all_rear_end+rear_end, all_sideswipe+sideswipe, all_near_miss+near_miss, all_vmt+vmt
            cur_time_used, cur_timesteps = cur_time_used+time_used, cur_timesteps+timesteps
            all_lists.extend(all_out_lists)
        pool.close()

        cur_update_rate, cur_time_used, cur_timesteps = cur_timesteps/cur_time_used, 0, 0
        if i == 0:
            initial_update_rate = cur_update_rate
        if 1.25*cur_update_rate < initial_update_rate:
            sleep(150)
        if i < batch_iters-1:
            with open('pickle files/' + save_name + '-part-'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(all_lists, f)
        cur_iters += replications_batch_size*n_processes
        args[0] = (pbar, cur_iters, (all_rear_end, all_sideswipe, all_near_miss, all_vmt, cur_update_rate))

    # save result
    if batch_iters > 1:
        for i in range(batch_iters-1):
            with open('pickle files/' + save_name + '-part-'+str(i)+'.pkl', 'rb') as f:
                extra_lists = pickle.load(f)
            all_lists.extend(extra_lists)
            os.unlink('pickle files/' + save_name + '-part-'+str(i)+'.pkl')
    with open('pickle files/' + save_name + '.pkl', 'wb') as f:
        pickle.dump([all_lists, lanes], f)
    after = datetime.now()
    config = {
        'n_simulations': batch_iters*replications_batch_size,
        'n_processes': n_processes,
        'save_crashes_only': save_crashes_only,
        'gamma_parameters': gamma_parameters,
        'xi_parameters': xi_parameters,
        'time elapsed': (after-now).total_seconds(),
        'near misses': all_near_miss,
        'rear ends': all_rear_end,
        'sideswipes': all_sideswipe,
        'vmt': all_vmt/1609.34
    }
    with open('pickle files/' + save_name + '_config'+'.config', 'wb') as f:
        pickle.dump(config, f)
    print('\n-----------SUMMARY-----------')
    print('near misses: {:n}'.format(all_near_miss))
    print('rear end crashes: {:n}'.format(all_rear_end))
    print('sideswipe crashes: {:n}'.format(all_sideswipe))
    print('vmt (miles): {:.0f}'.format(all_vmt / 1609.34))
    print('\nFinished  '+save_name+' at ' + after.strftime("%H:%M:%S"))
