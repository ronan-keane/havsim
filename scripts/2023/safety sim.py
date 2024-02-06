"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim
import pickle
import multiprocessing
import tqdm
import os
import sys
from datetime import datetime
from time import sleep


# -------  SETTINGS  ------- #
save_name = 'e94_14_15_3'
n_simulations = 1
n_workers = 1
batch_size = 150
save_crashes_only = False if n_simulations == 1 else True

sim_name = 'e94'
use_times = [14, 15]
gamma_parameters = [-.1, .33, .3, 2.0, 1.5]
xi_parameters = [1, 3]
# -------------------------- #


def do_simulation(show_pbar):
    # make + run simulation
    simulation, my_lanes = e94(use_times, gamma_parameters, xi_parameters)
    all_vehicles, elapsed_time, total_timesteps = simulation.simulate(pbar=show_pbar, verbose=False, return_times=True)
    if show_pbar:
        print('simulation time is {:.1f} seconds over {:.2e} timesteps ({:n} vehicles)\n'.format(
            elapsed_time, total_timesteps, len(all_vehicles)))

    # count statistics
    my_rear_end, my_sideswipe, my_near_miss, my_vmt = 0, 0, 0, 0
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
        my_vehs = []
        for crash in simulation.crashes:
            crash_times = [veh.crash_time for veh in crash]
            t_start, t_end = min(crash_times) - 100, max(crash_times) + 5
            my_vehs.extend(havsim.helper.add_leaders(crash, t_start, t_end))
        my_vehs = list(set(my_vehs))
        for veh in all_vehicles:
            if len(veh.near_misses) > 0:
                for times in veh.near_misses:
                    t_start, t_end = times[0] - 100, times[1] + 5
                    my_vehs.extend(havsim.helper.add_leaders([veh], t_start, t_end))
        my_vehs = list(set(my_vehs))
        [veh._remove_veh_references() for veh in my_vehs]  # makes pickling efficient
    else:
        [veh._remove_veh_references() for veh in all_vehicles]
        my_vehs = all_vehicles

    return my_rear_end, my_sideswipe, my_near_miss, my_vmt, elapsed_time, total_timesteps, my_vehs, my_lanes


if __name__ == '__main__':
    now = datetime.now()
    print('Starting job '+save_name+' at '+now.strftime("%H:%M:%S") + ', simulating times ' + str(use_times) +
          ' for {:n} replications ({:n} workers)'.format(n_simulations, n_workers))
    print('gamma parameters: ' + str(gamma_parameters) + '. xi parameters: ' + str(xi_parameters) + '.')

    all_rear_end, all_sideswipe, all_near_miss, all_vmt = 0, 0, 0, 0
    initial_update_rate, cur_update_rate, cur_time_used, cur_timesteps = 0, 0, 0, 0
    all_veh_lists, lanes = None, None
    batch_iters = int(n_simulations // batch_size)
    leftover = n_simulations - batch_iters * batch_size
    batch_iters = batch_iters + 1 if leftover > 0 else batch_iters
    print('\nWorking on first simulation...')
    pbar = tqdm.tqdm(range(n_simulations), total=n_simulations, file=sys.stdout)
    cur_iter = 0

    # do parallel simulations in batches
    for i in range(batch_iters):
        all_veh_lists = []
        cur_sims = leftover if i == batch_iters-1 and leftover > 0 else batch_size
        pool = multiprocessing.Pool(min(n_workers, cur_sims))
        args = [False for k in range(cur_sims)]
        args[0] = True if i == 0 else False

        for count, out in enumerate(pool.imap_unordered(do_simulation, args)):
            rear_end, sideswipe, near_miss, vmt, time_used, timesteps, vehs, lanes = out
            all_veh_lists.append(vehs)
            all_rear_end, all_sideswipe, all_near_miss, all_vmt = \
                all_rear_end + rear_end, all_sideswipe + sideswipe, all_near_miss + near_miss, all_vmt + vmt
            cur_time_used, cur_timesteps = cur_time_used+time_used, cur_timesteps+timesteps

            # reporting
            cur_iter += 1
            vmt_miles = all_vmt/1609.34
            crash_stats = (vmt_miles/max(all_rear_end, .69), vmt_miles/max(all_sideswipe, .69),
                           vmt_miles/max(all_near_miss, .69))
            sim_stats = (vmt_miles, cur_timesteps/cur_time_used, cur_time_used/(count+1))
            pbar.update()
            pbar.set_description('Simulations finished')
            pbar.set_postfix_str('Miles/Event (Rear end/Sideswipe/Near miss): {:.1e}/{:.1e}/{:.1e}'.format(
                *crash_stats) + ', Total Miles: {:.2e}, Steps/Sec: {:.1e}, Secs/Sim:{:.0f}'.format(*sim_stats))

        pool.close()
        pool.join()

        # save in chunks
        if i < batch_iters-1:
            with open('pickle files/' + save_name + '-part-'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(all_veh_lists, f)
        # automatically pause simulations if needed
        cur_update_rate, cur_time_used, cur_timesteps = cur_timesteps/cur_time_used, 0, 0
        if i == 0:
            initial_update_rate = cur_update_rate
        if 1.1*cur_update_rate < initial_update_rate:
            sleep(120)
    pbar.close()

    # save result + config
    if batch_iters > 1:
        for i in range(batch_iters-1):
            with open('pickle files/' + save_name + '-part-'+str(i)+'.pkl', 'rb') as f:
                extra_lists = pickle.load(f)
            all_veh_lists.extend(extra_lists)
            os.unlink('pickle files/' + save_name + '-part-'+str(i)+'.pkl')
    with open('pickle files/' + save_name + '.pkl', 'wb') as f:
        pickle.dump([all_veh_lists, lanes], f)
    after = datetime.now()
    config = {
        'n_simulations': n_simulations,
        'batch_size': batch_size,
        'n_workers': n_workers,
        'save_crashes_only': save_crashes_only,
        'sim_name': sim_name,
        'use_times': use_times,
        'gamma_parameters': gamma_parameters,
        'xi_parameters': xi_parameters,
        'time elapsed': (after-now).total_seconds(),
        'initial update rate': initial_update_rate,
        'update rate': cur_update_rate,
        'near misses': all_near_miss,
        'rear ends': all_rear_end,
        'sideswipes': all_sideswipe,
        'vmt': all_vmt/1609.34,
        'timesteps_before': 100,
        'timesteps_after': 5,
        'dt': .2
    }
    with open('pickle files/' + save_name + '_config'+'.config', 'wb') as f:
        pickle.dump(config, f)
    print('\n-----------SUMMARY-----------')
    print('near misses: {:n}'.format(all_near_miss))
    print('rear end crashes: {:n}'.format(all_rear_end))
    print('sideswipe crashes: {:n}'.format(all_sideswipe))
    print('vmt (miles): {:.0f}'.format(all_vmt / 1609.34))
    print('\nFinished  '+save_name+' at ' + after.strftime("%H:%M:%S"))
