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
from safety_calibration import crash_confidence


# -------  SETTINGS  ------- #
save_name = 'e94_14_15'
n_simulations = 600
n_workers = 50
batch_size = 150
save_crashes_only = False if n_simulations == 1 else True

sim_name = 'e94'
use_times = [14, 15]
gamma_parameters = [-.08, .28, .2, 2.0, 1.8]
xi_parameters = [.5, 3.5]
# -------------------------- #


def do_simulation(show_pbar):
    # make + run simulation
    simulation, my_lanes = e94(use_times, gamma_parameters, xi_parameters)
    all_vehicles, my_stats = simulation.simulate(pbar=show_pbar, verbose=False, return_stats=True)
    if show_pbar:
        print('Simulation took {:.0f} seconds. Simulated {:.1e} miles and {:n} vehicles. '.format(
            my_stats[0], my_stats[2]/1609.34, len(all_vehicles)) + ' Simulation speed (updates/sec): {:.1e}\n'.format(
            my_stats[1] / my_stats[0]))

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

    return my_stats, my_vehs, my_lanes


if __name__ == '__main__':
    now = datetime.now()
    print('\nStarting job '+save_name+' at '+now.strftime("%H:%M:%S") + ', simulating times ' + str(use_times) +
          ' for {:n} replications ({:n} workers)'.format(n_simulations, n_workers))
    print('gamma parameters: ' + str(gamma_parameters) + '. xi parameters: ' + str(xi_parameters) + '.')

    all_rear_end, all_sideswipe, all_near_miss, all_vmt, all_re_veh, all_ss_veh, all_nm_veh = 0, 0, 0, 0, 0, 0, 0
    initial_update_rate, cur_update_rate, cur_time_used, cur_updates, time_used = 0, 0, 0, 0, 0
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
            stats, vehs, lanes = out
            time_used, n_updates, vmt, rear_end, sideswipe, near_miss, re_veh, ss_veh, nm_veh = stats
            all_veh_lists.append(vehs)
            all_rear_end, all_sideswipe, all_near_miss, all_vmt = \
                all_rear_end + rear_end, all_sideswipe + sideswipe, all_near_miss + near_miss, all_vmt + vmt
            all_re_veh, all_ss_veh, all_nm_veh = all_re_veh + re_veh, all_ss_veh + ss_veh, all_nm_veh + nm_veh
            cur_time_used, cur_updates = cur_time_used+time_used, cur_updates+n_updates

            # reporting
            cur_iter += 1
            vmt_miles = all_vmt/1609.34
            crash_stats = (vmt_miles/max(all_rear_end, .69), vmt_miles/max(all_sideswipe, .69),
                           vmt_miles/max(all_near_miss, .69))
            event_stats = (all_rear_end, all_sideswipe, all_near_miss)
            event_stats2 = (all_re_veh/max(all_rear_end, 1), all_ss_veh/max(all_sideswipe, 1),
                            all_nm_veh/max(all_near_miss, 1))
            sim_stats = (vmt_miles, cur_updates/cur_time_used*min(n_workers, cur_sims))
            pbar.update()
            pbar.set_description('Simulations finished')
            pbar.set_postfix_str('Miles/Event (Rear end/Sideswipe/Near miss): {:.1e}/{:.1e}/{:.1e}'.format(
                *crash_stats) + ',  Events: {:.0f}/{:.0f}/{:.0f}'.format(*event_stats) +
                ',  Vehicles/Event: {:.1f}/{:.1f}/{:.2f}'.format(*event_stats2) +
                ',  Miles: {:.2e},  Updates/Sec: {:.1e}'.format(*sim_stats))

        pool.close()
        pool.join()

        # save in chunks
        if i < batch_iters-1:
            with open('pickle files/' + save_name + '-part-'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(all_veh_lists, f)
        # include pauses if needed
        cur_update_rate, cur_time_used, cur_updates = cur_updates/cur_time_used, 0, 0
        if i == 0:
            initial_update_rate = cur_update_rate
        if 1.1*cur_update_rate < initial_update_rate and i < batch_iters-1:
            sleep(time_used*.25)
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
        'near miss vehicles': all_nm_veh,
        'sideswipe vehicles': all_ss_veh,
        'rear end vehicles': all_re_veh,
        'vmt': all_vmt/1609.34,
        'timesteps_before': 100,
        'timesteps_after': 5,
        'dt': .2
    }
    with open('pickle files/' + save_name + '_config'+'.config', 'wb') as f:
        pickle.dump(config, f)

    vmt_miles = all_vmt/1609.34
    out_rear_ends = crash_confidence(all_rear_end, n_simulations, vmt_miles / n_simulations)
    out_sideswipe = crash_confidence(all_sideswipe, n_simulations, vmt_miles / n_simulations)
    out_near_miss = crash_confidence(all_near_miss, n_simulations, vmt_miles / n_simulations)
    print('\n-----------SUMMARY-----------')
    print('Simulated {:.3} miles. Events: {:.0f} rear ends ({:.0f} vehicles)'.format(
          vmt_miles, all_rear_end, all_re_veh)+',  {:.0f} sideswipes ({:.0f} vehicles)'.format(
          all_sideswipe, all_ss_veh) + ',  {:.0f} near misses ({:.0f} vehicles).'.format(all_near_miss, all_nm_veh))
    print('rear end inverse crash rate: {:.3}. 95% confidence: [{:.3}, {:.3}].'.format(*out_rear_ends))
    print('sideswipe inverse crash rate: {:.3}. 95% confidence: [{:.3}, {:.3}]'.format(*out_sideswipe))
    print('near miss inverse crash rate: {:.3}. 95% confidence: [{:.3}, {:.3}]'.format(*out_near_miss))
    print('Finished  '+save_name+' at ' + after.strftime("%H:%M:%S"))
