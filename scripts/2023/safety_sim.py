"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim
import pickle
import multiprocessing
import tqdm
import os
from datetime import datetime
from time import sleep


def do_simulation(my_args):
    use_pbar, my_use_times, my_gamma_parameters, my_xi_parameters, crashes_only = my_args
    simulation, my_lanes = e94(my_use_times, my_gamma_parameters, my_xi_parameters)
    all_vehicles, my_stats = simulation.simulate(pbar=use_pbar, return_stats=True)

    # save vehicles
    if crashes_only:
        my_vehs = []
        for crash in simulation.crashes:
            crash_times = [veh.crash_time for veh in crash]
            t_start, t_end = min(crash_times) - 100, max(crash_times) + 5
            my_vehs.extend(havsim.helper.add_leaders(crash, t_start, t_end))
        my_vehs = list(set(my_vehs))
        for veh in all_vehicles:
            if len(veh.near_misses) > 0:
                for times in veh.near_misses:
                    t_start, t_end = times[0] - 100, times[1] + 25
                    my_vehs.extend(havsim.helper.add_leaders([veh], t_start, t_end))
        my_vehs = list(set(my_vehs))
        [veh._remove_veh_references() for veh in my_vehs]  # makes pickling efficient
    else:
        [veh._remove_veh_references() for veh in all_vehicles]
        my_vehs = all_vehicles

    return (*my_stats, len(all_vehicles)), my_vehs, my_lanes


if __name__ == '__main__':
    arg_names = ['save_name', 'n_simulations', 'sim_name', 'use_times', 'gamma_parameters', 'xi_parameters',
                 'n_workers', 'batch_size', 'save_crashes_only']
    default_args = ['e94_16_17_test', 40, 'e94', [16, 17], [-.13, .3, .2, .6, 1.5], [.8, 3.75],
                    round(.4*multiprocessing.cpu_count()), 300, True]
    d_str = 'Run multiple simulations in parallel using multiprocessing, and save the result.'
    arg_d = ['str, giving the filename (not including extension) to save to inside \'pickle files\' folder',
             'int, number of simulations to run.',
             'str, name of area to simulate, one of {\'e94\', \'w94\'}',
             'list of 2 floats, giving the starting and ending times (0-24) to simulate. The end time must '
             'be larger than the starting time.',
             'list of 5 floats, giving the parameters for gamma (see havsim.vehicles.StochasticVehicle)',
             'list of 2 floats, giving the parameters for xi (see havsim.vehicles.StochasticVehicle)',
             'int, number of simulations to run in parallel',
             'int, maximum number of simulations to run before partially writing to disk and making a new '
             'multiprocessing.Pool. If set too low, may run out of memory.',
             'bool, if True then save all vehicles. This will cause a large filesize per simulation. '
             ' If False, only vehicles in crashes are saved.']
    save_name, n_simulations, sim_name, use_times, gamma_parameters, xi_parameters, n_workers, batch_size, \
        save_crashes_only = havsim.parse_args(arg_names, default_args, d_str, arg_d, 0)

    pickle_path = os.path.join(os.path.dirname(__file__), 'pickle files')
    if not os.path.exists(pickle_path):
        print('Warning: the directory ' + pickle_path + ' does not exist.')
        os.makedirs(pickle_path)

    now = datetime.now()
    print('\nStarting job \'' + save_name + '\' at ' + now.strftime("%H:%M:%S"))
    print('Requested simulations: {:n}. Workers: {:n}. Simulation area: \''.format(n_simulations, n_workers)
          + sim_name + '\'. Simulation times: ' + str(use_times))
    print('gamma parameters: ' + str(gamma_parameters) + '. xi parameters: ' + str(xi_parameters) + '.\n')
    pbar = tqdm.tqdm(total=n_simulations, position=0, leave=True)
    pbar.set_description('Simulations')

    all_stats = (0,)*10
    batch_iters = int(n_simulations // batch_size)
    leftover = n_simulations - batch_iters * batch_size
    batch_iters = batch_iters + 1 if leftover > 0 else batch_iters

    # do parallel simulations in batches
    for i in range(batch_iters):
        all_veh_lists = []
        cur_sims = leftover if i == batch_iters - 1 and leftover > 0 else batch_size
        pool = multiprocessing.Pool(min(n_workers, cur_sims))
        inner_pbar = [1 if k % (2*n_workers) == 0 else False for k in range(cur_sims)]
        args = [(inner_pbar[k], use_times, gamma_parameters, xi_parameters, save_crashes_only) for k in range(cur_sims)]

        for count, out in enumerate(pool.imap_unordered(do_simulation, args)):
            stats, vehs, lanes = out
            all_stats = (*(all_stats[count] + k for count, k in enumerate(stats[:-1])), all_stats[-1])
            all_veh_lists.append(vehs)

            # reporting
            crash_stats = (all_stats[2] / 1609.34 / max(k, .69) for k in all_stats[3:6])
            update_stats = all_stats[1] / all_stats[0] * min(n_workers, cur_sims)
            pbar.update()
            pbar.set_postfix_str('Events: {:n}/{:n}/{:n}. Miles/Events: {:.1e}/{:.1e}/{:.1e}.'.format(
                *all_stats[3:6], *crash_stats) + '  Updates/Sec: {:.1e}.'.format(update_stats))
            if inner_pbar[count]:
                sleep(.05)
                total = int(18000*(use_times[1]-use_times[0]))
                postfix = ' [Simulated {:.1e} miles and {:n} vehicles. Updates/sec: {:.1e}. '.format(
                    stats[2]/1609.34, stats[-1], stats[1]/stats[0]) + 'Time used: {:.1f}.]'.format(stats[0])
                pbar_inner = tqdm.tqdm(total=total, position=1, leave=False, desc='Current simulation timesteps',
                                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'+postfix)
                pbar_inner.update(total)
                pbar_inner.set_postfix_str('')
        pool.close()
        pool.join()

        # save in chunks
        if i < batch_iters - 1:
            with open(os.path.join(pickle_path, save_name + '-part-' + str(i) + '.pkl'), 'wb') as f:
                pickle.dump(all_veh_lists, f)
        # include pauses if needed
        if i == 0:
            all_stats, cur_rate = (0, 0, *all_stats[2:-1], all_stats[1]/all_stats[0]), all_stats[1]/all_stats[0]
        else:
            all_stats, cur_rate = (0, 0, *all_stats[2:]), all_stats[1]/all_stats[0]
        if 1.1 * cur_rate < all_stats[-1] and i < batch_iters - 1:
            sleep(stats[0] * .5)
    pbar.close()
    pbar_inner.close()

    # save result + config
    if batch_iters > 1:
        for i in range(batch_iters - 1):
            with open(os.path.join(pickle_path, save_name + '-part-' + str(i) + '.pkl'), 'rb') as f:
                extra_lists = pickle.load(f)
            all_veh_lists.extend(extra_lists)
            os.unlink(os.path.join(pickle_path, save_name + '-part-' + str(i) + '.pkl'))
    with open(os.path.join(pickle_path, save_name + '.pkl'), 'wb') as f:
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
        'time elapsed': (after - now).total_seconds(),
        'initial update rate': all_stats[-1],
        'update rate': cur_rate*n_workers,
        'near misses': all_stats[5],
        'rear ends': all_stats[3],
        'sideswipes': all_stats[4],
        'near miss vehicles': all_stats[8],
        'sideswipe vehicles': all_stats[7],
        'rear end vehicles': all_stats[6],
        'vmt': all_stats[2] / 1609.34,
        'timesteps_before': 100,
        'timesteps_after': 25,
        'dt': .2
    }
    with open(os.path.join(pickle_path, save_name + '_config.config'), 'wb') as f:
        pickle.dump(config, f)

    out_rear_ends = havsim.helper.crash_confidence(all_stats[3], n_simulations, all_stats[2] / 1609.34 / n_simulations)
    out_sideswipe = havsim.helper.crash_confidence(all_stats[4], n_simulations, all_stats[2] / 1609.34 / n_simulations)
    out_near_miss = havsim.helper.crash_confidence(all_stats[5], n_simulations, all_stats[2] / 1609.34 / n_simulations)
    print('\n\n-----------SUMMARY-----------')
    print('Simulated {:.3} miles. Events: {:.0f} rear ends ({:.0f} vehicles)'.format(
        all_stats[2]/1609.34, all_stats[3], all_stats[6]) + ',  {:.0f} sideswipes ({:.0f} vehicles)'.format(
        all_stats[4], all_stats[7]) + ',  {:.0f} near misses ({:.0f} vehicles).'.format(all_stats[5], all_stats[8]))
    print('Rear end inverse crash rate, [95% confidence interval]: {:.3}, [{:.3}, {:.3}].'.format(*out_rear_ends))
    print('Sideswipe inverse crash rate, [95% confidence interval]: {:.3}, [{:.3}, {:.3}]'.format(*out_sideswipe))
    print('Near miss inverse rate, [95% confidence interval]: {:.3}, [{:.3}, {:.3}]'.format(*out_near_miss))
    print('Finished  \'' + save_name + '\' at ' + after.strftime("%H:%M:%S"))
