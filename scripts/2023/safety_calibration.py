"""Script to calibrate the stochastic crash parameters."""
import havsim
from make_simulation import e94
import multiprocessing
import tqdm
import os
from datetime import datetime
from time import sleep


def do_simulation(my_args):
    use_pbar, my_use_times, my_gamma_parameters, my_xi_parameters = my_args
    simulation, my_lanes = e94(my_use_times, my_gamma_parameters, my_xi_parameters)
    all_vehicles, my_stats = simulation.simulate(pbar=use_pbar, return_stats=True)
    return *my_stats, len(all_vehicles)


def calculate_objective_value(n_rear_end, n_sideswipe, n_near_miss, vmt, re_veh, ss_veh, nm_veh, n_sims,
                              n_rear_end_data, n_sideswipe_data, n_days_data):

    pass  # l2 crash rate target, regularizer for near miss + nveh/crash + confidence width


if __name__ == '__main__':
    arg_names = ['save_name', 'n_workers', 'use_times', 'gamma_bounds', 'xi_bounds', 'min_simulations', 'n_simulations',
                 'prev_opt_name', 'n_iter', 'init_points', 'init_guesses']
    default_args = ['e94_calibration_1', round(.4*multiprocessing.cpu_count()), [[11, 12], [16, 17]],
                    [(-1, .2), (.2, .75), (0, 2), (0, 2), (1, 2.5)], [(.2, 2), (2, 5.5)], 100, None, 100, 0,
                    [[-.13, .3, .2, .6, 1.5, .8, 3.75], [-.5, .5, .3, .5, 1.5, .8, 3.75]]]
    desc_str = 'Calibrate gamma/xi parameters by simulating the crash rate under realistic conditions, '\
        'and compare against crashes data. This is an intensive procedure which requires running many simulations.'
    arg_de = \
        ['str, name of file for saving optimizer result (not including extension). Inside of the ./pickle files '
         'folder, the optimizer state is saved in save_name.json. The save_name_res.pkl is a list of each '
         'step done by the optimizer (giving the parameters and objective function value for that step).',
         'int, number of simulations to run in parallel',
         'list of lists, each inner list has 2 floats, giving the starting/ending times (0-24) to calibrate. The'
         'objective function is summed over each given time interval.',
         'list of tuples, where each tuple is the upper/lower bounds for the gamma parameter with that index',
         'list of tuples, where each tuple is the upper/lower bounds for the xi parameter with that index',
         'int, minimum number of simulations to run for each multiprocessing.pool. The total number of simulations'
         ' will always be a multiple of this number. If set too small, will make the pool inefficient. The number'
         ' of simulations that is run per objective function evaluation is determined dynamically, based on'
         ' the confidence interval of the simulated crashes, and the (un)certainty in the crashes data.',
         'int, "typical" number of simulations to run for each time interval.',
         'if not None, the save_name from a previous run, and the .json log will be loaded, which will restart'
         ' the opimizer.',
         'int, number of optimization steps to perform (number of objective function evaluations)',
         'int, number of randomly sampled initial points for the optimizer',
         'if not None, list of lists, where each list is a set of parameters (7 floats) for the optimizer'
         'to evaluate (gamma parameters concatenated with xi parameters).']
    e94_rear_ends = [0, 1, 1, 1, 0, 1, 3, 12, 23, 9, 5, 7, 5, 3, 11, 53, 92, 105, 81, 21, 4, 6, 2, 3]
    e94_sideswipes = [3, 0, 0, 1, 2, 3, 4, 5, 11, 10, 6, 7, 3, 9, 5, 15, 13, 21, 18, 10, 8, 4, 4, 8]
    save_name, n_workers, use_times, gamma_bounds, xi_bounds, min_sims, n_sims, prev_opt_name, n_iter, \
        init_points, init_guesses = havsim.parse_args(arg_names, default_args, desc_str, arg_de, 0)
    gamma_bounds.extend(xi_bounds)


    def evaluate_crash_rate(my_args):
        gamma_parameters = [my_args[str(i)] for i in range(5)]
        xi_parameters = [my_args[str(5 + i)] for i in range(2)]

        pbar = tqdm.tqdm(total=min_sims*len(use_times), position=1, leave=False)
        stats = [(0,)*9 for i in range(len(use_times))]
        cur_t_ind, cur_n_sims = 0, 0
        can_evaluate = False
        while not can_evaluate:
            pool = multiprocessing.Pool(n_workers)
            inner_pbar = [2 if k % (2*n_workers) == 0 else False for k in range(min_sims)]
            args = [(inner_pbar[k], use_times[cur_t_ind], gamma_parameters, xi_parameters) for k in range(min_sims)]

            for count, out in enumerate(pool.imap_unordered(do_simulation, args)):
                stats[cur_t_ind] = tuple((out[count2] + k for count2, k in enumerate(stats[cur_t_ind])))

                # reporting
                pbar.update()
                crash_stats = (stats[cur_t_ind][2] / 1609.34 / max(k, .69) for k in stats[cur_t_ind][3:6])
                update_stats = stats[cur_t_ind][1] / stats[cur_t_ind][0] * n_workers
                pbar.set_postfix_str('Events: {:n}/{:n}/{:n}. Miles/Events: {:.1e}/{:.1e}/{:.1e}.'.format(
                    *stats[cur_t_ind][3:6], *crash_stats) + '  Updates/Sec: {:.1e}.'.format(update_stats))
                if inner_pbar[count]:
                    sleep(0.01)
                    total = int(18000 * (use_times[cur_t_ind][1] - use_times[cur_t_ind][0]))
                    postfix = ' [Simulated {:.1e} miles and {:n} vehicles. Updates/sec: {:.1e}. '.format(
                        out[2] / 1609.34, out[-1], out[1] / out[0]) + 'Time used: {:.1f}.]'.format(out[0])
                    pbar_inner = tqdm.tqdm(total=total, position=1, leave=False, desc='Current simulation timesteps',
                                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}' + postfix)
                    pbar_inner.update(total)
                    pbar_inner.set_postfix_str('')
            pool.close()
            pool.join()

            # check status of the time interval cur_t_ind
            cur_n_sims += min_sims
            start, end = use_times[cur_t_ind]
            vmt = stats[cur_t_ind][2]/1609.34/cur_n_sims
            data_re = sum(e94_rear_ends[int(start):int(end)]) - e94_rear_ends[int(start)] * (start - int(start)) \
                + e94_rear_ends[int(end)] * (end - int(end))
            data_ss = sum(e94_sideswipes[int(start):int(end)]) - e94_sideswipes[int(start)] * (start - int(start)) \
                + e94_sideswipes[int(end)] * (end - int(end))
            out_data_re = havsim.helper.crash_confidence(data_re, 2600, vmt)
            out_data_ss = havsim.helper.crash_confidence(data_ss, 2600, vmt)
            # calculate the data confidence, calculate simulated confidence, compare, maybe calculate loss.
            havsim.helper.crash_confidence(stats[cur_t_ind][3], cur_n_sims, stats[cur_t_ind][2]/1609.34/cur_n_sims)



    now = datetime.now()
    print('\nStarting job \'' + save_name + '\' at ' + now.strftime("%H:%M:%S"))
    pickle_path = os.path.join(os.path.dirname(__file__), 'pickle files')
    if not os.path.exists(pickle_path):
        print('Warning: the directory ' + pickle_path + ' does not exist.')
        os.makedirs(pickle_path)

    havsim.bayes_opt(evaluate_crash_rate, gamma_bounds, n_iter=n_iter, init_points=init_points,
                     init_guesses=init_guesses, save_name=save_name, prev_opt_name=prev_opt_name)
