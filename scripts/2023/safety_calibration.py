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


def calculate_objective_value(cur_t_ind, cur_n_sims, data_re, data_ss, stats):
    out = [0 for i in range(len(stats))]
    for i in range(cur_t_ind+1):
        vmt = stats[i][2]/1609.34/cur_n_sims[i]
        out_data_re = havsim.helper.crash_confidence(data_re[i], 2600, vmt)
        out_data_ss = havsim.helper.crash_confidence(data_ss[i], 2600, vmt)
        out_re = havsim.helper.crash_confidence(stats[i][3], cur_n_sims[i], vmt)
        out_ss = havsim.helper.crash_confidence(stats[i][4], cur_n_sims[i], vmt)
        out_nm = havsim.helper.crash_confidence(stats[i][5], cur_n_sims[i], vmt)
        nm_ratio = 1/(1/out_re[0] + 1/out_ss[0])/out_nm[0]

        # main term is to match the empirical crash rate
        out[i] += 100*(abs(out_data_re[0] - out_re[0])/min(out_data_re[0], out_re[0])
                       + abs(out_data_ss[0] - out_ss[0])/min(out_data_ss[0], out_ss[0]))
        # regularizer for near misses
        if 10 < nm_ratio < 20:
            pass
        else:
            out[i] += 10 + abs(nm_ratio-15)
        # regularizer for vehicles per crash
        if stats[i][3] > 0:
            out[i] += 100 * abs(stats[i][6] / stats[i][3] - 2.205)
        if stats[i][4] > 0:
            out[i] += 100 * abs(stats[i][7]/stats[i][4] - 2.035)
        # regularizer for confidence width
        if out_re[2] - out_re[1] > out_data_re[2] - out_data_re[1]:
            out[i] += 10*((out_re[2]-out_re[1])/(out_data_re[2] - out_data_re[1]) - .5)**2
        if out_ss[2] - out_ss[1] > out_data_ss[2] - out_data_ss[1]:
            out[i] += 10 * ((out_ss[2] - out_ss[1]) / (out_data_ss[2] - out_data_ss[1]) - .5) ** 2

    if cur_t_ind < len(stats)-1:  # if not test all intervals, add a penalty based on the worst interval's value
        out[cur_t_ind+1:len(stats)] = [max(out)*3 for i in range(len(stats)-cur_t_ind-1)]
    return -sum(out)/len(stats)


if __name__ == '__main__':
    arg_names = ['save_name', 'n_workers', 'use_times', 'gamma_bounds', 'xi_bounds', 'min_simulations', 'n_simulations',
                 'prev_opt_name', 'n_iter', 'init_points', 'init_guesses']
    default_args = ['e94_calibration_1', round(.38*multiprocessing.cpu_count()), [[17, 18], [14, 15]],
                    [(-1, .2), (.2, .75), (0, 2), (0, 2), (1, 2.5)], [(.2, 2), (2, 5.5)],
                    2*round(.38*multiprocessing.cpu_count()), 6*round(.38*multiprocessing.cpu_count()),
                    None, 100, 0,
                    [[-.13, .3, .2, .6, 1.5, .8, 3.75], [-.11, .3, .25, .65, 1.5, .8, 3.25],
                     [-.13, .35, .1, .5, 2., .5, 3.75]]]
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


    def evaluate_crash_rate(**my_args):
        gamma_parameters = [my_args[str(i)] for i in range(5)]
        xi_parameters = [my_args[str(5 + i)] for i in range(2)]

        data_re, data_ss = [], []
        for start, end in use_times:
            data_re.append(sum(e94_rear_ends[int(start):int(end)]) - e94_rear_ends[int(start)] * (start - int(start))
                           + e94_rear_ends[int(end)] * (end - int(end)))
            data_ss.append(sum(e94_sideswipes[int(start):int(end)]) - e94_sideswipes[int(start)] * (start - int(start))
                           + e94_sideswipes[int(end)] * (end - int(end)))

        # repeatedly do pools of simulations, until can evaluate the loss
        pbar = tqdm.tqdm(total=n_sims*len(use_times), position=1, leave=False, desc='Current test')
        stats, cur_n_sims = [(0,)*9 for i in range(len(use_times))], [0 for i in range(len(use_times))]
        cur_t_ind = 0
        while cur_t_ind < len(use_times):
            pool = multiprocessing.Pool(n_workers)
            inner_pbar = [2 if k % (2*n_workers) == 0 else False for k in range(min_sims)]
            args = [(inner_pbar[k], use_times[cur_t_ind], gamma_parameters, xi_parameters) for k in range(min_sims)]

            for count, out in enumerate(pool.imap_unordered(do_simulation, args)):
                stats[cur_t_ind] = tuple((k + out[count2] for count2, k in enumerate(stats[cur_t_ind])))

                # reporting
                pbar.update()
                crash_stats = (stats[cur_t_ind][2] / 1609.34 / max(k, .69) for k in stats[cur_t_ind][3:6])
                update_stats = stats[cur_t_ind][1] / stats[cur_t_ind][0] * n_workers
                pbar.set_postfix_str('Times: '+str(use_times[cur_t_ind])+'. Events: {:n}/{:n}/{:n}. '.format(
                    *stats[cur_t_ind][3:6]) + 'Miles/Events: {:.1e}/{:.1e}/{:.1e}.'.format(*crash_stats) +
                    '  Updates/Sec: {:.1e}.'.format(update_stats))
            pool.close()
            pool.join()
            cur_n_sims[cur_t_ind] += min_sims

            # check status of the time interval cur_t_ind
            vmt = stats[cur_t_ind][2]/1609.34/cur_n_sims[cur_t_ind]
            out_data_re = havsim.helper.crash_confidence(data_re[cur_t_ind], 2600, vmt)
            out_data_ss = havsim.helper.crash_confidence(data_ss[cur_t_ind], 2600, vmt)
            out_re = havsim.helper.crash_confidence(stats[cur_t_ind][3], cur_n_sims[cur_t_ind], vmt)
            out_ss = havsim.helper.crash_confidence(stats[cur_t_ind][4], cur_n_sims[cur_t_ind], vmt)
            if out_re[2] < out_data_re[1] or out_re[1] > out_data_re[2]:
                break  # if completely outside confidence, stop immediately
            if out_ss[2] < out_data_ss[1] or out_ss[1] > out_data_ss[2]:
                break
            if cur_n_sims[cur_t_ind] < n_sims:  # if less than requested, keep doing
                pass
            elif cur_n_sims[cur_t_ind] > 3*n_sims:  # too many simulations, need to move to next one
                cur_t_ind += 1
            elif (out_re[2] - out_re[1]) < 1.5*(out_data_re[2]-out_data_re[1]) and \
                    (out_ss[2] - out_ss[1]) < 1.5*(out_data_ss[2] - out_data_ss[1]):
                cur_t_ind += 1  # confidence width met, move to next one
            elif out_re[0] < out_data_re[1] or out_re[0] > out_data_re[2]:
                break  # outside confidence and done more than n_sims, so stop immediately
            elif out_ss[0] < out_data_ss[1] or out_ss[0] > out_data_ss[2]:
                break
            else:  # need more simulations, update pbar
                cur_total = sum(cur_n_sims[0:cur_t_ind]) + cur_n_sims[cur_t_ind] + (len(cur_n_sims)-cur_t_ind-1)*n_sims
                pbar.total = cur_total + min_sims

        pbar.disable = True
        return calculate_objective_value(cur_t_ind, cur_n_sims, data_re, data_ss, stats)

    now = datetime.now()
    print('\nStarting job \'' + save_name + '\' at ' + now.strftime("%H:%M:%S"))
    print('use_times: '+str(use_times)+'. n_workers: '+str(n_workers)+'. min_simulations: '+str(min_sims))
    print('parameter bounds: '+str(gamma_bounds)+'\n')
    pickle_path = os.path.join(os.path.dirname(__file__), 'pickle files')
    if not os.path.exists(pickle_path):
        print('Warning: the directory ' + pickle_path + ' does not exist.')
        os.makedirs(pickle_path)

    prev_opt_name = None if prev_opt_name is None else os.path.join(pickle_path, prev_opt_name)
    havsim.bayes_opt(evaluate_crash_rate, gamma_bounds, init_points=init_points, init_guesses=init_guesses,
                     n_iter=n_iter, save_name=os.path.join(pickle_path, save_name), prev_opt_name=prev_opt_name)
