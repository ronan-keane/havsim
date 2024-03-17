"""Script to calibrate the stochastic crash parameters """
import havsim
from make_simulation import e94
import multiprocessing
import tqdm
import os
from datetime import datetime


def do_simulation(my_args):
    use_pbar, my_use_times, my_gamma_parameters, my_xi_parameters = my_args
    simulation, my_lanes = e94(my_use_times, my_gamma_parameters, my_xi_parameters)
    all_vehicles, my_stats = simulation.simulate(pbar=use_pbar, return_stats=True)
    return *my_stats, len(all_vehicles)


def calculate_objective_value(n_rear_end, n_sideswipe, n_near_miss, vmt, re_veh, ss_veh, nm_veh, n_sims,
                              n_rear_end_data, n_sideswipe_data, n_days_data):

    pass  # l2 crash rate target, regularizer for near miss + nveh/crash + confidence width


if __name__ == '__main__':
    arg_names = ['save_name', 'n_workers', 'use_times', 'gamma_bounds', 'xi_bounds', 'n_simulations', 'min_simulations',
                 'prev_opt_name', 'n_iter', 'init_points', 'init_guesses']
    default_args = ['e94_calibration_1', round(.4*multiprocessing.cpu_count()), [[11, 12], [16, 17]],
                    [(-1, .2), (.2, .75), (0, 2), (0, 2), (1, 2.5)], [(.2, 2), (2, 5.5)], 300, 100, None, 100, 0,
                    [[-.13, .3, .2, .6, 1.5, .8, 3.75], [-.5, .5, .3, .5, 1.5, .8, 3.75]]]
    desc_str = 'Calibrate gamma/xi parameters by simulating the crash rate under realistic conditions, '\
        'and compare against crashes data. This is an intensive procedure which requires running many simulations.'
    arg_de = \
        ['str, name of file for saving optimizer result (not including extension). Inside of the ./pickle files '
         'folder, the optimizer state is saved in save_name.json. The save_name_res.pkl is a list of each '
         'objective function evaluation done by the optimizer.',
         'int, number of simulations to run in parallel',
         'list of lists, each inner list has 2 floats, giving the starting/ending times (0-24) to calibrate. The'
         'objective function is summed over each given time interval.',
         'list of tuples, where each tuple is the upper/lower bounds for the gamma parameter with that index',
         'list of tuples, where each tuple is the upper/lower bounds for the xi parameter with that index',
         'int, requested number of simulations to normally run. We may run additional if the confidence interval is '
         'too large, or fewer if the crashes are too unrealistic.',
         'int, the smallest number of simulations to run before evaluating the crash rate.',
         'if not None, the save_name from a previous run, and the .json log will be loaded, which will restart'
         ' the opimizer.',
         'int, number of optimization steps to perform (number of objective function evaluations)',
         'int, number of randomly sampled initial points for the optimizer',
         'if not None, list of lists, where each list is a set of parameters (7 floats) for the optimizer'
         'to evaluate (gamma parameters concatenated with xi parameters).']
    e94_rear_ends = [0, 1, 1, 1, 0, 1, 3, 12, 23, 9, 5, 7, 5, 3, 11, 53, 92, 105, 81, 21, 4, 6, 2, 3]
    e94_sideswipes = [3, 0, 0, 1, 2, 3, 4, 5, 11, 10, 6, 7, 3, 9, 5, 15, 13, 21, 18, 10, 8, 4, 4, 8]
    save_name, n_simulations, n_workers, use_times, gamma_bounds, xi_bounds, batch_size, prev_opt_name, n_iter, \
        init_points, init_guesses = havsim.parse_args(arg_names, default_args, desc_str, arg_de, 0)
    gamma_bounds.extend(xi_bounds)


    def evaluate_crash_rate(my_args):
        gamma_parameters = [my_args[str(i)] for i in range(5)]
        xi_parameters = [my_args[str(5 + i)] for i in range(2)]
        can_evaluate = False
        while not can_evaluate:
            pool = multiprocessing.Pool(n_workers)


    now = datetime.now()
    print('\nStarting job \'' + save_name + '\' at ' + now.strftime("%H:%M:%S"))
    pickle_path = os.path.join(os.path.dirname(__file__), 'pickle files')
    if not os.path.exists(pickle_path):
        print('Warning: the directory ' + pickle_path + ' does not exist.')
        os.makedirs(pickle_path)

    havsim.bayes_opt(evaluate_crash_rate, gamma_bounds, n_iter=n_iter, init_points=init_points,
                     init_guesses=init_guesses, save_name=save_name, prev_opt_name=prev_opt_name)
