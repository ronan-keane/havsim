import havsim
from make_simulation import e94
import multiprocessing
import tqdm
import os


def do_simulation(my_args):
    use_pbar, my_use_times, my_gamma_parameters, my_xi_parameters = my_args
    simulation, my_lanes = e94(my_use_times, my_gamma_parameters, my_xi_parameters)
    all_vehicles, my_stats = simulation.simulate(pbar=use_pbar, return_stats=True)
    return *my_stats, len(all_vehicles)


if __name__ == '__main__':
    arg_names = ['save_name', 'n_simulations', 'n_workers', 'use_times', 'gamma_bounds', 'xi_bounds', 'batch_size'
                 'prev_opt_name', 'n_iter', 'init_points', 'init_guesses']
    default_args = ['e94_calibration_1', 300,  round(.4*multiprocessing.cpu_count()), [[11, 12], [16, 17]],
                    [], [], 300, None, 100, 0, []]
    desc_str = 'Calibrate gamma/xi parameters by simulating the crash rate under realistic conditions, '\
        'and compare against crashes data. This is an intensive procedure which requires running many simulations.'
    arg_de = \
        ['str, name of file for saving optimizer result (not including extension). Inside of the ./pickle files '
         'folder, the optimizer state is saved in save_name.json. The save_name_res.pkl is a list of each '
         'objective function evaluation done by the optimizer.',
         'int, minimum number of simulations to run. We may run additional if the confidence interval is too large.',
         'int, number of simulations to run in parallel',
         'list of lists, each inner list has 2 floats, giving the starting/ending times (0-24) to calibrate. The'
         'objective function is summed over each given time interval.',
         'list of tuples, where each tuple is the upper/lower bounds for the gamma parameter with that index',
         'list of tuples, where each tuple is the upper/lower bounds for the xi parameter with that index',
         'int, maximum number of simulations to run before clearing memory.',
         'if not None, the save_name from a previous run, and the .json log will be loaded, which will restart'
         ' the opimizer.'
         'int, number of optimization steps to perform (number of objective function evaluations)',
         'int, number of randomly sampled initial points for the optimizer',
         'if not None, list of lists, where each list is a set of parameters (7 floats) for the optimizer'
         'to evaluate.']
    save_name, n_simulations, n_workers, use_times, gamma_bounds, xi_bounds, batch_size, prev_opt_name, n_iter, \
        init_points, init_guesses = havsim.parse_args(arg_names, default_args, desc_str, arg_de, 0)



    def evaluate_crash_rate(my_args):
        gamma_parameters = [my_args[str(i)] for i in range(5)]
        xi_parameters = [my_args[str(5 + i)] for i in range(2)]
        pass  # l2 crash rate target, regularizer for near miss + nveh/crash