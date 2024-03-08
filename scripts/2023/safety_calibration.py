import havsim
from make_simulation import e94
import multiprocessing
import tqdm
import sys

# -------  SETTINGS  ------- #
all_times = [[14, 15], [16, 17]]
n_simulations = 300
n_workers = 50
batch_size = 150
gamma_parameters = [-.08, .35, .3, 1.5, 1.5]
xi_parameters = [.2, 6]
# -------------------------- #

def do_simulation(parameters):
    return


def evaluate_crash_rate(parameters):
    pass  # l2 crash rate target, regularizer for near miss + nveh/crash


if __name__ == '__main__':
    print('hi')
