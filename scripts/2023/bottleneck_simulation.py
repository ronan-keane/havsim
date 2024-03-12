"""Simple example simulation."""
from make_simulation import merge_bottleneck
import matplotlib.pyplot as plt
import os
import pickle
import havsim
import havsim.plotting as hp


if __name__ == '__main__':
    arg_names = ['save_name', 'save_output', 'make_plots', 'make_animation']
    default_args = ['bottleneck_sim_0', False, True, True]
    description_str = 'Simple example simulation of a merge bottleneck on a 2-lane highway.'
    arg_descriptions = ['if saving the output, save_name is a str for the filename, not including the extension',
                        'bool, if True then save the simulation result in save_name',
                        'bool, if True then make plots of the simulation', 'bool, if True then make animation']
    n_pos_arg = 0
    save_name, save_output, make_plots, make_animation = \
        havsim.helper.parse_args(arg_names, default_args, description_str, arg_descriptions, n_pos_arg)

    simulation, laneinds = merge_bottleneck()
    all_vehicles = simulation.simulate()

    if save_output:
        pickle_path = os.path.join(os.path.dirname(__file__), 'pickle files')
        if not os.path.exists(pickle_path):
            print('Warning: the directory ' + pickle_path + ' does not exist.')
            os.makedirs(pickle_path)
        with open(os.path.join(pickle_path, save_name + '.pkl'), 'wb') as f:
            pickle.dump([all_vehicles, laneinds], f)
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    if make_plots:
        hp.platoonplot(sim, None, siminfo, lane=1, opacity=0, timerange=[1000, 5000])

        hp.plotspacetime(sim, siminfo, timeint=150, xint=30, lane=1, speed_bounds=(0, 40))

        hp.plotflows(sim, [[100, 200], [800, 900], [1300, 1400]], [0, simulation.timesteps], 300, h=.2)
    if make_animation:
        ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(10000, simulation.timesteps)), show_id=False,
                              spacelim=(0, 2000), lanelim=(3, -1))
    plt.show()
