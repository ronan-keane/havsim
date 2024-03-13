"""Example of loading saved data and using plotting api."""
import pickle
import havsim as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    arg_names = ['save_name', 'show_plots', 'plot_times']
    default_args = ['e94_16_17_test', False, None]
    desc_str = 'Example of loading saved data and using plotting functions'
    arg_desc = ['str, name of file to load from pickle files folder, not including extension',
                'bool, whether or not to show plots (always save them)',
                'tuple of two floats. If specified, only show those times in all plots. ' +
                'If None, uses the full time for spacetime/flow plot, and a shorter time for others']
    n_pos_arg = 0
    save_name, show_plots, plot_times = hs.helper.parse_args(arg_names, default_args, desc_str, arg_desc, n_pos_arg)

    pickle_path = os.path.join(os.path.dirname(__file__), 'pickle files')
    file_path = os.path.join(pickle_path, save_name + '.pkl')
    assert os.path.exists(file_path), 'the file \'' + file_path + '\' does not exist'
    with open(file_path, 'rb') as f:
        all_vehicles, lanes = pickle.load(f)
    config_path = os.path.join(pickle_path, save_name+'_config.config')
    assert os.path.exists(config_path), 'the file \'' + file_path + '\' exists, but the .config does not'
    with open(os.path.join(pickle_path, save_name+'_config.config'), 'rb') as f:
        config = pickle.load(f)
        start, end = config.get('use_times', [16, 17])
        if plot_times is None:
            start2, end2 = start + (end-start)*.35, start + (end-start)*.65
        elif len(plot_times) == 2:
            start2, end2 = plot_times
            start, end = plot_times
    animation_path = os.path.abspath(os.path.join(pickle_path, '..', 'plots and animations'))
    if not os.path.exists(animation_path):
        os.makedirs(animation_path)

    all_vehicles = hs.reload(all_vehicles[0], lanes)
    sim, siminfo = hp.plot_format(all_vehicles, lanes)
    sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (7300, 9300))

    fig = hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 40))
    fig.savefig(os.path.join(animation_path, save_name+'_spacetime_lane1.png'), dpi=200)

    fig, fig2 = hp.plotflows(sim, [[5200, 5300], [7600, 7700], [8400, 8500], [9250, 9350]],
                             [18000*start, 18000*end], 300, h=.2)
    fig.savefig(os.path.join(animation_path, save_name + '_fundamental_diagram.png'), dpi=200)
    fig2.savefig(os.path.join(animation_path, save_name + '_flows.png'), dpi=200)

    fig = hp.platoonplot(sim2, None, siminfo2, lane=1, opacity=0, timerange=[int(18000*start2), int(18000*end2)])
    fig.savefig(os.path.join(animation_path, save_name + '_trajectories.png'), dpi=200)

    ani_filepath = os.path.join(animation_path, save_name + '_movie')
    ani = hp.animatetraj(sim2, siminfo2, usetime=list(range(int(18000*start2), int(18000*end2))),
                         show_id=False, lanelim=(3.5, -1), save_name=ani_filepath)
    if show_plots:
        plt.show()
