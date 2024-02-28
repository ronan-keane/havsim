"""Simple example simulation."""
from make_simulation import merge_bottleneck
import havsim.plotting as hp
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as ssi

make_plots = True
save_output = False
save_name = 'pickle files/bottleneck_sim_0'

main = lambda timeind: 3200/3600/2 * min(timeind, 10000)/10000
onramp = lambda timeind: 800/3600 * min(timeind, 12000)/12000
timesteps = 20000
simulation, laneinds = merge_bottleneck(main_inflow=main, onramp_inflow=onramp, timesteps=timesteps)

for i in range(100):
    all_vehicles = simulation.simulate(timesteps=10000)
    simulation.reset()
    del all_vehicles

if save_output:
    with open(save_name + '.pkl', 'wb') as f:
        pickle.dump([all_vehicles, laneinds], f)
if make_plots:
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    # hp.platoonplot(sim, None, siminfo, lane=1, opacity=0, timerange=[1000, 5000])

    # hp.plotspacetime(sim, siminfo, timeint=150, xint=30, lane=1, speed_bounds=(0, 35))
    # hp.plotspacetime(sim, siminfo, timeint=150, xint=30, lane=0, speed_bounds=(0, 40))
    #
    # hp.plotflows(sim, [[100, 200], [800, 900], [1100, 1200], [1400, 1500]], [0, timesteps], 300, h=.2)
    # plt.show()

    # ani = hp.animatetraj(sim, siminfo, usetime=list(range(10000, 13000)), show_id=False, spacelim=(800, 1500), lanelim=(3.5, -1))
    # ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(10000, timesteps)), show_id=False, spacelim=(0, 2000),
    #                      lanelim=(3, -1))
    # plt.show()