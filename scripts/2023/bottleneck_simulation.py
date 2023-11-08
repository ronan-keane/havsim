"""Simple example simulation."""
from make_simulation import merge_bottleneck
import havsim.plotting as hp
import matplotlib.pyplot as plt
import pickle
import scipy.interpolate as ssi

make_plots = True
save_output = False
save_name = 'pickle files/bottleneck_sim_0'

t = [3600*i for i in range(12)]
# main = ssi.interp1d(t, [100/3600, 3200/3600/2, 3600/3600/2, 3600/3600/2, 3600/3600/2, 3400/3600/2, 3000/3600/2, 3000/3600/2, 3000/3600/2, 3000/3600/2, 3000/3600/2, 3000/3600/2])
# onramp = ssi.interp1d(t, [0/3600, 200/3600, 600/3600, 600/3600, 600/3600, 600/3600, 600/3600, 200/3600, 200/3600, 1200/3600, 1200/3600, 1200/3600])
main = lambda timeind: 3200/3600/2
onramp = lambda timeind: 1150/3600
timesteps = 5000
simulation, laneinds = merge_bottleneck(main_inflow=main, onramp_inflow=onramp, timesteps=timesteps)

all_vehicles = simulation.simulate(verbose=True)

if save_output:
    with open(save_name + '.pkl', 'wb') as f:
        pickle.dump([all_vehicles, laneinds], f)
if make_plots:
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    hp.platoonplot(sim, None, siminfo, lane=1, opacity=0, timerange=[1000, 5000])

    # hp.plotspacetime(sim, siminfo, timeint=150, xint=30, lane=1, speed_bounds=(0, 35))
    hp.plotspacetime(sim, siminfo, timeint=150, xint=30, lane=0, speed_bounds=(0, 35))

    hp.plotflows(sim, [[900, 1000], [1300, 1400], [1900, 2000]], [0, timesteps], 300, h=.2)

    ani = hp.animatetraj(sim, siminfo, usetime=list(range(1000, 4000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))
    # ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(10000, timesteps)), show_id=False, spacelim=(0, 2000),
    #                      lanelim=(3, -1))
    plt.show()