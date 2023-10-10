from make_simulation import merge_bottleneck
import havsim.plotting as hp
import matplotlib.pyplot as plt
import time
import pickle
import scipy.interpolate as ssi

t = [3600*i for i in range(12)]
main = ssi.interp1d(t, [200/3600, 1600/3600, 1800/3600, 1900/3600, 1800/3600, 1800/3600, 1600/3600, 1600/3600, 1400/3600, 1200/3600, 1200/3600, 1200/3600])
onramp = ssi.interp1d(t, [0/3600, 200/3600, 300/3600, 600/3600, 600/3600, 500/3600, 500/3600, 300/3600, 200/3600, 200/3600, 200/3600, 200/3600])
simulation, laneinds = merge_bottleneck(main_inflow=main, onramp_inflow=onramp)
timesteps = 3600*11
make_plots = True
save_output = False
save_name = 'bottleneck_sim_0'

start = time.time()
simulation.simulate(timesteps)
end = time.time()

all_vehicles = simulation.prev_vehicles
all_vehicles.extend(simulation.vehicles)
print('simulation time is ' + str(end - start) + ' over ' + str(
    sum([timesteps - veh.start + 1 if veh.end is None else veh.end - veh.start + 1
         for veh in all_vehicles])) + ' timesteps')

if make_plots or save_output:
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    if save_output:
        with open(save_name+'.pkl', 'wb') as f:
            pickle.dump([sim, siminfo], f)
    if make_plots:
        hp.plotspacetime(sim, siminfo, timeint=50, xint=30, lane=1, speed_bounds=(0, 35))
        hp.plotspacetime(sim, siminfo, timeint=50, xint=30, lane=0, speed_bounds=(0, 35))

        hp.plotflows(sim, [[0, 100], [1300, 1400], [1900, 2000]], [0, timesteps], 150, h=.2)

        ani = hp.animatetraj(sim, siminfo, usetime=list(range(7200, 10000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))
        ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(13000, timesteps)), show_id=False, spacelim=(0, 2000),
                             lanelim=(3, -1))
        plt.show()