"""Simulation of traffic and crashes on I94 in Ann Arbor."""
from make_simulation import e94
import havsim.plotting as hp
import matplotlib.pyplot as plt
import pickle
import multiprocessing

n_processes = 10
replications = 2
make_plots = False
save_output = True
save_crashes_only = True
save_name = 'pickle files/e94_sim_1'

def do_simulation():
    simulation, laneinds = e94()
    near_miss = 0
    rear_end = 0
    sideswipe = 0
    vmt = 0
    for i in range(replications):
        all_vehicles = simulation.simulate()

        for crash in simulation.crashes:
            if crash[0].crashed == 'rear end':
                rear_end += 1
            else:
                sideswipe += 1
        for veh in all_vehicles:
            near_miss += len(veh.near_misses)
            vmt += veh.posmem[-1] - veh.posmem[0]

        if save_crashes_only:
            pass
        if i < replications - 1:
            simulation.reset()


def get_crashed_veh_and_leaders():
    pass

print('\n-----------SUMMARY-----------')
print('average near misses: {:n}'.format(near_miss/replications))
print('average rear end crashes: {:n}'.format(rear_end/replications))
print('average sideswipe crashes: {:n}'.format(sideswipe/replications))
print('average vmt (miles): {:.0f}'.format(vmt/replications/1609.34))

if save_output:
    with open(save_name+'.pkl', 'wb') as f:
        pickle.dump([all_vehicles, laneinds], f)
if make_plots:
    sim, siminfo = hp.plot_format(all_vehicles, laneinds)
    sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (8000, 10000))

    hp.platoonplot(sim2, None, siminfo2, lane=1, opacity=0, timerange=[6000, 10000])
    # hp.platoonplot(sim2, None, siminfo2, lane=2, opacity=0)

    hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))

    hp.plotflows(sim, [[7000, 7100], [9230, 9330], [11000, 11100]], [0, 18000], 300, h=.2)

    ani = hp.animatetraj(sim2, siminfo2, usetime=list(range(2000, 4000)), spacelim=(8000, 10000), lanelim=(3, -1))
    plt.show()

