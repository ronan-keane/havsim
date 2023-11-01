import pickle
import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt

timesteps_before = 100
timesteps_after = 25
include_leaders = True
max_crash_plots = 5
saved_sim = '/scripts/2023/e94_sim_0.pkl'

with open(saved_sim, 'rb') as f:
    all_vehicles, laneinds = pickle.load(f)
all_vehicles = hs.vehicles.reload(all_vehicles)
crashes = {}
for veh in all_vehicles:
    if veh.crashed:
        if veh.crash_time in crashes:
            crashes[veh.crash_time].append(veh)
        else:
            crashes[veh.crash_time] = [veh]
if len(crashes) == 0:
    print('no crashes in data')


def count_leadmem(my_veh, timeind):
    if timeind < my_veh.start:
        return 0
    for count, leadmem in enumerate(my_veh.leadmem[:-1]):
        if leadmem[1] <= timeind < my_veh.leadmem[count + 1][1]:
            break
    else:
        count = len(my_veh.leadmem) - 1
    return count


def prepare_speed_plot(my_veh, start, end):
    out = {}
    # find type of crash
    lead = my_veh.leadmem[count_leadmem(my_veh, my_veh.crash_time)]
    if len(my_veh.lanemem) > 1:
        lc_times = [lc[1] for lc in my_veh.lanemem[1:]]
    else:
        lc_times = []
    if len(lead.lanemem) > 1:
        lc_times.extend([lc[1] for lc in lead.lanemem[1:]])
    if veh.crash_time - 8 in lc_times:
        crash_type = 'sideswipe'
    else:
        crash_type = 'rear_end'

    # get stochastic behavior


all_platoons = []
all_usetime = []
all_spacelim = []
all_speed = []
for crash_time, crash in crashes.items():
    # animation
    t_start, t_end = crash_time-timesteps_before, crash_time+timesteps_after
    platoon = []
    for veh in crash:
        platoon.append(veh)
        last_count = count_leadmem(veh, t_end)
        first_count = count_leadmem(veh, t_start)
        for mem in veh.leadmem[first_count:last_count+1]:
            if mem[0] is not None:
                platoon.append(mem[0])
                if include_leaders:
                    for mem2 in mem[0].leadmem[count_leadmem(mem[0], t_start):count_leadmem(mem[0], t_end)+1]:
                        if mem2[0] is not None:
                            platoon.append(mem2[0])
    platoon = list(set(platoon))
    min_p, max_p = [], []
    for veh in platoon:
        min_p.append(veh.posmem[max(t_start-veh.start, 0)])
        max_p.append(veh.posmem[min(t_end - veh.start, len(veh.posmem)-1)])
    all_platoons.append([i.vehid for i in platoon])
    all_usetime.append((t_start, t_end))
    all_spacelim.append((min(min_p)-10, max(max_p)+10))

    # plot of speed
    speed = []
    for veh in crash:
        mem = veh.leadmem[count_leadmem(veh, crash_time)]
        if mem[0] in crash or len(crash)==1:
            pass


sim, siminfo = hp.plot_format(all_vehicles, laneinds)
all_ani = []
for count in range(min(len(all_platoons), max_crash_plots)):
    ani = hp.animatetraj(sim, siminfo, platoon=all_platoons[count], usetime=list(range(*all_usetime[count])),
                         spacelim=all_spacelim[count], lanelim=(3, -1), show_id=True, interval=20)
    all_ani.append(ani)
    plt.show()

