"""Plot crashes and near misses from saved data."""
import pickle
import havsim.simulation as hs
import havsim.plotting as hp
import numpy as np
import matplotlib.pyplot as plt

timesteps_before = 100
timesteps_after = 25
include_leaders = True
max_crash_plots = 5
dt = .2
saved_sim = 'pickle files/e94_sim_0.pkl'

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
    lead = my_veh.leadmem[count_leadmem(my_veh, my_veh.crash_time)][0]
    if len(my_veh.lanemem) > 1:
        lc_times = [lc[1] for lc in my_veh.lanemem[1:]]
    else:
        lc_times = []
    if len(lead.lanemem) > 1:
        lc_times.extend([lc[1] for lc in lead.lanemem[1:]])
    if my_veh.crash_time - 8 in lc_times:
        crash_type = 'sideswipe'
    else:
        crash_type = 'rear end'

    # get stochastic behavior - get state, relax, acc, lc_acc at every timestep
    start = max(start, my_veh.start)
    end = min(end, my_veh.end) if my_veh.end is not None else min(end, my_veh.start + len(my_veh.posmem)-1)
    veh_pos = my_veh.posmem[start-my_veh.start:end-my_veh.start+1]
    veh_speed = my_veh.speedmem[start-my_veh.start:end-my_veh.start+1]

    lead_pos = []
    lead_speed = []
    lead_len = []
    veh_leadmem = my_veh.leadmem[count_leadmem(my_veh, start):count_leadmem(my_veh, end)+1]
    for i, leadmem in enumerate(veh_leadmem):
        cur_end = end if i == len(veh_leadmem) - 1 else veh_leadmem[i+1][1]-1
        cur_start = start if i == 0 else leadmem[1]
        if leadmem[0] is not None:
            lead_pos.extend(leadmem[0].posmem[cur_start-leadmem[0].start:cur_end-leadmem[0].start+1])
            lead_speed.extend(leadmem[0].speedmem[cur_start - leadmem[0].start:cur_end - leadmem[0].start + 1])
            lead_len.extend([leadmem[0].len]*(cur_end-cur_start+1))
        else:
            lead_pos.extend(my_veh.posmem[cur_start-my_veh.start:cur_end-my_veh.start+1])
            lead_speed.extend([0] * (cur_end-cur_start+1))
            lead_len.extend([0] * (cur_end-cur_start+1))

    acc = []
    for i in range(len(veh_speed)-1):
        acc.append((veh_speed[i+1] - veh_speed[i])/dt)
    relax = [(0, 0) for i in range(len(veh_speed))]
    for cur_relax, relax_start in my_veh.relaxmem:
        relax_end = relax_start + len(cur_relax) - 1
        cur_start = max(relax_start, start)
        cur_end = min(relax_end, end)
        if cur_end >= cur_start:
            relax[cur_start-start:cur_end-start+1] = cur_relax[cur_start-relax_start:cur_end-relax_start+1]
    lc_acc = my_veh.lc_accmem[start-my_veh.start:end-my_veh.start]

    # make deterministic behavior
    params = {'cf_parameters': my_veh.cf_parameters, 'relax_parameters': my_veh.relax_parameters}
    test_veh = hs.Vehicle('test', None, **params)

    test_pos, test_speed = veh_pos[0], veh_speed[0]
    test_posmem = [test_pos]
    test_speedmem = [test_speed]
    test_accmem = []
    p = test_veh.cf_parameters
    pr = test_veh.relax_parameters
    min_acc = my_veh.minacc
    for ind in range(end-start):
        # call to set_cf
        if lead_len[ind] == 0:
            test_acc = test_veh.free_cf(p, test_speed)
        else:
            test_hd = lead_pos[ind] - test_pos - lead_len[ind]
            lspd = lead_speed[ind]
            test_relax = relax[ind]
            if test_relax[0] != 0 or test_relax[1] != 0:
                currelax, currelax_v = test_relax[0], test_relax[1]
                ttc = max(test_hd - 2 - pr[2]*test_speed, 0)/(test_speed - lspd + 1e-6)
                if pr[3] > ttc >= 0:
                    currelax = currelax * (ttc / p[3]) ** 2 if currelax > 0 else currelax
                    currelax_v = currelax_v * (ttc / p[3]) ** 2 if currelax_v > 0 else currelax_v
                test_hd += currelax
                lspd += currelax_v
            if test_hd < 0:
                test_acc = -100
            else:
                test_acc = test_veh.cf_model(p, [test_hd, test_speed, lspd])

        # call to update
        test_lc_acc = lc_acc[ind]
        test_acc = test_acc + test_lc_acc
        test_acc = max(test_acc, min_acc)
        nextspeed = test_speed + test_acc*dt
        if nextspeed < 0:
            nextspeed = 0
            test_acc = -test_speed/dt
        test_pos += test_speed*dt
        test_speed = nextspeed
        test_posmem.append(test_pos)
        test_speedmem.append(test_speed)
        test_accmem.append(test_acc)

    t = (np.array(list(range(start, end+1))) - my_veh.crash_time)*dt
    lead_pos, lead_len = np.array(lead_pos), np.array(lead_len)
    hd = lead_pos - lead_len - np.array(veh_pos)
    test_hd = lead_pos - lead_len - np.array(test_posmem)
    return t, hd, test_hd, np.array(veh_speed), np.array(test_speedmem), np.array(acc), np.array(test_accmem), \
        crash_type


def do_speed_plot(args):
    t, hd, test_hd, spd, test_spd, acc, test_acc, crash_type = args
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('gap, speed, and acceleration before and after '+str(crash_type))
    ax1 = plt.subplot(3, 1, 1)
    a1 = ax1.plot(t, spd, color='C0', alpha=.2)
    a2 = ax1.plot(t, test_spd, color='C0', linestyle='dashed')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('speed (m/s)')

    ax11 = ax1.twinx()
    a3 = ax11.plot(t, hd, color='C1', alpha=.2)
    a4 = ax11.plot(t, test_hd, color='C1', linestyle=(0, (5, 10)))

    ax11.set_ylabel('space gap (m)')
    ax1.legend(handles=[a3[0], a1[0], a4[0], a2[0]], labels=['gap', 'speed', 'gap (deterministic)', 'speed (deterministic)'])
    ax1.set_title('gap and speed before and after '+str(crash_type))
    ax2 = plt.subplot(2, 1, 2)
    a5 = ax2.plot(t[:-1], acc, color='C2', alpha=.2)
    a6 = ax2.plot(t[:-1], test_acc, color='C2', linestyle=(0, (5, 10)))
    ax2.set_title('acceleration before and after '+str(crash_type))
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('acceleration (m/s/s)')
    ax2.legend(handles=[a5[0], a6[0]], labels=['acceleration', 'acceleration (deterministic)'])
    fig.tight_layout()


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
        if mem[0] in crash or len(crash) == 1:
            all_speed.append(prepare_speed_plot(veh, t_start, t_end))


sim, siminfo = hp.plot_format(all_vehicles, laneinds)
all_ani = []
for count in range(min(len(all_platoons), max_crash_plots)):
    ani = hp.animatetraj(sim, siminfo, platoon=all_platoons[count], usetime=list(range(*all_usetime[count])),
                         spacelim=all_spacelim[count], lanelim=(3, -1), show_id=True, interval=20)
    all_ani.append(ani)
    do_speed_plot(all_speed[count])
plt.show()

