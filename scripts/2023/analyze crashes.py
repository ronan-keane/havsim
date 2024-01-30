"""Plot crashes, near misses and analyze crash rate from saved data from script safety sim."""
import pickle
import havsim
import havsim.plotting as hp
import numpy as np
import matplotlib.pyplot as plt

# -------  SETTINGS  ------- #
saved_sim = 'e94_16_17'
min_crash_plots = 0
max_crash_plots = 1
show_plots = False
save_plots = True
# -------------------------- #


def prepare_speed_plot(my_veh, start, end, crash_type=None):
    crash_type = my_veh.crashed[0] if crash_type is None else crash_type
    # get stochastic behavior - get state, relax, acc, lc_acc at every timestep
    start = max(start, my_veh.start)
    end = min(end, my_veh.end) if my_veh.end is not None else min(end, my_veh.start + len(my_veh.posmem)-1)
    veh_pos = my_veh.posmem[start-my_veh.start:end-my_veh.start+1]
    veh_speed = my_veh.speedmem[start-my_veh.start:end-my_veh.start+1]

    lead_pos = []
    lead_speed = []
    lead_len = []
    lead_durations = []
    veh_leadmem = my_veh.leadmem[havsim.helper.count_leadmem(my_veh, start):havsim.helper.count_leadmem(my_veh, end)+1]
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
        lead_durations.append(cur_end-cur_start+1)

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
    test_veh = havsim.simulation.Vehicle('test', None, **params)

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
    if crash_type == 'near miss':
        t = (np.array(list(range(start, end+1))) - end-5)*dt
    else:
        t = (np.array(list(range(start, end+1))) - my_veh.crash_time)*dt
    lead_pos, lead_len = np.array(lead_pos), np.array(lead_len)
    hd = lead_pos - lead_len - np.array(veh_pos)
    test_hd = lead_pos - lead_len - np.array(test_posmem)
    return t, hd, test_hd, np.array(veh_speed), np.array(test_speedmem), lead_durations, \
        np.array(acc), np.array(test_accmem), crash_type, my_veh.vehid


def do_speed_plot(args):
    t, hd, test_hd, spd, test_spd, lead_durations, acc, test_acc, crash_type, vehid = args
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('gap, speed, and acceleration before '+str(crash_type)+' (vehicle '+str(vehid)+')')
    ax1 = plt.subplot(3, 1, 2)
    a1 = ax1.plot(t, spd, color='C0', alpha=.8)
    a2 = ax1.plot(t, test_spd, color='C0', linestyle='dashed')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('speed (m/s)')
    ax1.legend(handles=[a1[0], a2[0]], labels=['stochastic', 'deterministic'])

    ax11 = plt.subplot(3, 1, 1)
    prev = 0
    for dur in lead_durations:
        a3 = ax11.plot(t[prev:prev+dur], hd[prev:prev+dur], color='C1', alpha=.8)
        a4 = ax11.plot(t[prev:prev+dur], test_hd[prev:prev+dur], color='C1', linestyle='dashed')
        prev = prev+dur
    ax11.set_xlabel('time (s)')
    ax11.set_ylabel('space gap (m)')
    ax11.legend(handles=[a3[0], a4[0]], labels=['stochastic', 'deterministic'])

    ax2 = plt.subplot(3, 1, 3)
    a5 = ax2.plot(t[:-1], acc, color='C2', alpha=.8)
    a6 = ax2.plot(t[:-1], test_acc, color='C2', linestyle='dashed')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('acceleration (m/s/s)')
    ax2.legend(handles=[a5[0], a6[0]], labels=['acceleration', 'acceleration (deterministic)'])
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    # todo printout
    # test sim settings / different BC (stochastic)
    e94_rear_ends = [0, 1, 1, 1, 0, 1, 3, 12, 23, 9, 5, 7, 5, 3, 11, 53, 92, 105, 81, 21, 4, 6, 2, 3]
    e94_sideswipes = [3, 0, 0, 1, 2, 3, 4, 5, 11, 10, 6, 7, 3, 9, 5, 15, 13, 21, 18, 10, 8, 4, 4, 8]

    with open('pickle files/' + saved_sim, 'rb') as f:
        all_vehicles_list, laneinds = pickle.load(f)
    with open('pickle files/' + saved_sim + '_config'+'.config', 'rb') as f:
        config = pickle.load(f)
    timesteps_before, timesteps_after, dt = config['timesteps_before'], config['timesteps_after'], config['dt']

    rear_end, sideswipes, near_misses, severity = [], [], [], []
    n_crashed_veh, n_near_miss_veh = 0, 0

    params = None
    for count, all_vehicles in enumerate(all_vehicles_list):
        if not params:
            if len(all_vehicles) > 0:
                if hasattr(all_vehicles[0], 'gamma_parameters'):
                    params = (all_vehicles[0].gamma_parameters, all_vehicles[0].xi_parameters,
                              type(all_vehicles[0]).__name__)
        all_vehicles = havsim.simulation.vehicles.reload(all_vehicles)
        crashes_only = {}
        for veh in all_vehicles:
            if veh.crashed:
                if veh.crashed[1] in crashes_only:
                    crashes_only[veh.crashed[1]].append(veh)
                else:
                    crashes_only[veh.crashed[1]] = [veh]
        for crash_veh_list in crashes_only.values():
            n_crashed_veh += len(crash_veh_list)
            crash_times = [veh.crash_time for veh in crash_veh_list]
            t_start, t_end = min(crash_times) - timesteps_before, max(crash_times) + timesteps_after
            platoon = havsim.helper.add_leaders(crash_veh_list, t_start, t_end)
            need_speed_plots = []
            for veh in crash_veh_list[:2]:
                mem = veh.leadmem[havsim.helper.count_leadmem(veh, veh.crash_time)]
                if mem[0] in crash_veh_list:
                    need_speed_plots.append(veh)
            if len(need_speed_plots) == 0:
                need_speed_plots.extend(crash_veh_list[:2])
            need_speed_plots.extend(crash_veh_list[2:])
            for veh in crash_veh_list:
                if veh.crash_time == veh.crashed[1]:
                    if veh.crashed[0] == 'rear end':
                        rear_end.append(((t_start, t_end), platoon, need_speed_plots, count, None))
                    else:
                        sideswipes.append(((t_start, t_end), platoon, need_speed_plots, count, None))
                    break
            veh1, veh2 = crash_veh_list[0], crash_veh_list[1]
            severity.append(abs(veh1.speedmem[veh1.crash_time-veh1.start]-veh2.speedmem[veh2.crash_time-veh2.start]))
            for veh in crash_veh_list[2:]:
                other = veh.leadmem[havsim.helper.count_leadmem(veh, veh.crash_time)][0]
                if veh.crashed[0] == 'sideswipe':
                    if other is not None:
                        if other in crash_veh_list[:2]:
                            pass
                    else:
                        other = crash_veh_list[0]
                severity.append(abs(other.speedmem[veh.crash_time-other.start]-veh.speedmem[veh.crash_time-veh.start]))
    if params:
        print('\nStochastic parameters (gamma, xi): '+str(params[0])+', '+str(params[1])+',  ('+str(params[2])+')')
    print('number of crashes: {:n} ({:n} rear ends) ({:n} vehicles)'.format(len(rear_end)+len(sideswipes),
                                                                            len(rear_end), n_crashed_veh))
    for count, all_vehicles in enumerate(all_vehicles_list):
        for veh in all_vehicles:
            if len(veh.near_misses) == 0:
                continue
            n_near_miss_veh += 1
            for times in veh.near_misses:
                t_start, t_end = times[0] - timesteps_before, times[1] + timesteps_after
                near_misses.append(((t_start, t_end), havsim.helper.add_leaders([veh], t_start, t_end),
                                    [veh], count, 'near miss'))
    print('number of near misses: {:n} ({:n} vehicles)'.format(len(near_misses), n_near_miss_veh))

    all_sim = []
    for all_vehicles in all_vehicles_list:
        sim, siminfo = hp.plot_format(all_vehicles,  laneinds)
        all_sim.append((sim, siminfo))

    all_ani, plot_crashes = [], []
    plot_crashes.extend(rear_end[min_crash_plots:max_crash_plots])
    plot_crashes.extend(sideswipes[min_crash_plots:max_crash_plots])
    plot_crashes.extend(near_misses[min_crash_plots:max_crash_plots])
    inds = list(range(min_crash_plots, min_crash_plots + len(rear_end[min_crash_plots:max_crash_plots])))
    inds.extend(list(range(min_crash_plots, min_crash_plots + len(sideswipes[min_crash_plots:max_crash_plots]))))
    inds.extend(list(range(min_crash_plots, min_crash_plots + len(near_misses[min_crash_plots:max_crash_plots]))))
    my_crash_types = ['rear end']*len(rear_end[min_crash_plots:max_crash_plots])
    my_crash_types.extend(['sideswipe']*len(sideswipes[min_crash_plots:max_crash_plots]))
    my_crash_types.extend(['near miss'] * len(near_misses[min_crash_plots:max_crash_plots]))
    print('\nplotting {:n} rear ends, {:n} sideswipes, {:n} near misses'.format(
        len(rear_end[min_crash_plots:max_crash_plots]), len(sideswipes[min_crash_plots:max_crash_plots]),
        len(near_misses[min_crash_plots:max_crash_plots])))
    fig = plt.figure()
    plt.hist(severity, bins=[0+i*.5 for i in range(20)])
    plt.xlabel('speed difference at crash (m/s)')
    plt.ylabel('frequency')
    if save_plots:
        fig.savefig('pickle files/animations/'+saved_sim+'_severity.png', dpi=200)
    for ind_count, cur in enumerate(plot_crashes):
        times, platoon, need_speed_plots, count, my_crash_type = cur
        t_start, t_end = times
        min_p, max_p = [], []
        for veh in platoon:
            min_p.append(veh.posmem[max(t_start - veh.start, 0)])
            max_p.append(veh.posmem[min(t_end - veh.start, len(veh.posmem) - 1)])
        sim, siminfo = all_sim[count]
        title_str = 'Events: '
        veh = need_speed_plots[0]
        cur_event = veh.crashed[0] if my_crash_type is None else my_crash_type
        title_str = title_str + cur_event + ' (vehicle '+str(veh.vehid)+')'
        for veh in need_speed_plots[1:]:
            cur_event = veh.crashed[0] if my_crash_type is None else my_crash_type
            title_str = title_str + ', ' + cur_event + ' (vehicle '+str(veh.vehid)+')'

        crash_str = my_crash_types[ind_count]
        filename = 'pickle files/animations/' + saved_sim + ' - ' + crash_str
        save_name = filename if save_plots else None
        ani = hp.animatetraj(sim, siminfo, platoon=[i.vehid for i in platoon], usetime=list(range(t_start, t_end+1)),
                             spacelim=(min(min_p)-5, max(max_p)+3), lanelim=(3, -1), show_id=True, show_axis=True,
                             title=title_str, save_name=save_name)
        all_ani.append(ani)

        for counter2, veh in enumerate(need_speed_plots):
            fig = do_speed_plot(prepare_speed_plot(veh, t_start, t_end, crash_type=my_crash_type))
            if save_plots:
                fig.savefig(filename+' - '+str(counter2)+'.png', dpi=200)
    if show_plots:
        plt.show()
