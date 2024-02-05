
# Also need to run first cell of relax_results_cf_calibration file
import havsim.simulation.calibration as hc
import havsim.simulation.calibration_models as hm
import math
import havsim.calibration.helper as helper
import numpy as np
import pickle

def analyze_res(res_list, veh_list, meas, platooninfo, dt, kwargs, mergeind = math.inf, times = 100, realistic = 1.1):
    out = {'overall mse':None, 'mse near LC':None, 'mse for merges':None, 'mse for many LC':None, 'realistic acc':None, 'short lc':None}
    temp = [ga['fun']/(3.28084**2) for ga in res_list]
    out['overall mse'] = (np.mean(temp), np.median(temp), np.std(temp))

    nearlc = []
    merge = []
    manylc = []
    real = []
    out2 = []
    shortlc = []
    for count, veh in enumerate(veh_list):
        res = res_list[count]
        cal = hc.make_calibration([veh], meas, platooninfo, dt, **kwargs)
        cal.simulate(res['x'])


        leadinfo = helper.makeleadinfo([veh], platooninfo, meas)
        lctimes = [cur[1] for cur in leadinfo[0]]
        rinfo = helper.makerinfo([veh], platooninfo, meas, leadinfo)
        if len(lctimes) > len(rinfo[0]):  # this handles an edge case
            rinfo[0].insert(0, [None, -1])
        t_nstar, t_n, T_nm1 = platooninfo[veh][:3]
        if lctimes[0] == t_nstar:
            lctimes = lctimes[1:]

        if len(leadinfo[0]) > 3:
            manylc.append(res['fun']/(3.28084**2))  # many lc
        curmeas = meas[veh][t_n-t_nstar:T_nm1+1-t_nstar,2]
        cursim = np.array(cal.all_vehicles[0].posmem)
        if count >= mergeind:
            # curmse = np.square(cursim[:times] - curmeas[:times])
            # curmse = np.sum(curmse)/len(curmse)/(3.28084**2)
            curmse = res_list[count]['fun']/(3.28084**2)
            merge.append(curmse)
        for count, curtime in enumerate(lctimes):
            curmse = np.square(cursim[curtime-t_n:curtime-t_n+times] - curmeas[curtime-t_n:curtime-t_n+times])
            curmse = np.sum(curmse)/len(curmse)/(3.28084**2)
            nearlc.append(curmse)  # near lc
            # try:
            curg = rinfo[0][count][1]
            if curg > 0:
                shortlc.append(curmse)
            # except:
            #     print('hello')

        simacc = [(cursim[i+2] - 2*cursim[i+1] + cursim[i])/(dt**2) for i in range(len(cursim)-2)]
        measacc = [(curmeas[i+2] - 2*curmeas[i+1] + curmeas[i])/(dt**2) for i in range(len(curmeas)-2)]
        upper = max(realistic*max(measacc), 4*3.28)
        lower = min(-6*3.28, realistic*min(measacc))
        temp1, temp2 = max(simacc), min(simacc)
        if temp1 < upper and temp2 > lower:
            real.append(1)
        else:
            real.append(0)
            out2.append(((upper, lower), (temp1, temp2)))
    out['mse near LC'] = (np.mean(nearlc), np.median(nearlc), np.std(nearlc))
    out['mse for merges'] = (np.mean(merge), np.median(merge), np.std(merge))
    out['mse for many LC'] = (np.mean(manylc), np.median(manylc), np.std(manylc))
    out['realistic acc'] = (np.mean(real))
    out['short lc'] = (np.mean(shortlc), np.median(shortlc), np.std(shortlc))
    print(out)
    return out, out2

lclist = lc_list.copy()
lclist.extend(merge_list)
mergeind = len(lc_list)
#%%
with open('IDMrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {}
idm1 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('IDMnorelax.pkl', 'rb') as f:
    res1, res2, res3 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':NoRelaxIDM}

idm2 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
idm3 = analyze_res(res3, nolc_list, meas, platooninfo, .1, kwargs)

#%%
with open('OVMrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.OVMCalibrationVehicle}
ovm1 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('OVMnorelax.pkl', 'rb') as f:
    res1, res2, res3 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':NoRelaxOVM}

ovm2 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
ovm3 = analyze_res(res3, nolc_list, meas, platooninfo, .1, kwargs)

#%%
with open('Newellrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.NewellCalibrationVehicle}
n1 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('Newellnorelax.pkl', 'rb') as f:
    res1, res2, res3 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':NoRelaxNewell}

n2 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
n3 = analyze_res(res3, nolc_list, meas, platooninfo, .1, kwargs)

#%%
with open('2pIDM.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.Relax2IDM}
idm4 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('2psrelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.RelaxShapeIDM}
idm5 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('ExpIDM.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class':hm.RelaxExpIDM}
idm6 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('NewellLL.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class': hm.NewellLL, 'event_maker':hm.make_ll_lc_event, 'lc_event_fun':hm.ll_lc_event}
n4 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)

with open('SKArelax.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class': hm.SKA_IDM}
idm7 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
#%%
with open('2pvhdIDM.pkl', 'rb') as f:
    res1, res2 = pickle.load(f)
res1.extend(res2)
kwargs = {'vehicle_class': hm.Relax2vhdIDM}
idm8 = analyze_res(res1, lclist, meas, platooninfo, .1, kwargs, mergeind = mergeind)
