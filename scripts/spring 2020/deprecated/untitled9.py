
"""
@author: rlk268@cornell.edu
"""
from havsim.plotting import plotvhd
import matplotlib.pyplot as plt
badvehlist = []
for i in sim_info.keys(): 
    if sim_info[i][0] > 30:
        badvehlist.append(i)

def helpmeplot(veh):
    plt.figure()
    t_nstar, t_n, T_nm1, = platooninfo[veh][:3]
    plt.plot(sim[veh][:,0])
    plt.plot(out[1][veh][t_n-t_nstar:T_nm1-t_nstar,2])
    plt.plot(meas[veh][t_n-t_nstar:T_nm1-t_nstar,2])
    plt.legend(['NN','benchmark','measurements'])
    
    
    plt.figure()
#    t_nstar, t_n, T_nm1, = platooninfo[veh][:3]
    plt.plot(sim[veh][:,1])
    plt.plot(out[1][veh][t_n-t_nstar:T_nm1-t_nstar,3])
    plt.plot(meas[veh][t_n-t_nstar:T_nm1-t_nstar,3])
    plt.legend(['NN','benchmark','measurements'])
    
    temp = copy.deepcopy(meas)
    temp2 = copy.deepcopy(platooninfo)
    temp2['NN'] = temp2[veh]
    temp2['benchmark'] = temp2[veh]
    temp[veh] = meas[veh]
    temp['benchmark'] = out[1][veh]
    temp['NN'] = meas[veh].copy()
    temp['NN'][t_n-t_nstar:T_nm1-t_nstar+1,[2,3]] = sim[veh][:T_nm1-t_n+1,[0,1]]
    plotvhd(temp,None,temp2, [veh, 'benchmark'] ,plot_color_line = True)
    plotvhd(temp,None,temp2, [veh,'NN'] ,plot_color_line = True)
    #meas = blue, benchamrk = red, NN = green
    
for i in [61,62,63,64,65,66]:
    out = benchmark(meas,platooninfo, [[badvehlist[i]]], usemodel = 'IDM')
    helpmeplot(badvehlist[i])