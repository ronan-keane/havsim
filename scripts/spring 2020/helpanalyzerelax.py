
"""
script for trying to understand effects of relaxation on specific timesteps in simulation
"""
#%% Estimate for empircal TTE
import numpy as np
import copy
def get_veh(vehid):
    for veh in all_vehicles:
        if veh.vehid == vehid:
            break
    return veh
vehid = 813
headway = []
veh = get_veh(vehid)

start = veh.starttime
leadmem = veh.leadmem.copy()
for count, i in enumerate(leadmem.copy()):
    if count < len(leadmem)-1:
        leadmem[count] = (*i, leadmem[count+1][1])
    else:
        leadmem[count] = (*i, veh.endtime)
        
for entry in leadmem:
    lead, curstart, curend = entry
    if lead is None:
        curhd = np.zeros((curend-curstart,1))
    else:
        leadstart = lead.starttime
        curhd = np.array(lead.posmem[curstart-leadstart:curend-leadstart]) - lead.len - veh.posmem[curstart-start:curend-start]
        curhd = np.expand_dims(curhd, axis=1)
        
    headway.append(curhd)
headway = np.concatenate(headway, axis=0)

plt.figure()
plt.subplot(1,2,1)
plt.plot(headway)
plt.subplot(1,2,2)
plt.plot(veh.speedmem)
plt.title('headway, speed time series for vehicle '+str(veh.vehid))
plt.figure()
plt.plot(headway, veh.speedmem[:-1])




#%% was for simulation code
for veh in all_vehicles:
    if veh.vehid == 125:
        break
vehtn = veh.starttime

# changetime = veh.lanemem[1][1]
relaxind = 0
relax = veh.relaxmem[relaxind][0]
# relax = veh.relax
relaxstart = veh.relaxmem[relaxind][-1]
# relaxstart = veh.relax_start
lead = veh.leadmem[2][0]
leadtn = lead.starttime

# timeinds = list(range(11938, 11948))
timeinds = [648, 649, 650]
for timeind in timeinds:
    hd = lead.posmem[timeind - leadtn] - veh.posmem[timeind - vehtn] - lead.len
    leadspeed = lead.speedmem[timeind - leadtn]
    vehspeed = veh.speedmem[timeind - vehtn]

    print('baseline cf model is '+str(veh.cf_model(veh.cf_parameters, [hd, vehspeed, leadspeed])))

    ttc = (hd -2 -.2*vehspeed) / (vehspeed - leadspeed +1e-6)
    if ttc < 1.5 and ttc > 0:
        temp = (ttc/1.5)**2
        currelax, currelax_v = relax[timeind-relaxstart]
        currelax, currelax_v = temp*currelax, temp*currelax_v
        # currelax = relax[timeind-relaxstart]*temp
    else:
        currelax, currelax_v = relax[timeind-relaxstart]
        # currelax = relax[timeind-relaxstart]
    print('original relax formulation is '+str(veh.cf_model(veh.cf_parameters, [hd + currelax, vehspeed, leadspeed + currelax_v])))
    
    acc, normal_relax = havsim.simulation.models.relaxation_model_ttc([1.5, 2, .3, 1], [hd, vehspeed, leadspeed], .25)
    ###
    if normal_relax:
        currelax, currelax_v = relax[timeind-relaxstart]
        # currelax = self.relax[timeind - self.relax_start]

        acc = veh.cf_model(veh.cf_parameters, [hd +currelax, vehspeed, leadspeed+currelax_v])
    print('alternative relax formulation is '+str(acc))
    print('s star is '+str(hd - 2 - .3*vehspeed))
    print('ttc is '+str(max(hd-2-.3*vehspeed, 0)/(vehspeed-leadspeed)))
        
    # print(veh.cf_model(veh.cf_parameters, [hd , vehspeed, leadspeed]) + currelax)
#%% try for calibration code

relax = veh.relax
relax = None
# relaxstart = veh.relaxmem[relaxind][0]
relaxstart = veh.relax_start
lead = veh.leadveh
# lead.set_len(12.805)
lead.set_len(14.3015)
vehtn = veh.inittime

timeinds = [vehtn+1]
for timeind in timeinds:
    lead.update(timeind)
    hd = lead.pos - lead.len - veh.posmem[timeind-vehtn]
    leadspeed = lead.speed
    vehspeed = veh.speedmem[timeind - vehtn]

    print(veh.cf_model(veh.cf_parameters, [hd, vehspeed, leadspeed]))

    if relax is None:
        break
    ttc = hd / (vehspeed - leadspeed)
    if ttc < 1.5 and ttc > 0:
        temp = (ttc/1.5)**2
        currelax, currelax_v = relax[timeind-relaxstart, :]*temp
        # currelax = relax[timeind-relaxstart]*temp
    else:
        currelax, currelax_v = relax[timeind-relaxstart, :]
        # currelax = relax[timeind-relaxstart]
    print(veh.cf_model(veh.cf_parameters, [hd + currelax, vehspeed, leadspeed + currelax_v]))
    # print(veh.cf_model(veh.cf_parameters, [hd , vehspeed, leadspeed]) + currelax)

