
"""
script for trying to understand effects of relaxation on specific timesteps in simulation
"""
#%% was for simulation code
for veh in all_vehicles:
    if veh.vehid == 1202:
        break
vehtn = veh.starttime

# changetime = veh.lanemem[1][1]
# relaxind = 0
# relax = veh.relaxmem[relaxind][-1]
relax = veh.relax
# relaxstart = veh.relaxmem[relaxind][0]
relaxstart = veh.relax_start
lead = veh.leadmem[1][0]
leadtn = lead.starttime

# timeinds = list(range(11938, 11948))
timeinds = [4770]
for timeind in timeinds:
    hd = lead.posmem[timeind - leadtn] - veh.posmem[timeind - vehtn] - lead.len
    leadspeed = lead.speedmem[timeind - leadtn]
    vehspeed = veh.speedmem[timeind - vehtn]

    print(veh.cf_model(veh.cf_parameters, [hd, vehspeed, leadspeed]))

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

