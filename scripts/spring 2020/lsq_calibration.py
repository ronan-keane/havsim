
"""
Fit a trajectory using least squares.
"""
import numpy as np
import scipy.optimize as sc
# Uses new VehicleData format
veh = 1013
dt = .1

times = vehdict[veh].longest_lead_times
s = np.array(vehdict[veh].leadmem.pos[slice(*times)]) -  np.array(vehdict[veh].leadmem.len[slice(*times)]) - vehdict[veh].posmem[slice(*times)]
v = vehdict[veh].speedmem[slice(*times)]
vl = vehdict[veh].leadmem.speed[slice(*times)]
s = s*3.28
v = np.array(v)*3.28
vl = np.array(vl)*3.28
xhat = [(v[i+1] - v[i])/dt for i in range(len(v)-1)]
statelist = [(s[i], v[i], vl[i]) for i in range(len(xhat))]

def cf_model(p, state):
    s = state[0]
    v = state[1]
    vl = state[2]
    vvl = v*vl
    vs = v*s
    hi = 1+p[1]*v+p[2]*s+p[3]*vs+p[4]*vl+p[5]*vvl
    lo = p[0] + p[6]*v+p[7]*s+p[8]*vs+p[9]*vl+p[10]*vvl
    return hi/lo

pguess = [0.1910479, -0.17384205, 0.20569427, -0.40989588, 0.64067768, 0.48721555,
                      -0.13793132, 0.18327771, -0.05461281, -0.08321112, 0.02858983]
mybounds = [(-1, 15), (-2,-.0001), (.00001, 2), (-2, -.00001), (.00001, 2), (-1, 1),
                        (-5, 5),(-5, 5),(-5, 5),(-5, 5),(-1,1)]
lb = [i[0] for i in mybounds]
ub = [i[1] for i in mybounds]

def fun(p):
    m = len(xhat)
    out = np.zeros((m,))
    for i in range(m):
        curstate = statelist[i]
        cur_target = xhat[i]
        cur_pred = cf_model(p, curstate)
        out[i] = cur_target-cur_pred
    return out

sol = sc.least_squares(fun, pguess, bounds=(lb, ub), )



