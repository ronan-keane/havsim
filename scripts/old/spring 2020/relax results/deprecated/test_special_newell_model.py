
"""
@author: rlk268@cornell.edu
"""

from special_newell_model import make_calibration
import scipy.optimize as sc
import time

# #test for errors
# for veh in lc_list:
#     cal = make_calibration([veh], meas, platooninfo, .1)
#     cal.simulate([.05, 20, 60, 1])


#test calibration
veh = lc_list[103]
pguess = [.05,20,60,1]
mybounds = [(1,100),(1,30),(30,110),(.5,3)]
cal = make_calibration([veh], meas, platooninfo, .1)
start = time.time()
# cal.simulate([.05,20,60,1])
cal.simulate([ 0.1       ,  5.        , 43.68263174,  0.5       ])
print('time to evaluate objective is '+str(time.time()-start))
start = time.time()
out = sc.differential_evolution(cal.simulate, bounds = mybounds)
print(str(time.time()-start)+' to find mse '+str(out['fun']))
# bfgs = sc.fmin_l_bfgs_b(cal.simulate, pguess, bounds = mybounds, approx_grad=1)  # BFGS
# print('time to calibrate is '+str(time.time()-start)+' to find mse '+str(bfgs[1]))

# plt.plot(cal.all_vehicles[0].posmem)
plt.plot(cal.all_vehicles[0].speedmem)
t_nstar, t_n, T_nm1 = platooninfo[veh][:3]
plt.plot(meas[veh][t_n-t_nstar:T_nm1-t_nstar,3])
# plt.figure()
# plt.plot(cal.all_vehicles[0].posmem)
# plt.plot(meas[veh][t_n-t_nstar:T_nm1-t_nstar,2])

plt.figure()
plt.plot(cal.all_vehicles[0].DeltaNmem)
