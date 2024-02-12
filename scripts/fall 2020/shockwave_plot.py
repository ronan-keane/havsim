"""Some different plots, example of shockwave in ngsim I-80 data, FD plots."""
import havsim
import pickle
import matplotlib.pyplot as plt

with open('C:\\Users\\tawit\\Documents\\GitHub\\havsim\\data\\recon-ngsim-old.pkl', 'rb') as f:
    meas, platooninfo = pickle.load(f)

platoon = [1732, 1739, 1746, 1748, 1755, 1765, 1772, 1780, 1793, 1795, 1804, 1810, 1817, 1821,
           1829, 1845, 1851, 1861, 1868, 1873, 1882, 1887, 1898, 1927, 1929 ,1951 ,1941 ,1961, 1992 ,
           1984 ,1998, 2006, 2019 ,2022, 2035, 2042, 2050, 2058, 2065, 2071, 2095, 2092, 2097, 2108, 2113,
           2122, 2128, 2132, 2146, 2151, 2158, 2169, 2176, 2186, 2190, 2199, 2234, 2253,]

ani = havsim.plotting.animatetraj(meas, platooninfo, show_id=False,
                                  spacelim=(0, 1600), lanelim=(8, 4.5), usetime=list(range(1000, 5000)),
                                  save_name='ngsim_shockwave')

# #%%
# ani2 = havsim.plotting.animatevhd(meas,None,platooninfo, [998, 1013])
# plt.show()

# #%%  FD plots
# import havsim.simulation as hs
# from havsim.simulation.simulation_models import OVMVehicle
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from havsim.simulation.models import IDM_parameters
# #IDM spd-hd, flow-density
# cf_p, unused = IDM_parameters()
# tempveh = hs.Vehicle(-1, None, cf_p, None, maxspeed = cf_p[0]-1e-6)
# spds = np.arange(0,cf_p[0],.01)
# hds = np.array([tempveh.get_eql(i, input_type='v') for i in spds])
# fig, ax = plt.subplots()
# plt.plot(hds,spds, 'C0')
# plt.ylabel('speed (m/s)')
# plt.xlabel('headway (m)')
#
# flows = np.array([tempveh.get_flow(i) for i in spds])
# density = np.divide(flows,spds)
# fig, ax2 = plt.subplots()
# plt.plot(density*1000,flows*3600)
# plt.ylabel('flow (veh/hr)')
# plt.xlabel('density (veh/km)')
#
# #OVM spd-hd, flow-density
# cf_p2 = [12.5,.086, 1.4, 1.4, .175 ] #[16.8,.06, 1.545, 2, .12 ] #[10, .086, 1.545, 1.4, .175]
# tempveh2 = OVMVehicle(-1, None, cf_p2, None, maxspeed = cf_p2[0]*(1-math.tanh(-cf_p2[2]))-.1, eql_type='s')
# hds = np.arange(cf_p2[4]/cf_p2[1], 1000)
# spds = np.array([tempveh2.get_eql(i, input_type='s') for i in hds])
# ax.plot(hds, spds, 'C1')
# ax.legend(['IDM', 'OVM'])
#
# flows = np.array([tempveh2.get_flow(i, input_type='s') for i in hds])
# density = np.divide(flows, spds)
# ax2.plot(density*1000, flows*3600)
# ax2.legend(['IDM', 'OVM'])
