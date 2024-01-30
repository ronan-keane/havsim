"""Example of loading saved data and using plotting api."""
import pickle
import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt

with open('pickle files/e94_16_18.pkl', 'rb') as f:
    all_vehicles, lanes = pickle.load(f)

all_vehicles = hs.vehicles.reload(all_vehicles[0], lanes)
sim, siminfo = hp.plot_format(all_vehicles, lanes)
# sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (800, 1350))
sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (7300, 9300))

# ani = hp.animatetraj(sim, siminfo, usetime=list(range(1000, 3000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))
# ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(5000, 9000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))

# hp.platoonplot(sim2, None, siminfo2, lane=1, opacity=0, timerange=[288000, 300000], colorcode=True)
# hp.platoonplot(sim, None, siminfo, lane=0, opacity=0, timerange=[1000, 3000], colorcode=False)

hp.plotflows(sim, [[9230, 9330], [11000, 11100]], [18000*16, 18000*18], 300, h=.2)

ani2 = hp.animatetraj(sim2, siminfo2, usetime=list(range(int(18000*16.5), int(18000*16.9))), show_id=False,
                      spacelim=(8700, 9300), lanelim=(3.5, -1), interval=10)

hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))
# hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=0, speed_bounds=(0, 35))

plt.show()