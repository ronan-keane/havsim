import pickle
import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt

with open('pickle files/e94_sim_0.pkl', 'rb') as f:
    all_vehicles, laneinds = pickle.load(f)
# with open('C:\\Users\\tawit\\Documents\\GitHub\\havsim\\scripts\\2023\\bottleneck_sim_0.pkl', 'rb') as f:
#     all_vehicles, laneinds = pickle.load(f)

all_vehicles = hs.vehicles.reload(all_vehicles)
sim, siminfo = hp.plot_format(all_vehicles, laneinds)
sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (8000, 10000))

# ani = hp.animatetraj(sim, siminfo, usetime=list(range(1000, 3000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))
# ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(5000, 9000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))

# hp.platoonplot(sim, None, siminfo, lane=1, opacity=0, timerange=[1000, 3000], colorcode=False)
# hp.platoonplot(sim, None, siminfo, lane=0, opacity=0, timerange=[1000, 3000], colorcode=False)


ani2 = hp.animatetraj(sim2, siminfo2, usetime=list(range(8100, 8500)), show_id=False, spacelim=(8000, 10000), lanelim=(3, -1))

# hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))
# hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=0, speed_bounds=(0, 35))

plt.show()