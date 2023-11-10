"""Example of loading saved data and using plotting api."""
import pickle
import havsim.simulation as hs
import havsim.plotting as hp
import matplotlib.pyplot as plt
from matplotlib import animation

with open('pickle files/bottleneck_sim_0.pkl', 'rb') as f:
    all_vehicles, laneinds = pickle.load(f)
# with open('C:\\Users\\tawit\\Documents\\GitHub\\havsim\\scripts\\2023\\bottleneck_sim_0.pkl', 'rb') as f:
#     all_vehicles, laneinds = pickle.load(f)

all_vehicles = hs.vehicles.reload(all_vehicles)
sim, siminfo = hp.plot_format(all_vehicles, laneinds)
sim2, siminfo2 = hp.clip_distance(all_vehicles, sim, (800, 1350))

# ani = hp.animatetraj(sim, siminfo, usetime=list(range(1000, 3000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))
# ani2 = hp.animatetraj(sim, siminfo, usetime=list(range(5000, 9000)), show_id=False, spacelim=(0, 2000), lanelim=(3, -1))

# hp.platoonplot(sim, None, siminfo, lane=1, opacity=0, timerange=[1000, 3000], colorcode=False)
# hp.platoonplot(sim, None, siminfo, lane=0, opacity=0, timerange=[1000, 3000], colorcode=False)


ani2 = hp.animatetraj(sim2, siminfo2, usetime=list(range(100, 1500)), show_id=False, spacelim=(800, 1350), lanelim=(3.5, -1),
                      show_lengths=4, interval=100)
writergif = animation.PillowWriter(fps=30)
ani2.save('no_lcd_2.gif', writer=writergif)

# hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=1, speed_bounds=(0, 35))
# hp.plotspacetime(sim, siminfo, timeint=40, xint=30, lane=0, speed_bounds=(0, 35))

plt.show()