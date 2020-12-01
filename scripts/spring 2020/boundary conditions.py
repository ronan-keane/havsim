"""
Code reviewing how to construct data driven boundary conditions
"""
# %%get boundary conditions (careful with units)
# #option 1 -
# #could get them directly from data
# entryflows, unused = getentryflows(meas, [3],.1,.25)
# unused, unused, exitspeeds, unused = boundaryspeeds(meas, [], [3],.1,.1)

# #option 2 - use calculateflows, which has some aggregation in it and uses a different method to compute flows
# q,k = calculateflows(meas, [[200,600],[1000,1400]], [0, 9900], 30*10, lane = 6)

#option 3 - can also just make boudnary conditions based on what the FD looks like
cf_p, unused = IDM_parameters()
tempveh = hs.Vehicle(-1, None, cf_p, None, maxspeed = cf_p[0]-1e-6)
spds = np.arange(0,cf_p[0],.01)
flows = np.array([tempveh.get_flow(i) for i in spds])
density = np.divide(flows,spds)
plt.figure()
plt.plot(density*1000,flows*3600)