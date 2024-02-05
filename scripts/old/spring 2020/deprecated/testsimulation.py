
"""
@author: rlk268@cornell.edu
"""
#I wrote this to test the code in simulationold2

from havsim.simulation.simulation import simulate_sn
from havsim.calibration.helper import boundaryspeeds, getentryflows
from havsim.calibration.algs import makeplatoonlist
import pickle

#get boundary conditions to use 
#with open('reconngsim.pkl','rb') as f: 
#    data = pickle.load(f)[0]
#    
#meas, platooninfo = makeplatoonlist(data, 1, False)
#
#entryflows, unused = getentryflows(meas, [3],.1,.25) 
#
#unused, unused, exitspeeds, unused = boundaryspeeds(meas, [], [3],.1,.1)

#%%
#simple  road with two lanes 
roadinfo = {0:[2, [(None,None), (None, None)], 800, [], [], [0,0], ['0','1'], [None, None], [None, None]]}
init = [0 for i in range(21)]
auxinfo = {'0':init.copy(), '1':init.copy()}
auxinfo['0'][1] = None
auxinfo['0'][11] = ['', None, '1']
auxinfo['0'][20] = [set(), None, set()]
auxinfo['0'][16] = [[None, 0]]
auxinfo['1'][1] = None
auxinfo['1'][11] = ['0', None, '']
auxinfo['1'][20] = [set(), None, set()]
auxinfo['1'][16] = [[None, 0]]
curstate = {}
modelinfo = {}
roadinfo[0][3] = [entryflows[0][:1000], entryflows[0][1000:2000]]
roadinfo[0][4] = [exitspeeds[0][:1000], exitspeeds[0][1000:2000]]

sim, curstate, auxinfo, roadinfo, endtime = simulate_sn(curstate, auxinfo, roadinfo, modelinfo, 1000, .25, 0)

