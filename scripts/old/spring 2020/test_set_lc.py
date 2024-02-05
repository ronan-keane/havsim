
"""
@author: rlk268@cornell.edu
"""
#tests for set_lc method - turn off cooperative model
from havsim.simulation.simulation import lane, vehicle
import pickle
import numpy as np
import matplotlib.pyplot as plt

#make dummy road network
road = {'name':'test'}
lane0 = lane(0, 1000, road, 0)
lane1 = lane(0, 1000, road, 1)
lane2 = lane(0,1000, road, 2)

#make vehicles for test
testveh = vehicle(0, lane1, [27, 1.2, 2, 1.1, 1.5], [-2, .2, .2, 0, .1, 1])
lead =  vehicle(0, lane1, [28, 1.2, 2, 1.1, 1.5], [-1.5, .2, .2, 0, .1, 1])
fol = vehicle(0, lane1, [25, 1.2, 2, 1.1, 1.5], [-1.5, .2, .2, 0, .1, 1])
lfol = vehicle(0, lane0, [20, 1.2, 1.1, 1.1, 1.5], [-1.5, .2, .2, 0, .1, 1])
llead = vehicle(0, lane0, [23, 1.2, 2, 1.1, 1.5], [-1.5, .2, .2, 0, .1, 1])
rfol = vehicle(0, lane2, [33, .9, 1.5, 1.1, 1.5], [-1.5, .2, .2, 0, .1, 1])
rlead = vehicle(0, lane2, [27, 1.2, 2, 1.1, 1.5], [-1.5, .2, .2, 0, .1, 1])
#set relationships needed to test set lc
testveh.llane = lane0
testveh.rlane = lane2
testveh.lead = lead
testveh.fol = fol
testveh.lfol = lfol
testveh.rfol = rfol
lead.fol = testveh
fol.lead = testveh
rfol.lead = rlead
lfol.lead = llead

#helper fns
def set_state(veh, pos, spd):
    veh.pos = pos
    veh.speed = spd
def maketest(curdata):
    vehlist = [testveh, llead, lead, rlead, lfol, fol, rfol]
    vehlist2 = [testveh, lfol, fol, rfol]
    for count, veh in enumerate(vehlist): 
        set_state(veh, curdata[count,0], curdata[count,1])
    for veh in vehlist2: 
        veh.hd = veh.lane.get_headway(veh, veh.lead)
        veh.set_cf(0, 1)
    lc_actions = {}
    testveh.set_lc(lc_actions, 0, 1)
    return lc_actions.get(testveh)

#test 1- discretionary both sides 
#other tests are basically just copy paste of this 
#generate test data 
test1 = np.random.rand(2000,7,2)
for i in range(len(test1)):
    curdata = test1[i]
    curdata[:,0] = curdata[:,0]*75
    curdata[:,1] = curdata[:,1]*25
    curdata[1:4,0] += curdata[0,0]
    curdata[4:,0] -= curdata[0,0]
with open('testlc1.pkl', 'wb') as f:
    pickle.dump(test1, f)
with open('testlc1.pkl', 'rb') as f:
    test1 = pickle.load(f)
    
testveh.l = 'discretionary'
testveh.r = 'discretionary'
test1result = [maketest(test1[i]) for i in range(len(test1))]
with open('test1result.pkl', 'wb') as f:
    pickle.dump(test1result, f)
with open('test1result.pkl', 'rb') as f:
    test1result = pickle.load(f)
    
#test 2 - mandatory one side 
test2 = np.random.rand(2000,7,2)
for i in range(len(test2)):
    curdata = test2[i]
    curdata[:,0] = curdata[:,0]*40
    curdata[:,1] = curdata[:,1]*15
    curdata[1:4,0] += curdata[0,0]
    curdata[4:,0] -= curdata[0,0]
with open('testlc2.pkl', 'wb') as f:
    pickle.dump(test2, f)
with open('testlc2.pkl', 'rb') as f:
    test2 = pickle.load(f)
    
testveh.l = 'discretionary'
testveh.r = 'mandatory'
test2result = [maketest(test2[i]) for i in range(len(test2))]
with open('test2result.pkl', 'wb') as f:
    pickle.dump(test2result, f)
with open('test2result.pkl', 'rb') as f:
    test2result = pickle.load(f)


#test 3 - discretionary one side 
test3 = np.random.rand(2000,7,2)
for i in range(len(test3)):
    curdata = test3[i]
    curdata[:,0] = curdata[:,0]*75
    curdata[:,1] = curdata[:,1]*25
    curdata[1:4,0] += curdata[0,0]
    curdata[4:,0] -= curdata[0,0]
with open('testlc3.pkl', 'wb') as f:
    pickle.dump(test3, f)
with open('testlc3.pkl', 'rb') as f:
    test3 = pickle.load(f)
    
testveh.l = 'discretionary'
testveh.r = None
test3result = [maketest(test3[i]) for i in range(len(test3))]
with open('test3result.pkl', 'wb') as f:
    pickle.dump(test3result, f)
with open('test3result.pkl', 'rb') as f:
    test3result = pickle.load(f)

#test 4 - relaxation (turn relax settings to True - True)
testveh.in_relax = True
testveh.relax = [20]
testveh.relax_start = 0
rfol.in_relax = True
rfol.relax = [-30]
rfol.relax_start = 0
test4 = test1
    
testveh.l = 'discretionary'
testveh.r = 'discretionary'
test4result = [maketest(test4[i]) for i in range(len(test4))]
with open('test4result.pkl', 'wb') as f:
    pickle.dump(test4result, f)
with open('test4result.pkl', 'rb') as f:
    test4result = pickle.load(f)

