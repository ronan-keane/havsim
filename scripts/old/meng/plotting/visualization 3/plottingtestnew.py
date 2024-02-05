
"""
@author: rlk268@cornell.edu
"""
#imports 
import pickle 
import matplotlib.pyplot as plt
from havsim.plotting import platoonplot, calculateflows, plotflows
#%%
#load data

try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/meng/plottingtesting.pkl','rb') as f:
         meas, platooninfo, platoonlist, sim = pickle.load(f)
except:
    with open("/users/qiwuzou/Documents/assignment/M.Eng/follow_up/hav-sim-master/visualization/plottingtesting.pkl", 'rb') as f:
        meas, platooninfo, platoonlist, sim = pickle.load(f)

#with open('D:/assignment/Meng/plottingtesting.pkl','rb') as f:
#    meas, platooninfo, platoonlist, sim = pickle.load(f)

# with open("/users/qiwuzou/Documents/assignment/M.Eng/follow_up/hav-sim-master/visualization/plottingtesting.pkl", 'rb') as f:
#     meas, platooninfo, platoonlist, sim = pickle.load(f)
     
    
#test on actual data
plt.close('all')
plt.figure()
plotflows(meas,[[600,1000]], [1000, 9000], 300, 'line', lane = 2)
plt.figure()
plotflows(meas,[[600,1000]], [1000, 9000], 300, 'FD', lane = 2)
    
    
#%%
#test on toy example 
import numpy as np
testmeas = {}
for i in range(1):
    testmeas[i] = np.zeros((1001,3))
    testmeas[i][:,1] = np.linspace(0,1000,1001)
    testmeas[i][:,2] = np.linspace(0,1000,1001)
 #3 vehicles, all of them have s = 0-1000. the times are 0-1000 for vehicle 0
#plt.figure()
#plotflows(testmeas,[[200,400],[800,1000]],[0,1000],300,'line')
q,k = calculateflows(testmeas,[[200,400],[800,1000]],[0,1000],300)
print(q,k)
# #q = [.00166, .0166, 0, 0] for first region, [0, 0, .166, .005] for second.
# #k is the same as q in this example.
#
#
testmeas2 = {}
for i in range(3):
    testmeas2[i] = np.zeros((1001,3))
    testmeas2[i][:,1] = np.linspace(0+100*i,1000+100*i,1001) #equivalent to list(range(1001+100*i))[100*i:]
    testmeas2[i][:,2] = np.linspace(0,1000,1001)

q,k = calculateflows(testmeas2,[[200,400],[800,1000]],[0,1000],300)
print(q,k)
#q = [.00166, .08333, 0, 0] for first region, [0, 0, .00166, .01]
#k is the same as q in this example


#%%
#another toy example I want you to test on to verify on
meas[898][100:200,7] = 3
meas[905][150:250,7] = 3
platoonplot(meas,None,platooninfo,platoon=[898, 905, 909, 916, 920], lane=None,opacity =.1, colorCode= True, speed_limit = [10,35]) 
plt.plot([2600, 2600, 2800, 2800, 2600], [400, 800, 800, 400, 400], 'k-')
plt.plot([2800, 2800, 3000, 3000, 2800], [400, 800, 800, 400, 400], 'k-')
testmeas3 = {}
for i in [898, 905, 909, 916, 920]:
    testmeas3[i] = meas[i].copy()

q, k = calculateflows(testmeas3, [[400, 800]], [2600, 3000], 200, lane = 2)
print(q,k)
#measurements taken by hand (if lane = 2): 
#q - [.0128125, .0082125] #q[1] has incorrect value in both cases for lane == none and lane is 2
#k - [.0068875, .0051]
 
#if lane = None: 
#q - [.0166375, .0083625]
#k  - [00925, .0052625]
"""
TO DO 
q and k are incorrect for this example, both when lane is None and when lane is 2. I worked out what the correct values are and what all the components should be
"""
#region 1
#vehicle 1 - (2605, 400 - 2632, 473), (2733, 702 - 2800, 733)
#vehicle 2 - (2629, 400, - 2712, 624)
#vehicle 3 - (2658, 400 - 2800, 660)
#vehicle 4 - (2673, 400 - 2800, 630)
#vehicle 5 - (2695, 400 - 2800, 607)
#xi - 73, 31, 224, 260, 230, 207 (if lane = none): add 229, 77 (5 numbers should be 333, 301, 260, 230, 207, where lane is None)
#ti - 27, 67, 83, 142, 127, 105 (if lane = None): add 101, 88 (5 numbers should be 195, 171, 142, 127, 105, where lane is None)


#region 2 - 
#vehicle 1 - (2800, 733 - 2845,800)
#vehicle 2 - (2813, 713 - 2862, 800)
#vehicle 3 - (2800, 660 - 2884, 800)
#vehicle 4 - (2800, 630 - 2902, 800)
#vehicle 5 - (2800, 607 - 2928, 800)
#xi - 67, 87, 140, 170, 193 (if lane = none): add 12 (5 numbers should be 67, 100, 140, 170, 193, where lane is None)
#ti - 45, 49, 84, 102, 128 (if lane = none): add 13 (5 numbers should be 45, 62, 84, 102, 128, where lane is None)



