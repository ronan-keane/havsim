# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:57:32 2019

@author: rlk268
"""

from ..havsim.plotting import * 
#from calibration import checksort


#get vehicles in the lane 6 
vehIDs = np.unique(data[data[:,7]==6,0])
#for each vehicle we look at their average speed, then sort it and we can look at how the average speed is changing over time 
#vehicles which change lanes marked special color, look also at how the speed looked like compared to the immediate follower; if the follower has a faster speed we predict this is bad. 

#also why don't you try doing a simulation where we have the merging rules and look at the how the flow changes according to the in flow amounts 
avgspd = {} #dict where key is id, value is average speed
for i in vehIDs: 
    curmeas = meas[i]
    curmeas = curmeas[curmeas[:,7]==6]
    cur = np.mean(curmeas[:,3]) #average speed
    avgspd[i] = cur
    
sortedvehID = sortveh3(vehIDs,6,meas,platooninfo) #sort all the vehicles 
#sanity check 
#sortedvehID = sortveh4(vehIDs,6,meas,platooninfo)
#checksort(sortedvehID,meas,6)

#spd = []
#for i in range(len(vehIDs)):
#    spd.append(avgspd[sortedvehID[i]])
#plt.plot(spd,'k.')
plt.figure()
for count, i in enumerate(range(len(sortedvehID))):
    x,y = count,avgspd[sortedvehID[i]]
    curmeas = meas[sortedvehID[i]]
    curmeas = curmeas[curmeas[:,7]==6]
    leads = np.unique(curmeas[:,4])
    n = len(leads)
    if 0 in leads: 
        n = n - 1
    if n > 1: 
        plt.plot(x,y,'r.')
    else: 
        plt.plot(x,y,'k.')