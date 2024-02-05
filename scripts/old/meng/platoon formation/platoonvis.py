
"""
@author: rlk268@cornell.edu
purpose of this script is to visually inspect platoons - 
requires platoonmetrics script to be run first 
also write some code which gets some results to play with 
"""
import havsim.plotting as hplt
import havsim.calibration.helper as helper

def platoonvis(meas,platooninfo,platoonlist):
    k = .9
    scores = []
    badplatoons = {}
    for i in platoonlist:
            scores.append(helper.chain_metric(i, platooninfo, meas=meas, k=k) / len(i))
    violation = helper.cirdep_metric(platoonlist, platooninfo, meas=meas, metrictype='veh')
    for i in violation: 
        try:
            badplatoons[i[1]].append(i[0])
        except:
            badplatoons[i[1]] = [i[0]]
    badplatoonslist = list(badplatoons.keys())
    badplatoonslist = sorted(badplatoonslist)
    
    return scores, badplatoons, badplatoonslist
scores, badplatoon,badplatoonlist = platoonvis(meas,platooninfo,platoons)
#%%
out = hplt.animatetraj(meas,platooninfo,platoon=cycle)

#%%
cmetriclist = [] #True if useless, False otherwise, counts number of useless vehicles
useless =[] #for every useless vehicle, tuple of (vehicle, platoon, platoonindex)
for platcount, i in enumerate(platoons): 
    for count, j in enumerate(i):
        T = set(range(platooninfo[j][1],platooninfo[j][2]+1))
        cur = helper.c_metric(j, i, T, platooninfo, meas=meas)
#        try:
#            cur2 = helper.c_metric(j,i,T,platooninfo,type ='follower')
#        except: 
#            print(j)
#            print(i)
        cur2 = helper.c_metric(j, i, T, platooninfo, meas=meas, metrictype='follower')
        if cur ==0 and cur2 == 0:
            cmetriclist.append(True)
            useless.append((j,i,platcount))
        else:
            cmetriclist.append(False)
        