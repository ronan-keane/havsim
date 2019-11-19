
"""
@author: rlk268@cornell.edu

create metric for evaluating how good platoons are 
"""
import pickle 
import numpy as np 
from havsim.calibration.algs import makeplatoonlist, makeplatoonlist_s

#load data
try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/trajectory data/mydata.pkl', 'rb') as f: #replace with path to pickle file
        rawdata, truedata, data, trueextradata = pickle.load(f) #load data
except:
    with open("/users/qiwuzou/Documents/assignment/M.Eng/mydata.pkl", 'rb') as f:
        rawdata, truedata, data, trueextradata = pickle.load(f) #load data
#%%
    
#existing platoon formation algorithm
meas, platooninfo, platoons = makeplatoonlist(data, n = 5)
#existing platoon formation algorithm in a single lane 
unused, unused, laneplatoons = makeplatoonlist(data,n=5,lane=2,vehs=[582,1146])
#platoon formation based on sorting    
unused, unused, sortedplatoons = makeplatoonlist_s(data,n=5,lane=6)
    
#%%
#note that havsim.calibration.helper.makeleadfolinfo can be used to get the leaders 
#for a platoon, which may be convenient. 

from havsim.calibration.helper import makeleadfolinfo

testplatoon = [381.0, 391.0, 335.0, 326.0, 334.0]
leadinfo, folinfo, unused = makeleadfolinfo(testplatoon, platooninfo, meas)

#leadinfo[2] = [[316.0, 1302, 1616], [318.0, 1617, 1644]]
#this means second vehicle in the platoon (testplatoon[1], which is 391)
#follows vehicle 316 from time 1302 to 1616, and it follows 318 from 1617 to 1644. 
#folinfo has the same information but for followers instead of leaders. 

"""
TO DO 
Implement functions which calculate the metrics for a given platoon
Calculate manually what the chain metric should be for the platoon [[], 391, 335, 326] for k = 1 and k = 0. Show your work. 
"""




def chain_metric(platoon, platooninfo, k = .9, type = 'lead' ):
    res = 0
    for i in platoon:
        T = set(range(platooninfo[i][1], platooninfo[i][2]+1))
        res += c_metric(i, platoon, T, platooninfo, k, type)
    return res


def c_metric(veh, platoon, T, platooninfo, k = .9, type = 'lead'):
    leadinfo, folinfo, unused = makeleadfolinfo(platoon, platooninfo, meas)
    if veh not in platoon:
        return 0
    targetsList = leadinfo[platoon.index(veh)] if type == 'lead' else folinfo[platoon.index(veh)]
    def getL(veh, platoon, T):
        L = set([])
        if veh not in platoon:
            return L
        # targetsList = leadinfo[platoon.index(veh)]
        temp = set([])
        for i in targetsList:
            if i[0] not in platoon:
                continue
            temp.update(range(i[1], i[2]+1))
        L = T.intersection(temp)
        return L

    def getLead(veh, platoon, T):
        if veh not in platoon:
            return []
        # targetsList = leadinfo[platoon.index(veh)]
        leads = []
        for i in targetsList:
            if i[1] in T or i[2] in T:
                leads.append(i[0])
        return leads

    def getTimes(veh, lead, T):
        # targetsList = leadinfo[platoon.index(veh)]
        temp = set([])
        for i in targetsList:
            if i[0] == lead:
                temp = T.intersection(set(range(i[1], i[2]+1)))
        return temp

    res = len(getL(veh, platoon, T))
    leads = getLead(veh, platoon, T)
    for i in leads:
        res += k* c_metric(i, platoon, getTimes(veh, i, T), platooninfo, k=k)
    return res

def cirdep_metric(platoonlist, platooninfo, k = .9, type = 'veh'):
    if type == 'veh':
        cirList = []
        after = set([])
        for i in range(len(platoonlist)):
            after.update(platoonlist[i])
        for i in range(len(platoonlist)):
            after -= set(platoonlist[i])
            leadinfo, folinfo, unused = makeleadfolinfo(platoonlist[i], platooninfo, meas)
            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                if len(leaders.intersection(after))>0:
                    cirList.append((platoonlist[i][j], i))
        return cirList
    elif type == 'num':
        res = 0
        cirList = []
        after = set([])
        for i in range(len(platoonlist)):
            after.update(platoonlist[i])
        for i in range(len(platoonlist)):
            after -= set(platoonlist[i])
            leadinfo, folinfo, unused = makeleadfolinfo(platoonlist[i], platooninfo, meas)
            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                if len(leaders.intersection(after)) > 0:
                    cirList.append((platoonlist[i][j], i))
        for i in cirList:
            T = set(range(platooninfo[i[0]][1], platooninfo[i[0]][2]))
            res += c_metric(i[0], platoonlist[i[1]], T, platooninfo, k=k, type='follower')
        return res



platoon = [391, 335, 326]
# veh = 391
# T = set(range(platooninfo[391][1], platooninfo[391][2]))
# c = c_metric(veh, platoon, T, platooninfo)




Chain = chain_metric(platoon, platooninfo, 0)
print(Chain)

# 507 + 343 + 397 + k* 34 (326 has a leader 335 at (1583, 1616)) = 1281 if k == 1 else 1247


cir = cirdep_metric([[391, 335, 326], [307, 318, 316]], platooninfo, k=1, type='veh')
print(cir)

# leaders:
# 391: 381
# 335: 316, 318
# 326: 307, 318, 335, 316
# Thus, 335 and 326 violates circular dependency