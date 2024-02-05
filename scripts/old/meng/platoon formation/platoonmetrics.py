
"""
@author: rlk268@cornell.edu

create metric for evaluating how good platoons are 
"""
import pickle 
import numpy as np 
from havsim.calibration.algs import makeplatoonlist, makeplatoonlist_s
import havsim.calibration.helper as helper

#load data
try:
    with open('C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/trajectory data/mydata.pkl', 'rb') as f: #replace with path to pickle file
        rawdata, truedata, data, trueextradata = pickle.load(f) #load data
except:
    try:
        with open("/users/qiwuzou/Documents/assignment/M.Eng/mydata.pkl", 'rb') as f:
            rawdata, truedata, data, trueextradata = pickle.load(f) #load data
    except:
        with open("D:/assignment/Meng/mydata.pkl", 'rb') as f:
            rawdata, truedata, data, trueextradata = pickle.load(f) #load data
#%%
    
#existing platoon formation algorithm
meas, platooninfo, platoons = makeplatoonlist(data, n = 5,cycle_num=1e4, X = 10, cirdep = False)
#existing platoon formation algorithm in a single lane
#unused, unused, laneplatoons = makeplatoonlist(data,n=5,lane=2,vehs=[582,1146])
#platoon formation based on sorting
#unused, unused, sortedplatoons = makeplatoonlist_s(data,n=5,lane=2, vehs = [582, 1146])
    
#%%
##note that havsim.calibration.helper.makeleadfolinfo can be used to get the leaders 
##for a platoon, which may be convenient. 
#
#from havsim.calibration.helper import makeleadfolinfo, makeleadinfo, makefolinfo
#
#testplatoon = [381.0, 391.0, 335.0, 326.0, 334.0]
#leadinfo, folinfo, unused = makeleadfolinfo(testplatoon, platooninfo, meas)
#
##leadinfo[2] = [[316.0, 1302, 1616], [318.0, 1617, 1644]]
##this means second vehicle in the platoon (testplatoon[1], which is 391)
##follows vehicle 316 from time 1302 to 1616, and it follows 318 from 1617 to 1644. 
##folinfo has the same information but for followers instead of leaders. 
#
#"""
#TO DO 
#Implement functions which calculate the metrics for a given platoon
#Calculate manually what the chain metric should be for the platoon [[], 391, 335, 326] for k = 1 and k = 0. Show your work. 
#"""
#
#
#
#
#def chain_metric(platoon, platooninfo, k = .9, type = 'lead' ):
#    res = 0
#    for i in platoon:
#        T = set(range(platooninfo[i][1], platooninfo[i][2]+1))
#        res += c_metric(i, platoon, T, platooninfo, k, type)
#    return res
#
#
#def c_metric(veh, platoon, T, platooninfo, k = .9, type = 'lead', depth=0):
#    # leadinfo, folinfo= makeleadinfo(platoon, platooninfo, meas),  makefolinfo(platoon, platooninfo, meas)
#    # if veh not in platoon:
#    #     return 0
#    # targetsList = leadinfo[platoon.index(veh)] if type == 'lead' else folinfo[platoon.index(veh)]
#
#    if type == 'lead':
#        leadinfo = makeleadinfo([veh], platooninfo, meas)
#        targetsList = leadinfo[0]
#    else:
#        folinfo = makefolinfo([veh], platooninfo, meas)
#        targetsList = folinfo[0]
#
#    def getL(veh, platoon, T):
#        L = set([])
#        # if veh not in platoon:
#        #     return L
#        # targetsList = leadinfo[platoon.index(veh)]
#        temp = set([])
#        for i in targetsList:
#            if i[0] not in platoon:
#                continue
#            temp.update(range(i[1], i[2]+1))
#        L = T.intersection(temp)
#        return L
#
#    def getLead(veh, platoon, T):
#        # if veh not in platoon:
#        #     return []
#        # targetsList = leadinfo[platoon.index(veh)]
#        leads = []
#        for i in targetsList:
#            if i[0] in platoon and (i[1] in T or i[2] in T):
#                leads.append(i[0])
#        return leads
#
#    def getTimes(veh, lead, T):
#        # targetsList = leadinfo[platoon.index(veh)]
#        temp = set([])
#        for i in targetsList:
#            if i[0] == lead:
#                temp = T.intersection(set(range(i[1], i[2]+1)))
#        return temp
#
#    res = len(getL(veh, platoon, T))
#    leads = getLead(veh, platoon, T)
#    for i in leads:
#        res += k*c_metric(i, platoon, getTimes(veh, i, T), platooninfo, k=k, type=type, depth=depth+1)
#    return res
#
#def cirdep_metric(platoonlist, platooninfo, k = .9, type = 'veh'):
#    if type == 'veh':
#        cirList = []
#        after = set([])
#        for i in range(len(platoonlist)):
#            after.update(platoonlist[i])
#        for i in range(len(platoonlist)):
#            after -= set(platoonlist[i])
#            leadinfo, folinfo = makeleadinfo(platoonlist[i], platooninfo, meas),  makefolinfo(platoon, platooninfo, meas)
#            for j in range(len(platoonlist[i])):
#                leaders = [k[0] for k in leadinfo[j]]
#                leaders = set(leaders)
#                if len(leaders.intersection(after))>0:
#                    cirList.append((platoonlist[i][j], i))
#        return cirList
#    elif type == 'num':
#        res = 0
#        cirList = []
#        after = set([])
#        for i in range(len(platoonlist)):
#            after.update(platoonlist[i])
#        for i in range(len(platoonlist)):
#            after -= set(platoonlist[i])
#            leadinfo, folinfo = makeleadinfo(platoonlist[i], platooninfo, meas),  makefolinfo(platoon, platooninfo, meas)
#            for j in range(len(platoonlist[i])):
#                leaders = [k[0] for k in leadinfo[j]]
#                leaders = set(leaders)
#                leaders_after = leaders.intersection(after)
#                if len(leaders_after) > 0:
#                    cirList.append((list(leaders_after), i))
#        res = []
#        for i in cirList:
#            for j in i[0]:
#                T = set(range(platooninfo[j][1], platooninfo[j][2]))
#                res.append(c_metric(j, platoonlist[i[1]], T, platooninfo, k=k, type='follower'))
#        return res
#
#
#
#platoon = [391, 335, 326]
##chainmetric = 34 - 326 follows 335 
#testplatoon = [381.0, 391.0, 335.0, 326.0, 334.0]
##chainmetric = 507 - 391 follows 381,
##34 - 326 follows 335
## 59 - 334 follows 326, 194 + 56 - 334 follows 335 
##chainmetric = 507 + 34 + 59 + 194 + 56 = 850
#
#
#platoon1 = [307, 318, 335]
#platoon2 = [316, 262, 307]
#platoon3 = [259, 247, 315, 237]
#platoon4 = [995, 998, 1013, 1023]
#"""
#platoon1:
#'307': 0 because 307 follows 308 and 308 is not in platoon
#'318': 225 because 318 follows 307 and 307 is leader in [1395, 1619]
#'335': 28 + k*3 because 335 follows 318 and 318 is leader in [1617, 1644], and 318 follows 307 in [1617, 1619]
#metric = 225 + 28 + 3*k 
#
#platoon2:
#'316': 40 because 316 follows 262 in [981, 1020]
#Others contribute 0 because leaders are not in platoon
#metric = 40
#
#platoon3:
#'259': 418 + 400*k because 259 follows 247 in [1028, 1445] and 247 follows 237 in [1028, 1427] (during T = times(249, 247, T))
#'247': 428 because 247 follows 237 in [1000, 1427]
#'315': 0 because leaders are not in platoon
#'237': 0 because leaders are not in platoon
#metric = 846 + 400*k
#
#platoon4:
#998: 520 when it follows 995 in [2770, 3289]
#1013: 528 when it follows 998 in [2785, 3312], k*505 when 998 follows 995 in [2785, 3289]
#1023: 509 when it follows 1013 in [2826, 3334], k*487   when 1013 follows 998 in [2826, 3312], k**2 *464 when 998 follows 995 in [2826, 3289]
#metric = 1557 + k*992 + k**2 * 464
#platoon5:
#409 - 
#163 when 409 follows 393 in  [1246, 1408]
#420 - 
#103 when 420 follow 409 in [1306,1408] (409 follows 393 in [1306, 1408])
#190 when 420 follow 409 in [1705,1894]
#40 when 420 folow 393 in [1409,1448]
#34 when 420 folow 393 in [1470,1503]
#33 when 420 folow 393 in [1539, 1571]
#metric = 163   +103+190+40+34+33 + k*103 
#"""

#%% metrics test
#
# Chain = chain_metric(platoon4, platooninfo, 1)
# print(Chain)
#
def metricstest():
    print()
    k = .9
    platoon4 = [995, 998, 1013, 1023]
    testplatoon2 = [platoon4, [956]]
    
    cir = helper.cirdep_metric(testplatoon2, platooninfo, k=k, metrictype='num',meas = meas)
    print('circular metric testplatoon2 result is '+str(cir[0]))
    print('expected result is '+str(37+k*13))
    
    testplatoon3 = [[925, 937, 956], [920]]
    cir = helper.cirdep_metric(testplatoon3, platooninfo, k=k, metrictype='num', meas = meas)
    print('circular metric testplatoon3 result is '+str(cir[0]))
    print('expected result is '+str(509 + k*469+k**2*440))
    #should be 1224.89
    
    testplatoon4 = [[393, 409, 420], [395]]
    cir = helper.cirdep_metric(testplatoon4, platooninfo, k=k, metrictype='num', meas = meas)
    print('circular metric testplatoon4 result is '+str(cir[0]))
    print('expected result is '+str(193 + k*163+k*18 + k**2*103))
    
    testplatoon5 = [[393, 409, 420],[395, 411, 389]]
    cir = helper.cirdep_metric(testplatoon5, platooninfo, k=k, metrictype='num', meas = meas)
    print('circular metric testplatoon5 result is '+str(cir[0]))
    print('expected result is '+str(193 + k*163+k*18 + k**2*103 + 190+27+k*(22+34+33) + 244))
    #
    testplatoon6 = [[393, 409, 420],[395, 411], [389]]
    cir = helper.cirdep_metric(testplatoon6, platooninfo, k=k, metrictype='num', meas = meas)
    print('circular metric testplatoon6 result is '+str(cir[0]))
    print('expected result is '+str(193 + k*163+k*18 + k**2*103 + 190+27+k*(22+34+33) + 244))
    
    platoon = [391, 335, 326]
    platoon0  = [381,391,335,326,334]
    platoon1 = [307, 318, 335]
    platoon2 = [316, 262, 307]
    platoon3 = [259, 247, 315, 237]
    platoon4 = [995, 998, 1013, 1023]
    platoon5 = [393,409,420]
    
    res = helper.chain_metric(platoon,platooninfo,k=k,meas = meas)
    print('chain metric platoon result is '+str(res))
    print('expected result is '+str(34))
    
    res = helper.chain_metric(platoon0,platooninfo,k=k,meas = meas)
    print('chain metric platoon0 result is '+str(res))
    print('expected result is '+str(507 + 34 + 59 + 194 + 56))
    
    res = helper.chain_metric(platoon1,platooninfo,k=k,meas = meas)
    print('chain metric platoon1 result is '+str(res))
    print('expected result is '+str(225+28+3*k))
    
    res = helper.chain_metric(platoon2,platooninfo,k=k,meas = meas)
    print('chain metric platoon2 result is '+str(res))
    print('expected result is '+str(40))
    
    res = helper.chain_metric(platoon3,platooninfo,k=k,meas = meas)
    print('chain metric platoon3 result is '+str(res))
    print('expected result is '+str(846+400*k))
    
    res = helper.chain_metric(platoon4,platooninfo,k=k,meas = meas)
    print('chain metric platoon4 result is '+str(res))
    print('expected result is '+str(1557+k*992+k**2*464))
    
    res = helper.chain_metric(platoon5,platooninfo,k=k,meas = meas)
    print('chain metric platoon5 result is '+str(res))
    print('expected result is '+str(163+103+190+40+34+33+k*103))
    print()
#
#%%

#"""
#testplatoon2 = [[995, 998, 1013, 1023], [956]]
#995 violates circular dependency because 995 follows 956
#
#testplatoon2:
#37 when 956 has follower 995 in [2746, 2782]
#k*13 when 995 has follower 998 in [2770, 2782], 
#No others could contribute since the time sequence exceeds 2782
#
#metric = 37 + k*13
#
#testplatoon3 = [[925, 937, 956], [920]]
#925 violates circular dependency because 925 follows 920
#
#testplatoon3:
#509 when 920 has follower 925 in [2635, 3143]
#k*469 when 925 has follower 937 in [2675, 3143]
#k**2 * 440 when 937 has follower 956 in [2701, 3140]
#
#metric = 509 + k*469+k**2*440
#
#testplatoon4:  [[393, 409, 420],[395]]
#circular dependency because 393 follows 395
#193 when 393 follow 395 in [1234, 1426]
#k*163 when 409 follows 393 in [1246, 1408]
#k*18 when 420 follow 393 in [1409, 1426]
#k**2*103 when 420 follow 409 in [1306, 1408]
#metric = 193 + k*163+k*18 + k**2*103


#testplatoon5: [[393, 409, 420],[395, 411, 389]]
#circular dependency when 393 follows 395, 411, 389
#from dependencies associated with 395 - 
#193 + k*163+k*18 + k**2*103 (from above)
#from 411 - 
#190 when 393 follow 411 in [1427, 1616]
#k*(22+34+33) when 420 follow 393 in [1427, 1448], [1470, 1503], [1539, 1571]
#27 when 420 follow 411 in [1678, 1704]
#from 389 - 
#244 when 393 follow 389
#metric = 193 + k*163+k*18 + k**2*103 + 190+27+k*(22+34+33) + 244

#"""
#%%

import statistics

def benchmark(platoon_list, meas,platooninfo, n = 5, k = .9):
    chain_metric_scores = []
    cirdep_metric_scores = []
    veh_set = set([])
    platoon_set = set([])
    violate_veh_set = set([])
    lenlist = []
    
    #counting total number of vehicles and chain scores
    for i in platoon_list:
#        if len(i) == 0:
#            continue
        chain_metric_scores.append(helper.chain_metric(i, platooninfo, meas=meas, k=k) / len(i))
        veh_set.update(i)
        lenlist.append(len(i))
    violation = helper.cirdep_metric(platoon_list, platooninfo, meas=meas, metrictype='veh')
    cirdep_metric_scores = helper.cirdep_metric(platoon_list, platooninfo, meas=meas, metrictype='num')
    cirdep_metric_scores = np.asarray(cirdep_metric_scores)
    cirdep_metric_scores = cirdep_metric_scores[cirdep_metric_scores>0]
    #counting circular dependencies
    for i in violation:
        platoon_set.add(i[1])
        violate_veh_set.add(i[0][0])
    average = sum(chain_metric_scores)/len(chain_metric_scores)
    median = list(sorted(chain_metric_scores))[len(chain_metric_scores)//2]
    std = statistics.stdev(chain_metric_scores)
    nover = np.asarray(lenlist) 
    nover = nover[nover>n]

    print('number vehicles/number unique vehicles:',np.sum(lenlist),'/',len(veh_set))
    print('normal chain score avg/med/sdev:', round(average,2),'/',round(median,2),'/', round(std,2))
    print('number of circ. dep. vehicles:', len(violate_veh_set), "\nfraction of total vehicles/platoons:", round(len(violate_veh_set)/len(veh_set),5),'/',
          round(len(platoon_set)/len(platoon_list),5))
    print('average/median platoon size:', round(np.mean(lenlist),5),"/",np.median(lenlist), 
          "\nmax platoon size / % platoons over n:", max(lenlist),' / ',round(len(nover)/len(lenlist),5))
    if len(cirdep_metric_scores)>0: #circular dependency scores
        average = round(sum(cirdep_metric_scores)/len(cirdep_metric_scores),2)
        median = round(list(sorted(cirdep_metric_scores))[len(cirdep_metric_scores)//2],2)
        std = round(statistics.stdev(cirdep_metric_scores),2)
        print('cirdep score avg/med/sdev:', average,'/', median,'/', std)
    else:
        print("No circular dependency violation found")
    return chain_metric_scores

benchmark_list = [platoons]
names = ["platoons"]
outlist = []
for i in range(len(benchmark_list)):
    print("Performance for", names[i])
    out = benchmark(benchmark_list[i], meas, platooninfo)
    outlist.append(out)
    print()
#%% do some testing on what the platoons look like and verify ideas for hueristics make sense. 
    
    
# res = makeplatoonlist(rawdata,n=5)
# print(res)
#%%

#"""
#For original
#Performance for platoons
#chain score average: 1540.7533936651585 
#median: 1369 
#standard deviation: 1189.9745801510596
#number of vehicles: 70 
#fraction of total vehicles: 0.03443187407771766 
#fraction of total platoons: 0.11085972850678733
#cirdep score average: 176.52900000000002 
#median: 106.0 
#standard deviation: 170.14791833607816
#
#For 33 (vehicles in platoons: correct)
#Performance for platoons
#chain score average: 1555.9027149321266 
#median: 1370 
#standard deviation: 1200.0188611676674
#number of vehicles: 70 
#fraction of total vehicles: 0.03443187407771766 
#fraction of total platoons: 0.11085972850678733
#cirdep score average: 176.6342253521127 
#median: 106.0 
#standard deviation: 170.9755430493697
#
#For 332    (vehicles in platoons: 1938)
#Performance for platoons
#chain score average: 2404.7698412698414 
#median: 2047 
#standard deviation: 1422.7000192466544
#number of vehicles: 33 
#fraction of total vehicles: 0.030136986301369864 
#fraction of total platoons: 0.021164021164021163
#cirdep score average: 178.83632911392408 
#median: 86.0 
#standard deviation: 179.88970141398693
#
#For 341    (vehicles in platoons: 2033)
#Performance for platoons
#chain score average: 1788.2736572890026 
#median: 1595 
#standard deviation: 1109.154440841137
#number of vehicles: 55 
#fraction of total vehicles: 0.02705361534677816 
#fraction of total platoons: 0.10230179028132992
#cirdep score average: 223.25582 
#median: 154.0 
#standard deviation: 256.14688018575555
#
#For 342     (vehicles in platoons: 1991)
#Performance for platoons
#chain score average: 2755.6761363636365 
#median: 1570 
#standard deviation: 4244.103060647846
#number of vehicles: 42 
#fraction of total vehicles: 0.025915492957746478 
#fraction of total platoons: 0.05113636363636364
#cirdep score average: 479.5660755714286 
#median: 294.0 
#standard deviation: 512.5728355167643
#
# 
#Combined 341 and 33   (fastest)
#Performance for platoons
#chain score average: 1406.7219827586207 
#median: 1267 
#standard deviation: 1132.3763590060512
#number of vehicles: 79 
#fraction of total vehicles: 0.03885882931628136 
#fraction of total platoons: 0.15732758620689655
#cirdep score average: 153.15322580645162 
#median: 106.0 
#standard deviation: 141.772048476937
#
#Combined 332 and 342 (best score)
#Performance for platoons
#chain score average: 2847.3485714285716 
#median: 2232 
#standard deviation: 3920.259142022884
#number of vehicles: 51 
#fraction of total vehicles: 0.055855855855855854 
#fraction of total platoons: 0.02
#cirdep score average: 480.9292548333336 
#median: 359.83799999999997 
#standard deviation: 478.28581593271167
#"""