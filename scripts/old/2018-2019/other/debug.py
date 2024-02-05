# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:31:23 2019

@author: rlk268
"""
#import jax.numpy as jnp 
#import pickle
#with open('dataautodiff.pkl','rb') as f:
#    meas,platooninfo = pickle.load(f)
#meas2 = {}
#for i in meas.keys():
#    meas2[i] = jnp.asarray(meas[i])
#    
#with open('datauatodiff2.pkl','wb') as f: 
#    pickle.dump([meas2,platooninfo],f)
    

#from calibration import interp1ds, binaryint
#test = [1,2,3,4,5]
#test2 = [2,3,4,5,6]
#test3 = [3.25,4,4.75]
#out = interp1ds(test,test2,test3)

#eql(IDM_b3,30+speedoffset,p, length)


#egobj([10,1.6,2,1.3,1.5],*args)
#data = []
#for i in universe: 
#    data.append(i.dx)
#with open('simeg33.33-1.5-2-1.1-1.5IDM_b3.pkl','wb') as f:
#    pickle.dump([data],f)

#import havsim.calibration.helper as helper
#platoon4 = [995, 998, 1013, 1023]
#testplatoon2 = [platoon4, [956]]
#cir = helper.cirdep_metric(testplatoon2, platooninfo, k=0.9, type='num', meas = meas)

#%%
#cleaning some stuff up for qiwu
#import numpy as np 
#import pickle 
#
#testdata = np.random.rand(50) #note jax handles random numbers in a special way but here we are just treating this as some arbitrary data so it should be fine
#testdata2 = {}
#for i in range(6):
#    testdata2[i] = np.random.rand(50)
#
#testdata3 = [[1,5,10],[2,10,15],[5,15,25]]
#testdata4 = [25,5]
#testdata5 = 20
#
#x1 = [1]  #eg1
#x2 = np.arange(1,10, dtype = np.float) #eg2
#p = 2*np.random.rand(20)+.5
#pfinal = [10,2,3]
#p7 = [2,1,15]
#
#with open('autodiffeg.pkl','wb') as f:
#    pickle.dump([x1,x2,p,pfinal, testdata,testdata2,testdata3,testdata4,testdata5,X,Y,times,p7], f)

#%%
#for i in list(meas.keys()): 
#    cur = helper.makeleadinfo([i],platooninfo,meas)
#    temp = [j[0] for j in cur[0]]
#    if len(np.unique(temp)) < len(temp):
#        print(i)

#meas, platooninfo,platoons = makeplatoonlist(data,n=5)
#helper.c_metric(384,platoons[0],T,platooninfo,type='follower')

#%%
from havsim.calibration.algs import makedepgraph, oldmakedepgraph
totfollist = [672.0, 503.0, 582.0, 423.0, 455.0, 518.0, 435.0, 533.0, 470.0, 439.0, 413.0, 511.0]
leaders = [492.0, 521.0, 460.0, 380.0, 420.0, 409.0, 400.0, 411.0, 406.0, 394.0, 312.0, 657.0, 613.0, 384.0]

Glen = math.inf
Gedge= math.inf
for i in totfollist:
    curG, depth = makedepgraph([i],leaders,platooninfo,math.inf)
    if len(curG.nodes()) < Glen or len(curG.edges())<Gedge:
        G = curG
        print('best new G is '+str(i)+' source with '+str(len(G.nodes()))+' nodes')
        
oldG = oldmakedepgraph(totfollist,leaders,platooninfo)


#%%
totfollist = [1696.0, 1697.0, 1676.0, 1709.0, 1708.0, 1713.0, 1683.0, 1685.0, 1690.0, 1692.0, 1694.0, 1695.0]
leaders = [1670.0, 1673.0, 1684.0, 1679.0, 1678.0, 1681.0, 1687.0, 1488.0, 1686.0, 1171.0, 1175.0, 1075.0, 819.0, 692.0, 697.0, 707.0, 717.0, 658.0, 724.0, 729.0, 711.0, 382.0, 277.0, 282.0, 264.0, 268.0, 269.0, 332.0, 280.0, 308.0, 270.0, 237.0, 247.0, 242.0, 251.0, 312.0, 46.0, 384.0]
cycle = [1696.0, 1710.0, 1736.0, 1726.0, 1729.0, 1683.0, 1698.0, 1709.0, 1713.0, 1689.0, 1677.0, 1720.0, 1702.0, 1717.0, 1676.0, 1694.0, 1685.0, 1695.0, 1757.0, 1743.0, 1745.0, 1758.0, 1759.0, 1750.0, 1751.0, 1742.0, 1744.0, 1737.0, 1756.0, 1721.0, 1734.0, 1753.0, 1797.0, 1748.0, 1778.0, 1787.0, 1707.0, 1725.0, 1733.0, 1746.0, 1735.0, 1714.0, 1727.0, 1711.0, 1774.0, 1739.0, 1700.0, 1708.0, 1724.0, 1692.0, 1732.0, 1762.0, 1690.0, 1716.0, 1718.0, 1722.0, 1706.0, 1715.0, 1703.0]

