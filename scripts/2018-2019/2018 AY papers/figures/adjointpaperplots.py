# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:40:05 2019

@author: rlk268
"""
from matplotlib import cm
#from calibration import *

#%% #make velocity headway plots for different models
##section 0 stuff - everything before case study
#########first we make a figure that shows the complexity of a trajectory in the speed/headway plane, and how different models attempt to describe it
#plt.close('all')
#sim = copy.deepcopy(meas)
#curplatoon = [[],1013] #this is the vehicle we're going to test.
#
#pguess = [10*3.3,.086/3.3, 1.545, 2, .175 ] #original guess
#mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] #less conservative bounds
#
#leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon,platooninfo,sim)
#
#plt.figure(figsize = (12.5,10))
#plt.subplot(2,2,3)
#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False, 5),0,mybounds)
#plotvhd(meas,sim,platooninfo,curplatoon[1],newfig = False)
#rmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon)
#plt.title('calibrated OVM, RMSE = '+str(round(rmse,2))+' (ft)')
#
#plt.subplot(2,2,4)
#pguess =  [40,1,1,3,10] #IDM guess 1
#mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(IDM_b3, IDMadjsys_b3, IDMadj_b3, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False, 5),0,mybounds)
#plotvhd(meas,sim,platooninfo,curplatoon[1], newfig = False)
#rmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon)
#plt.title('calibrated IDM, RMSE = '+str(round(rmse,2))+' (ft)')
#
#plt.subplot(2,2,1)
#pguess = [1,20,70] #daganzo guess 1
#mybounds = [(.1,10),(0,100),(30,120)] #fairly conservative bounds
#bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,pguess,None,(daganzo, daganzoadjsys, daganzoadj, meas,  sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,3),0,mybounds)
#re_diff(sim,platooninfo,curplatoon)
#T_nm1 = platooninfo[curplatoon[1]][2]
#plotvhd(meas,sim,platooninfo,1013,end=T_nm1-1, newfig = False)
#rmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon,dim=1)
#plt.title('calibrated Daganzo model, RMSE = '+str(round(rmse,2))+' (ft)')
#
#plt.subplot(2,2,2)
#pguess =  [1.5,60] #guess 1
#mybounds = [(0,5),(0,200)]
#bfgs = sc.fmin_l_bfgs_b(TTobjfn_obj,pguess,TTobjfn_fder,(None, None, None, meas,sim,platooninfo, curplatoon, leadinfo, folinfo,rinfo,False,2),0,mybounds)
#re_diff(sim,platooninfo,curplatoon)
#plotvhd(meas,sim,platooninfo,1013,delay=bfgs[0][0], newfig = False)
#rmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon,delay=bfgs[0][0])
#plt.title('calibrated Newell model, RMSE = '+str(round(rmse,2))+' (ft)')
#plt.savefig('vhd.png',dpi=200)

#%% plots showing trajectory data and the output of calibration.
#default colors in matplotlib
#prop_cycle = plt.rcParams['axes.prop_cycle'] #for default coloring scheme
#colors = prop_cycle.by_key()['color']
###
#sim = copy.deepcopy(meas)
#results = custom61 #will be using the results from TNC-3
#meas2, followerchain = makefollowerchain(956,data,18)
#plt.figure(figsize = (10,4))
#plt.subplot(1,2,1)
#platoonplot(meas2,followerchain,[],newfig=False, clr=colors[0])
#plt.title('Trajectory data (single lane pictured)')
#
##now show the calibrated trajectories
#vehlist2 = [] #debug purposes
#for i in meas.keys():
#    if len(platooninfo[i][4]) >=1:
#        vehlist2.append([[],i])
#
#pltveh = list(followerchain.keys())
#vehlist = []  #this will be a list of the indices of the results we want to plot
#for i in pltveh: #this works
#    for count, j in enumerate(vehlist2):
#        if j[1]==i:
#            vehlist.append(count)
#
#
#rmse = []
#for count,i in enumerate(vehlist):
#    curplatoon = [[],pltveh[count]] #curplatoon
#    print(curplatoon)
#    leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,meas)
#    obj = platoonobjfn_obj(results[0][i][0],OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo)
#    curr = convert_to_rmse(obj,platooninfo,curplatoon)
#    rmse.append(curr)
#    if count ==0:
#        simdata = sim[curplatoon[1]].copy()
#    else:
#        simdata = np.append(simdata,sim[curplatoon[1]],axis=0)
#    sim[curplatoon[1]] = meas[curplatoon[1]].copy()
#
#
#sim2, followerchain2 = makefollowerchain(956,simdata,18)
#plt.subplot(1,2,2)
#platoonplot(meas2,followerchain,[],newfig = False,clr=colors[0])
#platoonplot(sim2,followerchain,[],newfig = False,clr=colors[1])
##custom legend
#labels = {0:'Measurements', 1:'Simulation'}
#lp = lambda i: plt.plot([],color=colors[i],linestyle='',
#                        label=labels[i],marker = 'o')[0]
#handles = [lp(i) for i in labels.keys()]
#plt.legend(handles=handles)
##plt.title('OVM after calibration (RMSE = '+str(round(np.mean(rmse),2))+' ft)')
#plt.title('Example of calibrated trajectories')
#plt.savefig('trajectorydata.png',dpi=200)

#%% section 1 stuff - results
#this makes the main table, parteo front plots, and shows the distributions of the fits as well.
#give a list of all the result outputs, we will create a table summarizing the results, a plot showing the pareto optimal algorithms, and a plot showing the distribution of the fits for those
#pareto optimal algorithms.
################
#if you run adjointcontent.py it will load all the results.
#also you need to have meas and platooninfo
import copy
import numpy as np
import matplotlib.pyplot as plt


#bfgs 0 , bfgs 7.5, bfgs inf, GA, NM, finite bfgs 0, 7.5, inf, gd 0 , 7.5, inf, sqp 0, 7.5, inf, spsa, finite gd inf, finite sqp inf, tnc 0, 7.5, inf, finite tnc inf
results = [bfgs_1, bfgs_2, bfgs_3, GA, NM, bfgsf_1, bfgsf_2, bfgsf_3, custom11, custom12, custom13, custom21, custom22, custom23, custom1[0],
           custom4[0], custom5[0], custom61, custom62, custom63, custom64[0], custom65, custom66 ]

#there is a problem with the calibrate custom results for gd and sqp. the objeval, gradeval, and hessevals will be wrong for those 4, and will need to be updated.
#this is a relatively minor problem but because everything will have to be recomputed it will take a long time to fix those numbers.
#the results themselves are correct, the analysis can proceed but the main table will need to be fixed at some point.

#the other thing is to do TNC with finite differences for last

#already have average RMSE (4)  and times (5).
#want % found global opt (0), average % over global (1) opt, average rmse over global opt (2), average initial guesses (3) average obj evals (6), average grad evals (7), avg hess evals (8)
tableres = dict(zip([x for x in range(len(results))] ,[np.zeros(9) for y in range(len(results))]  ))#initialize output for table
plotres1 = [[] for x in range(len(results))] #initialize output for cumulative plot of absolute amount (rmse) over global min
plotres2 = [results[j][2].copy() for j in range(len(results))] #initialize output for rmse
plotres2copy = copy.deepcopy(plotres2)

vehlist = [] #platoonlist results were on
for i in meas.keys():
    if len(platooninfo[i][4]) >=1:
        vehlist.append([[],i])

globalopt = [] #get global optimum for each vehicle
for i in range(len(vehlist)):
    vehopt = float('inf')
    for j in range(len(results)):
        curopt = results[j][2][i]
        if curopt < vehopt:
            vehopt = curopt
    globalopt.append(vehopt)

tol = 1/4 #global minumum needs to be within tol of the global optimum to be considered
entries = len(vehlist)
for i in range(len(vehlist)): #get % found global optimum for each algorithm
    vehopt = globalopt[i]
    for j in range(len(results)):
        curopt = results[j][2][i]
        if curopt <= vehopt + tol:
            tableres[j][0] += 1 #count the time the algorithm found the global opt
        tableres[j][1] += abs(curopt - vehopt)/vehopt #percent over global opt
        tableres[j][2] += abs(curopt-vehopt) #absolute amount over global opt in rmse
        #this is for plotting
        plotres1[j].append(abs(curopt-vehopt))

for j in range(len(results)): #average results
    tableres[j][0] = tableres[j][0] / entries
    tableres[j][1] = tableres[j][1] / entries
    tableres[j][2] = tableres[j][2] / entries
    tableres[j][4] = np.mean(results[j][2]) #rmses
    tableres[j][5] = np.mean(results[j][1]) #times

guesstypes = [0,1,2,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20, 21, 22] #GA, NM, spsa all don't have multiple guesses
for j in guesstypes:
    curguesses = 0
    for i in range(entries):
        curguesses += results[j][0][i][-1] #add the number of guesses used for vehicle i and algorithm j
    curguesses = curguesses/entries
    tableres[j][3] = curguesses #add the average number of guesses

types = [3,4] #objective evaluations for GA and NM
for j in types:
    curobjevals = 0
    for i in range(entries):
        curobjevals += results[j][0][i]['nfev']
    curobjevals = curobjevals / entries
    tableres[j][6] = curobjevals

types = [0,1,2,5,6,7] #bfgs
for j in types:
    curobjevals = 0
    for i in range(entries):
        curobjevals += results[j][0][i][2]['funcalls']
    curobjevals = curobjevals /entries
    tableres[j][6] = curobjevals
    tableres[j][7] = curobjevals

types = [8,9,10,11,12,13,14,15,16] #calibrate custom functions
hesstypes = [11,12,13,16] #functions with hessian evaluations
for j in types:
    curobjevals = 0
    curgradevals = 0
    curhessevals = 0
    for i in range(entries):
        curobjevals += results[j][0][i][2]['objeval']
        curgradevals +=results[j][0][i][2]['gradeval']
        if j in hesstypes:
            curhessevals +=results[j][0][i][2]['iter'] - results[j][0][i][-1]
    curobjevals = curobjevals /entries
    curgradevals = curgradevals/entries
    curhessevals = curhessevals/entries
    tableres[j][6] = curobjevals
    tableres[j][7] = curgradevals
    tableres[j][8] = curhessevals

types = [17,18,19,20, 21, 22] #TNC results
for j in types:
    curobjevals = 0
    curgradevals = 0
    for i in range(entries):
        curobjevals += results[j][0][i][1]
        curgradevals +=results[j][0][i][1]
    curobjevals = curobjevals /entries
    curgradevals = curgradevals/entries
    tableres[j][6] = curobjevals
    tableres[j][7] = curgradevals

#the above will do all the things needed to make tables of the results. we also want to make some plots.
#plots of time vs fit to identify algorithms on pareto front.
plt.close('all') #close all plots

def is_pareto_efficient(costs, return_mask = True): #copy pasted from stack exchange. finds pareto front assuming lower is better
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

mycmap = cm.get_cmap('viridis',7) #unused
pltdata = np.zeros((len(results),3)) #initialize data
for i in range(len(results)):
    pltdata[i,0] = tableres[i][5] #average time
    pltdata[i,1] = tableres[i][4] #average rmse
    pltdata[i,2] = tableres[i][0] #% time found global opt

#calculate the pareto front
temp = pltdata.copy()
temp[:,2] = -temp[:,2] #function uses lower is better, last column is a higher is better metric

paretofront = is_pareto_efficient(temp,True)
paretofront2 = paretofront.copy()
paretofront = pltdata[paretofront,:]

usec = np.array([1,1,1,5,6,1,1,1,2,2,2,3,3,3,7,2,3,4,4,4,4,4,4]) #for colors
usec = (usec-1)
#one option would be to convert the normalized values into colors using some colormap and the to_rgba function in matplotlib.colors
labels2 = {0:'BFGS',3:'GA',4:'NM',8:'GD',11:'SQP',17:'TNC',14:'SPSA'} #used for custom legend
labels = {0:'BFGS',3:'GA',4:'NM',8:'GD',11:'SQP',17:'TNC'} #used for custom legend
prop_cycle = plt.rcParams['axes.prop_cycle'] #for default coloring scheme
colors = prop_cycle.by_key()['color']

markers = ['*','*','*','^','<','*','*','*','+','+','+','x','x','x','h','+','x','s','s','s','s','s','s'] #marker types
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
for i in range(len(usec)): #this plots the points with custom colors and markers
#    plt.scatter(pltdata[i,0],pltdata[i,1],c =np.array([usec[i]]), marker = markers[i])
    plt.scatter(pltdata[i,0],pltdata[i,1], c = np.array([colors[usec[i]]]), marker = markers[i])

#code for custom legend; you make an empty plot with all the info for each type
lp = lambda i: plt.plot([],color=colors[usec[i]],linestyle='',
                        label=labels[i], marker=markers[i])[0]
handles = [lp(i) for i in labels.keys()]
plt.legend(handles=handles)
plt.xlabel('Average calibration time (sec)')
plt.ylabel('Average RMSE (ft)')
plt.ylim([6,10])
#circle the points on the pareto front
plt.scatter(paretofront[:,0],paretofront[:,1],s=90,facecolors='none',edgecolors='k')

plt.subplot(1,2,2)
for i in range(len(usec)):
#    plt.scatter(pltdata[i,0],pltdata[i,1],c =np.array([usec[i]]), marker = markers[i])
    plt.scatter(pltdata[i,0],pltdata[i,2], c = np.array([colors[usec[i]]]), marker = markers[i])

#code for custom legend; you make an empty plot with all the info for each type
lp = lambda i: plt.plot([],color=colors[usec[i]],linestyle='',
                        label=labels2[i], marker=markers[i])[0]
handles = [lp(i) for i in labels2.keys()]
plt.legend(handles=handles)
plt.xlabel('Average calibration time (sec)')
plt.ylabel('% of time found global optimum')
plt.scatter(paretofront[:,0],paretofront[:,2],s=90,facecolors='none',edgecolors='k')
plt.savefig('Paretofig.png',dpi=200)

###############################
#plots of distributions of rmse and errors above global optimum
#for j in range(len(results)): #this will give a result 0 rmse over the global opt if we count it as having found the global optimum
#    for i in range(entries):
#        if plotres1[j][i]<=tol:
#            plotres1[j][i] = 0
for j in range(len(results)):
    plotres1[j].sort()
    plotres2[j].sort()

y = np.array(range(entries))
y = (y+1)/entries

#plt.figure()
#plt.plot(np.log(plotres1[0]),y, np.log(plotres1[1]),y,np.log(plotres1[2]),y,np.log(plotres1[3]),y)
#
#######################original plot
#labels3 = ['adj BFGS-0','adj BFGS-7.5','GA','fin BFGS-0','fin BFGS-7.5','adj TNC-0','adj TNC-7.5', 'adj TNC-inf']
#plt.figure(figsize=(10,4))
#plt.subplot(1,2,1)
#for i, j in enumerate(paretofront2):
#    if j: #if i is in the pareto front
#        plt.plot(plotres2[i],y)
#    plt.legend(labels3)
#    plt.xlim([2,12])
#    plt.xlabel('RMSE (ft)')
#    plt.ylabel('Cumulative % of solutions')
#
#plt.subplot(1,2,2)
##plt.plot(np.log10(plotres1[0]),y, np.log10(plotres1[1]),y,np.log10(plotres1[2]),y,np.log10(plotres1[3]),y)
#for i, j in enumerate(paretofront2):
#    if j: #if i is in the pareto front
#        plt.plot(np.log10(plotres1[i]),y)
#
#    plt.legend(labels3)
#    plt.xlabel('log of RMSE over global optimum (ft)')
#    plt.ylabel('Cumulative % of solutions')
#plt.savefig('CDF.png',dpi=200)
##########################

################second plot
labels3 = ['GA','adj BFGS-0','adj TNC-0']
plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
for i, j in enumerate([3,0,17]):
#    if j: #if i is in the pareto front
#        plt.plot(np.log10(plotres1[i]),y)

    plt.plot(np.log10(plotres1[j]),y)
    plt.legend(labels3, loc = 'lower right')
    plt.xlabel('log of tolerance (ft)')
    plt.ylabel('% found global optimum')
    plt.xlim([-7,1])
    plt.ylim([.3,1.01])

plt.subplot(2,2,2)
#plt.plot(np.log10(plotres1[0]),y, np.log10(plotres1[1]),y,np.log10(plotres1[2]),y,np.log10(plotres1[3]),y)
for i, j in enumerate([3,0,17]):
#    if j: #if i is in the pareto front
#        plt.plot(np.log10(plotres1[i]),y)

    plt.plot(np.log10(plotres1[j]),y)
    plt.legend(labels3, loc = 'lower right')
    plt.xlabel('log of tolerance (ft)')
    plt.ylabel('% found global optimum')
    plt.xlim([-1.5,.5])
    plt.ylim([.9,1.003])

labels4 = ['GA','adj BFGS-0','adj BFGS-7.5', 'fin BFGS-0','fin BFGS-7.5','adj TNC-0','adj TNC-7.5','adj TNC-inf']
plt.subplot(2,2,3)
#plt.plot(np.log10(plotres1[0]),y, np.log10(plotres1[1]),y,np.log10(plotres1[2]),y,np.log10(plotres1[3]),y)
for i, j in enumerate([3,0,1,5,6,17,18,19]):
#    if j: #if i is in the pareto front
#        plt.plot(np.log10(plotres1[i]),y)

    plt.plot(np.log10(plotres1[j]),y)
    plt.legend(labels4, loc = 'lower right')
    plt.xlabel('log of tolerance (ft)')
    plt.ylabel('% found global optimum')
    plt.xlim([-7,1])
    plt.ylim([.3,1.01])

plt.subplot(2,2,4)
#plt.plot(np.log10(plotres1[0]),y, np.log10(plotres1[1]),y,np.log10(plotres1[2]),y,np.log10(plotres1[3]),y)
for i, j in enumerate([3,0,1,5,6,17,18,19]):
#    if j: #if i is in the pareto front
#        plt.plot(np.log10(plotres1[i]),y)

    plt.plot(np.log10(plotres1[j]),y)
    plt.legend(labels4, loc = 'lower right')
    plt.xlabel('log of tolerance (ft)')
    plt.ylabel('% found global optimum')
    plt.xlim([-1.5,.5])
    plt.ylim([.9,1.003])



plt.figure(figsize = (10,4))
plt.subplot(2,2,1)
for i, j in enumerate([3,0,17]):

    plt.plot(plotres2[j],y)
    plt.legend(labels3)
    plt.xlim([2,12])
    plt.xlabel('RMSE (ft)')
    plt.ylabel('Cumulative % of solutions')


plt.subplot(2,2,3)
for i, j in enumerate([3,0,1,5,6,17,18,19]):

    plt.plot(plotres2[j],y)
    plt.legend(labels4)
    plt.xlim([2,12])
    plt.xlabel('RMSE (ft)')
    plt.ylabel('Cumulative % of solutions')

plotres3 = []
plotres4 = []
#do difference between LC no LC vehicles
for i in range(len(plotres2copy)):
    curLC = []
    curnoLC = []
    for j in range(len(vehlist)):
        if len(platooninfo[vehlist[j][1]][4]) >1:
            curLC.append(plotres2copy[i][j])
        else:
            curnoLC.append(plotres2copy[i][j])
    curLC.sort()
    curnoLC.sort()
    plotres3.append(curLC)
    plotres4.append(curnoLC)

y1 = np.array(range(len(plotres3[0])))
y1 = (y1+1)/len(plotres3[0])
y2 = np.array(range(len(plotres4[0])))
y2 = (y2+1)/len(plotres4[0])

prop_cycle = plt.rcParams['axes.prop_cycle'] #for default coloring scheme
colors = prop_cycle.by_key()['color']

plt.subplot(2,2,2)
for i, j in enumerate([3,0,17]):

    plt.plot(plotres3[j],y1)
    plt.legend(labels3)
    plt.xlim([2,12])
    plt.xlabel('RMSE (ft)')
    plt.ylabel('Cumulative % of solutions')
for i, j in enumerate([3,0,17]):

    plt.plot(plotres4[j],y2, color = colors[i],linestyle = '--')

plt.subplot(2,2,4)
for i, j in enumerate([3,0,1,5,6,17,18,19]):

    plt.plot(plotres3[j],y1)
    plt.legend(labels4)
    plt.xlim([2,12])
    plt.xlabel('RMSE (ft)')
    plt.ylabel('Cumulative % of solutions')
for i, j in enumerate([3,0,1,5,6,17,18,19]):

    plt.plot(plotres4[j],y2, color = colors[i],linestyle = '--')




#plt.savefig('CDF.png',dpi=200)



#%%%%%%% #would like to see how much of a difference is made by the lane changing vehicles.

#results = [bfgs_1, bfgs_2, bfgs_3, GA, NM, bfgsf_1, bfgsf_2, bfgsf_3, custom11, custom12, custom13, custom21, custom22, custom23, custom1[0],
#           custom4[0], custom5[0], custom61, custom62, custom63, custom64[0], custom64[0], custom64[0] ]
#
#noLClist = [] #platoonlist results were on all vehicles
#for i in meas.keys():
#    if len(platooninfo[i][4]) ==1:
#        noLClist.append(1)
#    elif len(platooninfo[i][4]) > 1:
#        noLClist.append(0)
#
#ind = [0,3,-6] #these are the ones we want to look at.
#res = []
#noLCcount = sum(noLClist)
#LCcount = len(noLClist) - noLCcount
#for i in ind:
#    noLCrmse = 0
#    LCrmse = 0
#    for j in range(len(noLClist)):
#        if noLClist[j]==1: #if it's no LC then add the rmse
#            noLCrmse += results[i][2][j] #i = current algorithm results, 2 = rmse, j = current vehicle
#        else:
#            LCrmse += results[i][2][j]
#    noLCrmse = noLCrmse / noLCcount
#    LCrmse = LCrmse / LCcount
#    res.append(noLCrmse)
#    res.append(LCrmse)
#



#%% #redo 1424
#section 2 stuff - platoon of 10
#note - can modify existing algorithms to add option not to reset sim after a platoon. Also needs to make bounds/guesses the right size.
#how to get guesses for platoons?
#to evaluate rmse, want to always look at the total rmse of the platoon rather than the rmse averaged over all the vehicles.
#want to show how the algorithms scale. Only going to consider TNC, bfgs, and GA since those were the ones on the pareto front.

#from calibration import *
import math

#originally we did 6 tests
#scalingtest.pkl - uses platoon starting at 956
#scalingtest.pkl2 - uses platoon starting at 1424
#scalingtest3.pkl - redoes scalingtest.pkl2 TNC and bfgs algorithms with different rules for cutoff2, updating guesses, bfgs maximum iterations
#scalingtest4.pkl - redoes 1424 test for adjoint TNC and bfgs with the above different rules
#scalingtest5.pkl - was supposed to do a new platoon but had bug
#scalingtest6.pkl - see above

#now we are just going to redo everything so it's cleaner, more reproducible, and easier to work with

#%%

def platoontest(vehlist, meas, platooninfo):
    def pltn_helper(vehlist,size):
        out = []
        length =len(vehlist)
        num_full = math.floor(length/size)
        leftover = length-num_full*size
        for i in range(num_full): #add full platoons
    #        curplatoon = [[],vehlist[i:i+size]]
            curplatoon = [[]]
            for j in range(size):
                curplatoon.append(vehlist[i+j])
            out.append(curplatoon)
        if leftover >0:
            temp = [[]]
            for j in range(leftover):
                temp.append(vehlist[i+j])
            out.append(temp)
        return out
    #pltn = [[[],i] for i in followerchain.keys()]
    maxsize = len(vehlist)
    lists = [] #lists we will test
    for i in range(maxsize):
        lists.append(pltn_helper(vehlist,i+1))

    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]

    output = []
    output2 = []

    #initially did 4 first tests with cutoff2=2.5

    for i in range(maxsize):
        out = calibrate_tnc2(plist_nor,bounds_nor,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,False,5,cutoff=0,cutoff2=5.5,order=1,budget = 3)
        out2 = calibrate_bfgs2(plist_nor,bounds_nor,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_objder,None,OVM,OVMadjsys,OVMadj,False,5,cutoff=0,cutoff2=5.5,order=1,budget = 3)
        output.append(out)
        output2.append(out2)

    output3 = []
    output4 = []
    output5 = []
    for i in range(3): #change this part here depending on whether you want to show that these scale awfully or not
        out3 = calibrate_GA(bounds_nor,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_obj,None,OVM,OVMadjsys,OVMadj,False,5,order=1)
        out4 = calibrate_tnc2(plist_nor,bounds_nor,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_obj,platoonobjfn_fder,OVM,OVMadjsys,OVMadj,False,5,cutoff=0,cutoff2=5.5,order=1,budget = 3)
        out5 = calibrate_bfgs2(plist_nor,bounds_nor,meas,platooninfo,lists[i],makeleadfolinfo,platoonobjfn_obj,platoonobjfn_fder,OVM,OVMadjsys,OVMadj,False,5,cutoff=0,cutoff2=5.5,order=1,budget = 3)
        output3.append(out3)
        output4.append(out4)
        output5.append(out5)

    return output, output2, output3, output4, output5, lists

#meas2, followerchain = makefollowerchain2(525,data,9) #525, 956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest21.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(956,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest22.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(1424,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest23.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(1736,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest24.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)

#meas2, followerchain = makefollowerchain2(1914,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest25.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(1831,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest26.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(2436,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest27.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(977,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest28.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(314,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest29.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)
#
#meas2, followerchain = makefollowerchain2(1226,data,9) #956, 1424, 1736
#vehlist = list(followerchain.keys())
#output, output2, output3, output4, output5, lists = platoontest(vehlist, meas, platooninfo)
#
#with open('scalingtest210.pkl','wb') as f:
#    pickle.dump([output,output2,output3,output4,output5, lists],f)

    #%%
    #make a plot of the results which will show how the different algorithms scale.
    #initialize this
#rmse = {}
#times = {}
#%%
##want plots of
##time versus size, eqiuiv obj evals vesus size (want standard deviations for these as well). in these we will only look at the platoons of exactly the right size, and not the leftovers
##overall rmse achieved by the different platoon sizes; these include the leftovers.
##would like to show average RMSEs (no std dev because it will be too big) and average % improvement for each platoon size with std devs.
#
##can say something about the calibration being done in pairs above and sequentially below. When doing sequentially you get higher error (since it can propogate)
##but the overall performance of each algorithm (relative to each other) is still the same.
#
#
##rmse = {}
##times = {}
#
#
#saveddatalist = ['scalingtest21.pkl','scalingtest22.pkl','scalingtest23.pkl','scalingtest24.pkl','scalingtest25.pkl',
#                 'scalingtest26.pkl','scalingtest27.pkl','scalingtest28.pkl','scalingtest29.pkl','scalingtest210.pkl']
#rmse = dict(zip([0,1,2,3,4],[[[] for i in range(10)] for j in range(5)]))
#relrmse = dict(zip([0,1,2,3,4],[[[] for i in range(10)] for j in range(5)]))
##
##
#from calibration import *
#for z in saveddatalist: #iterate over each platoon
#    with open(z,'rb') as f: #open current data
#        output,output2,output3,output4,output5, lists = pickle.load(f)
#    results = [output, output2, output3, output4, output5]
#    mytype = [0, 1, 2,0,1 ]
#    fullplatoon = lists[-1][0]
#    for i in range(len(results)): #iterate over each algorithm
#        currmse = []
#        curtimes = []
#        for j in range(len(results[i])): #each platoon size in each algorithm
#            if mytype[i] == 0:
#                objlist = [z[-2] for z in results[i][j][0]] #tnc
#            elif mytype[i] ==1:
#                objlist = [z[1] for z in results[i][j][0]] #bfgs
#            else:
#                objlist = [z['fun'] for z in results[i][j][0]]
#            rmse1 = convert_to_rmse(sum(objlist),platooninfo,fullplatoon)
#            if j ==0:
#                firstrmse = rmse1.copy()
#
#            rmse[i][j].append(rmse1)
#            relrmse2 = (rmse1-firstrmse)/firstrmse
##            relrmse2 = (rmse1-firstrmse)
#            relrmse[i][j].append(relrmse2)
##            timelist = [z for z in results[i][j][1] ]
##            curtimes.append(sum(timelist))
##        rmse[i] = currmse
##        times[i] =curtimes
#    #objlist = [i[-2] for i in out1[0]]
#
#saveddatalist = ['scalingtest21.pkl','scalingtest22.pkl','scalingtest23.pkl','scalingtest24.pkl','scalingtest25.pkl',
#                 'scalingtest26.pkl','scalingtest27.pkl','scalingtest28.pkl','scalingtest29.pkl','scalingtest210.pkl']
#equivobjeval = dict(zip([0,1,2,3,4],[[[] for i in range(10)] for j in range(5)]))
#equivtime = dict(zip([0,1,2,3,4],[[[] for i in range(10)] for j in range(5)]))
#for i in saveddatalist: #iterate over each platoon
#    with open(i,'rb') as f: #open current data
#        output,output2,output3,output4,output5, lists = pickle.load(f)
#    results = [output, output2, output3, output4, output5]
#    mytype = [0, 1, 2,3,4 ] #adj tnc, adj bfgs, ga, fin tnc, fin bfgs
#    fullplatoon = lists[-1][0]
#    m = 5
#    for j in range(len(results)): #each algorithm
#        for l in range(len(results[j])): #all subdivisions in platoon
#            for k in range(len(results[j][l][0])): #every subplatoon in each subdivision
#                #now we need to do something different for each type
#                if mytype[j] == 0:
#                    cursize = int(len(results[j][l][0][k][0])/m) #this is the size of the current platoon
#                    curobj = results[j][l][0][k][1] #number function calls
#                    curobj = curobj*4
#                    equivobjeval[j][cursize-1].append(curobj)
#                    curtime = results[j][l][1][k]
#                    equivtime[j][cursize-1].append(curtime)
#                if mytype[j] == 3:
#                    cursize = int(len(results[j][l][0][k][0])/m) #this is the size of the current platoon
#                    curobj = results[j][l][0][k][1]
#                    curobj = curobj*(5*cursize+1)
#                    equivobjeval[j][cursize-1].append(curobj)
#                    curtime = results[j][l][1][k]
#                    equivtime[j][cursize-1].append(curtime)
#                if mytype[j] == 1:
#                    cursize = int(len(results[j][l][0][k][0])/m) #this is the size of the current platoon
#                    curobj = results[j][l][0][k][2]['funcalls']
#                    curobj = curobj*4
#                    equivobjeval[j][cursize-1].append(curobj)
#                    curtime = results[j][l][1][k]
#                    equivtime[j][cursize-1].append(curtime)
#                if mytype[j] == 4:
#                    cursize = int(len(results[j][l][0][k][0])/m) #this is the size of the current platoon
#                    curobj = results[j][l][0][k][2]['funcalls']
#                    curobj = curobj*(5*cursize+1)
#                    equivobjeval[j][cursize-1].append(curobj)
#                    curtime = results[j][l][1][k]
#                    equivtime[j][cursize-1].append(curtime)
#                if mytype[j] == 2:
#                    cursize = int(len(results[j][l][0][k]['x'])/m) #this is the size of the current platoon
#                    curobj = results[j][l][0][k]['nfev']
##                    curobj = curobj*4
#                    equivobjeval[j][cursize-1].append(curobj)
#                    curtime = results[j][l][1][k]
#                    equivtime[j][cursize-1].append(curtime)
#plt.close('all')
#plt.figure()
##plt.subplot(1,2,2)
#for i in equivobjeval.keys(): #for each algorithm
#    if i in [1]: #can choose which algos you want to show...
#        continue
#    nveh = len(equivobjeval[i])+1
#    x = list(range(5,5*nveh,5)) #number of parameters
#    y = []
#    yerr = []
#    yerrlog = []
#    for j in equivobjeval[i]:
#        y.append(np.mean(j))
#        yerr.append(np.std(j))
#        yerrlog.append(np.std(np.log10(j)))
##    plt.errorbar(x,y,yerr=yerr)
##    plt.plot(x,y,'.')
##    plt.plot(x,y)
#    print(y)
##    plt.plot(x,y)
##    print(yerrlog)
#    plt.errorbar(x,np.log10(y),yerr = yerrlog,capsize=3,marker='.') #errorbar is for plotting points with standard deviation
##    plt.errorbar(x,np.log10(y),yerr = yerrlog,capsize=3,marker='.',linestyle='')
#plt.xlabel('number of parameters')
#plt.ylabel('log of equivalent objective evaluations')
#plt.legend(['Adj TNC', 'GA','Fin TNC','Fin BFGS'])
#
##this plots the times but they just look the same as plotting the objective evaluations....
##plt.subplot(1,2,1)
##for i in equivtime.keys():
##    nveh = len(equivobjeval[i])+1
##    x = list(range(5,5*nveh,5))
##    y = []
##    for j in equivtime[i]:
##        y.append(np.mean(j))
###    plt.errorbar(x,y,yerr=yerr)
###    plt.plot(x,y,'.')
##    plt.plot(x,y)
###    plt.plot(x,np.log10(y))
###    print(yerrlog)
###    plt.errorbar(x,np.log10(y),yerr = yerrlog)
#
#plt.figure()
#
#for i in rmse.keys():
#    if i in [1,3,4]:
#        continue
#    x = list(range(1,11,1))
#    y = []
#    rely = []
#    relyerr = []
#    for j in range(len(rmse[i])):
#        y.append(np.mean(rmse[i][j]))
#        rely.append(np.mean(relrmse[i][j]))
#        relyerr.append(np.std(relrmse[i][j]))
#    ax1 = plt.subplot(1,2,1)
#    ax1.plot(x,y,'.',markersize=8)
#    if i in [1,2,3,4]:
#        continue
#    ax2 = plt.subplot(1,2,2)
#    ax2.errorbar(x,rely,yerr=relyerr,marker='.',markersize=9,capsize=2,linestyle='')
#
#ax1.legend(['Adj TNC','GA'])
#ax2.legend(['Adj TNC'])
#ax2.set_xlabel('platoon size')
#ax2.set_ylabel('relative improvement in RMSE')
#ax1.set_xlabel('platoon size')
#ax1.set_ylabel('overall RMSE')



