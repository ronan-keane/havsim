# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 05:25:49 2019

@author: rlk268
"""
from calibration import * 
#vehicles which can be used to motivation introduction in relaxation paper: 156, 956, 
#idea is to show a similar picture as to the first one in adjoint paper but we want to show 2 things: first of all the relaxation phenomenon happening empiracly, 
#and second of all we want to show that the calibraiton looks bad for car following because there are no lane changing dynamics. 
#meas2, followerchain = makefollowerchain(25,data,30)
#
##platoonplot(meas2,followerchain,[])
#plt.close('all')
##vehicles 156 and 194 in these you can see the relaxation phenomenon. 
#platoonplot(meas2,followerchain,[[],143,156,158])
#
##platoonplot(meas2,followerchain,[[],191,194,204,206,218])
#
#meas2, followerchain = makefollowerchain(880,data,25,picklane=2)
#
##platoonplot(meas2,followerchain,[])
##plt.close('all')
##vehicles 156 and 194 in these you can see the relaxation phenomenon. 
#platoonplot(meas2,followerchain,[])
#
#plt.figure()
#plotspeed(meas,sim,platooninfo,156,True)
#
#plt.figure()
#plothd(meas,sim,platooninfo,156)
#%%
#YOU NEED RESULTS FROM RELAXCONTENT, ADJOINTCONTENT, AND POSTERCONTENT TO RUN THIS. (so run those scripts to load the results)
#this follows the script used in adjointpaperplots to make the trajectory data figure.
#note you need to manually combine these two choosen platoons into 1 manually

#pick some random vehicles to show as examples 
#just 2 is fine. can we also make an extra table that shows the average RMSE as you look at vehicles with an increasing number of lane changes? 

#360,3,0
#1045,3,0

temp = [360,3,0] #this is the bottom vehicle in the figure (373). 
#headway changes from 50 ft to 10 ft immediately after the change. Then it relaxes back to 50 feet over approximately 10 seconds 

#temp = [1045,3,0] #top vehicle in figure (1063).
#headway changes from  100 to 50 ft immediately after the change, it relaxes back to 100 feet over approximately 15 seconds. 

#default colors in matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle'] #for default coloring scheme
colors = prop_cycle.by_key()['color']
##
sim = copy.deepcopy(meas)
resultsnor = custom61 #will be using the results from TNC-3 
#2 results here; with LC and without LC. 
results = [LC_r2, noLC2]
#results2 = noLC2
meas2, followerchain = makefollowerchain(temp[0],data,temp[1],picklane=temp[2])
#plt.figure(figsize = (10,4))
plt.subplot(2,3,4)
platoonplot(meas2,followerchain,[],newfig=False, clr=colors[0])
plt.title('Trajectory data (measurements)')
plt.ylim([380,780]) #this is for the first vehicle with 360 leader 
plt.xlim([1440,1540])
plt.xlabel('time (.1 seconds)')
#plt.xlim([3170,3320])
#plt.ylim([605,1160])

#now show the calibrated trajectories
#need to get correct indices for noLC vehicles, and also LC vehicles 
vehlist2 = [] #debug purposes
for i in meas.keys():
    if len(platooninfo[i][4]) >1:
        vehlist2.append([[],i])
        
vehlist22 = []
for i in meas.keys():
    if len(platooninfo[i][4]) ==1:
        vehlist22.append([[],i])
        
pltveh = list(followerchain.keys()) #can change this line to choose what we are plotting
vehlist = []  #this will be a list of the indices of the results we want to plot
for i in pltveh: #this works
    if len(platooninfo[i][4])==1:
        ind = 1
        for count, j in enumerate(vehlist22):
            if j[1]==i:
                vehlist.append((ind,count))
    elif len(platooninfo[i][4]) >1:
        ind = 0
        for count, j in enumerate(vehlist2):
            if j[1]==i:
                vehlist.append((ind,count))
    
    
rmse = []
for count,i in enumerate(vehlist): 
    curplatoon = [[],pltveh[count]] #curplatoon 
    print(curplatoon)
    #need to do something different depending on which calibration result the vehicle is using 
    if i[0] == 0:
        leadinfo,folinfo,rinfo = makeleadfolinfo_r3(curplatoon,platooninfo,meas)
        obj = platoonobjfn_obj(results[i[0]][0][i[1]][0],OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,True,6)
    else: 
        leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,meas)
        obj = platoonobjfn_obj(results[i[0]][0][i[1]][0],OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo)
        
        
    curr = convert_to_rmse(obj,platooninfo,curplatoon)
    rmse.append(curr)
    if count ==0:
        simdata = sim[curplatoon[1]].copy()
    else:
        simdata = np.append(simdata,sim[curplatoon[1]],axis=0)
    sim[curplatoon[1]] = meas[curplatoon[1]].copy()
    
    
#now show the calibrated trajectories
vehlistnor2 = [] #debug purposes
for i in meas.keys():
    if len(platooninfo[i][4]) >=1:
        vehlistnor2.append([[],i])
        
pltveh = list(followerchain.keys())
vehlistnor = []  #this will be a list of the indices of the results we want to plot
for i in pltveh: #this works
    for count, j in enumerate(vehlistnor2):
        if j[1]==i:
            vehlistnor.append(count)
    
sim = copy.deepcopy(meas)    
rmse = []
for count,i in enumerate(vehlistnor): 
    curplatoon = [[],pltveh[count]] #curplatoon 
    print(curplatoon)
    leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,meas)
    obj = platoonobjfn_obj(resultsnor[0][i][0],OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo)
    curr = convert_to_rmse(obj,platooninfo,curplatoon)
    rmse.append(curr)
    if count ==0:
        simdata2 = sim[curplatoon[1]].copy()
    else:
        simdata2 = np.append(simdata2,sim[curplatoon[1]],axis=0)
    sim[curplatoon[1]] = meas[curplatoon[1]].copy()
        

sim2, followerchain2 = makefollowerchain(temp[0],simdata,temp[1],picklane=temp[2])

sim3, followerchain3 = makefollowerchain(temp[0],simdata2,temp[1],picklane=temp[2])


plt.subplot(2,3,6)
platoonplot(meas2,followerchain,[],newfig = False,clr=colors[0])
platoonplot(sim2,followerchain,[],newfig = False,clr=colors[1])
#custom legend 
labels = {0:'Measurements', 1:'Simulation'}
lp = lambda i: plt.plot([],color=colors[i],linestyle='',
                        label=labels[i],marker = 'o')[0]
handles = [lp(i) for i in labels.keys()]
plt.legend(handles=handles)
plt.ylim([380,780]) #this is for the first vehicle with 360 leader 
plt.xlim([1440,1540])
#plt.xlim([3170,3320])
#plt.ylim([605,1160])
#plt.title('OVM after calibration (RMSE = '+str(round(np.mean(rmse),2))+' ft)')
plt.title('Relaxation Phenomenon')
plt.xlabel('time (.1 seconds)')

plt.subplot(2,3,5)
platoonplot(meas2,followerchain,[],newfig = False,clr=colors[0])
platoonplot(sim3,followerchain,[],newfig = False,clr=colors[1])
#custom legend 
labels = {0:'Measurements', 1:'Simulation'}
lp = lambda i: plt.plot([],color=colors[i],linestyle='',
                        label=labels[i],marker = 'o')[0]
handles = [lp(i) for i in labels.keys()]
plt.legend(handles=handles)
#plt.title('OVM after calibration (RMSE = '+str(round(np.mean(rmse),2))+' ft)')
plt.title('Base Car Following model')
plt.xlabel('time (.1 seconds)')
plt.ylim([380,780]) #this is for the first vehicle with 360 leader 
plt.xlim([1440,1540])
#plt.xlim([3170,3320])
#plt.ylim([605,1160])
#plt.savefig('trajectorydata.png',dpi=200)



#%%

#make optimal velocity plot for one of the noLC vehicles. Get some plots showing: initial guess, calibration, example of unrealistic. 


#noLClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])==1:
#        noLClist.append([[],i])
#        
#
#sim = copy.deepcopy(meas)
#
#
#p = [10*3.3,.086/3.3, 1.545, 2, .175]
#
#
#
#obj = platoonobjfn_obj(p,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)
        
        


