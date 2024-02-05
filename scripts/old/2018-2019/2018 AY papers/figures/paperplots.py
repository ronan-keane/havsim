# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:35:48 2019
This houses the scripts used to make the figures in the relax paper.
Note we have all the scripts for tables in a seperate file called relaxation.py
Everything is divided up into sections. results from calibration saved in files relaxcontent and postercontent. One of the figures is in makeposter.
@author: rlk268
"""
import havsim.calibration.helper as helper
import matplotlib.pyplot as plt
from matplotlib import rc #latex title
import havsim
import numpy as np
import copy
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']}) #this will change every to computer modern font
### for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True) #enable latex
#
#
###plot the headway for a vehicle, with and without the modification to headway added.
#test = []
#for i in meas.keys():
#    unused,unused,rinfo = makeleadfolinfo_r6([[],i],platooninfo,meas)
#    pos = False
#    neg = False
#    for j in rinfo[0]:
#        if j[1] >0:
#            pos = True
#        if j[1] < 0:
#            neg = True
#    if len(rinfo[0])> 1 and pos and neg: #2 or more lane changes, also has both negative and positive lane change
#        test.append([[],i])
#
#test2 = []
#for i in meas.keys():
#    if len(platooninfo[i][4])==2:
#        test2.append([[],i])
#
#LClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])>1:
#        LClist.append([[],i])
#
#useveh = test[50] #43 50 are the ones I used in the paper images  #50 has 2 lane changes, one positive one negative. #43 has a bunch of lane changes.
##useveh = test2[13] #13 is a good example of a single lane change
#ind = float('inf')
#for i in range(len(LClist)):
#    if useveh == LClist[i]:
#        ind = i
#        break
#
##having found the vehicle we're interested in, now plot the trajectory and show the headway with and without the extra factor added.
###have results of calibration loaded in
        #%% this section plots the headway, and headway + relaxation amounts for the calibration results
#p = LC_2r[0][ind][0] #can change RHS here to plot different calibration  amounts; note you will need to manually uncomment the relevant parts for 1/2 parameter relax
#sim = copy.deepcopy(meas)
#leadinfo,folinfo,rinfo = makeleadfolinfo_r3(useveh,platooninfo,sim)
#
##obj = platoonobjfn_obj(p,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, useveh, leadinfo, folinfo, rinfo,True,6)
#obj = platoonobjfn_obj2(p,OVM, OVMadjsys, OVMadj2, meas, sim, platooninfo, useveh, leadinfo, folinfo, rinfo,True,7)
#
#######
#datalen = 9
#rp = p[-1]
#print('negative relaxation amount is '+str(rp))
#rp2 = p[-2]
#print('positive relaxation amount is '+str(rp2))
#h = .1
#print(rinfo[0])
#
#my_id = useveh[1]
#t_nstar,t_n,T_nm1,T_n = platooninfo[my_id][0:4]
#frames = [t_n,T_nm1]
#lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
#for j in leadinfo[0]:
#    curleadid = j[0] #current leader ID
#    leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
#    lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
#
#truelead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
#for j in leadinfo[0]:
#    curleadid = j[0] #current leader ID
#    leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
#    truelead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
#
##relax,unused = r_constant(rinfo[0],frames,T_n,rp,False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.
#relax,unused,unused = r_constant3(rinfo[0],frames,T_n,rp2,rp,False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.
#
#headway =  lead[:,2]-sim[my_id][t_n-t_nstar:,2]-lead[:,6] #don't plot this entire thing because the headway is undefined at the end
#
#plt.close('all')
#plt.subplot(1,2,1)
#plt.plot(sim[my_id][t_n-t_nstar:T_nm1+1-t_nstar,1],headway[:T_nm1+1-t_n],'k')
#plt.subplot(1,2,2)
#plt.plot(sim[my_id][t_n-t_nstar:T_nm1+1-t_nstar,1],headway[:T_nm1+1-t_n]+relax[:T_nm1+1-t_n],'k')

#%% use true measurments and some generic rp value instead of the actual calibration results
sim = copy.deepcopy(meas)
datalen = 9
rp = 15
h = .1
veh = 93

leadinfo,folinfo,rinfo = helper.makeleadfolinfo([veh],platooninfo,sim)

my_id = veh
t_nstar,t_n,T_nm1,T_n = platooninfo[my_id][0:4]
frames = [t_n,T_nm1]
lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
for j in leadinfo[0]:
    curleadid = j[0] #current leader ID
    leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
    lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation

truelead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
for j in leadinfo[0]:
    curleadid = j[0] #current leader ID
    leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
    truelead[j[1]-t_n:j[2]+1-t_n,:] = meas[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation

relax,unused = havsim.calibration.opt.r_constant(rinfo[0],frames,T_n,rp,False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.
#relax,unused,unused = r_constant3(rinfo[0],frames,T_n,rp2,rp,False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.

headway =  lead[:,2]-sim[my_id][t_n-t_nstar:,2]-lead[:,6] #don't plot this entire thing because the headway is undefined at the end

LCtimes = []
for i in rinfo[0]:
    LCtimes.append(i[0]-1) #this is the last time you have the previous leader
LCtimes.append(T_nm1) #this isn't actually a LC time but we need to append this for the plotting to work correctly
LCtimes = np.asarray(LCtimes)

plt.close('all')
plt.figure(figsize=(15,4))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.subplot(1,2,1)

plt.ylim(10,55) #5, 100 for vehicle 50 #60, 275 for vehicle 43 #10 55 for vehicle 13
# plt.xlim(140,165) #for vehicle 50 140, 165 #none for vehicle 43 #60,105 for 13 test2

plt.ylabel('headway (ft)',fontsize = 11)
plt.xlabel('time (s)',fontsize = 11)

prevsim = t_n-t_nstar
prevhd = 0
for i in LCtimes:
    plt.plot(sim[my_id][prevsim:i+1-t_nstar,1]/10,headway[prevhd:i+1-t_n],'k')
    prevsim = i+1-t_nstar
    prevhd = i+1-t_n
#plt.plot(LCtimes/10,headway[LCtimes-t_n],'r.',markersize=8) #these are kinda clunky but can add them if you want
plt.subplot(1,2,2)

plt.ylim(10,55)
# plt.xlim(140,165)

plt.ylabel('headway + '+r' $r(t)\gamma_s$'+' (ft)',fontsize = 11)
plt.xlabel('time (s)',fontsize = 11)

prevsim = t_n-t_nstar
prevhd = 0
for i in LCtimes:
    plt.plot(sim[my_id][prevsim:i+1-t_nstar,1]/10,headway[prevhd:i+1-t_n]+relax[prevhd:i+1-t_n],'k')
    prevsim = i+1-t_nstar
    prevhd = i+1-t_n
#plt.plot(LCtimes/10,headway[LCtimes-t_n-1]+relax[LCtimes-t_n],'r.',markersize=8)
#plt.savefig('headway3.png',dpi=200)

#%% histogram for main result tables ; need to have the results loaded in from relaxation
#d1 = out[1]+out[2]+out[3]+out[4]
#d3 = out3[1]+out3[2]+out3[3]+out3[4]
#d4 = out4[1]+out4[2]+out4[3]+out4[4]
#d5 = out5[1]+out5[2]+out5[3]+out5[4]
#plt.close('all')
#plt.xlabel('RMSE (ft)')
#plt.ylabel('Frequency')
#plt.hist([d3,d4,d5],bins = 10, range = (.15,30))
#plt.legend(['Relax', '2 Param Relax', 'Baseline'])
#plt.savefig('Newellrelax.png',dpi=300)

#%% #have merger results from relaxation loaded in
##recall that out01 and out02 are for 1p, out03, out04 are for 2p, out05, out06 are for no relax. odd numbers signify merger rule, even numbers are normal LC
##each number is divided up into [0] and [1] index, [0] is positive gamma, [1] is negative gamma
#
##out01[0].extend(out02[0])
##out01[1].extend(out02[1])
##
##out03[0].extend(out04[0])
##out03[1].extend(out04[1])
##
##out05[0].extend(out05[0])
##out05[1].extend(out05[1])
#
#merge1 = out01[0]+out01[1]
#merge2 = out03[0]+out03[1]
#merge3 = out05[0]+out05[1]
#
##plt.hist([merge1,merge3],cumulative = True, density = True, histtype = 'step')
#plt.close('all')
#plt.xlabel('RMSE (ft)')
#plt.ylabel('Frequency')
#plt.hist([merge1,merge3],bins = 12)
#plt.legend(['Relax','Baseline'])
#plt.savefig('merger.png',dpi=300)


#%% Here we are going to make plots that show the difference between calibrating with the relaxation phenomenon and calibrating without it
###assume you have the relevant results, also you need to have LClist loaded in. #no time delay i.e. no newell here
#        #OVm 1 parameter I made stuff for ind 1 and 185
#        #IDM 1 parameter I made stuff for 51 and 185
#results = iLC_r # choose the results to look at
#results_nor = iLC_nor  #results with no LC
#
#sim = copy.deepcopy(meas) #initialize simulation
######change stuff here##########
#objfn = platoonobjfn_obj #put in the relevant objective function
#objfn2 = platoonobjfn_obj
#model = IDM_b3
#modeladjsys = IDMadjsys_b3
#modeladj = IDMadj_b3
#infofn = makeleadfolinfo_r3
#args = (True,6)
#args2 = (False,5)
##############################
##########pick the vehicle by changing vehind
##vehind = 1 #this is the first example
#vehind = 185 #this one can be nice
#curplatoon = LClist[vehind]
#print('Results for vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s). Relaxation info is ')
#leadinfo,folinfo,rinfo = infofn(curplatoon,platooninfo,sim)
#print(rinfo)
#print(' with relaxation found '+str(results[0][vehind][1])+' with no relaxation found '+str(results_nor[0][vehind][1]))
#
##t_nstar,t_n, T_nm1, T_n = platooninfo[curplatoon[1]][0:4]
#
#p = results[0][vehind][0]
#obj = objfn(p,model,modeladjsys,modeladj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#rmse = convert_to_rmse(obj,platooninfo,curplatoon)
#print('with relaxation rmse is '+str(rmse))
#
#t_nstar = platooninfo[curplatoon[1]][0]
#LCtimes = []
#y = []
#for i in rinfo[0]:
#    LCtimes.append(i[0])
#    y.append(sim[curplatoon[1]][i[0]-t_nstar,3])
#
#plt.close('all')
#plt.figure(figsize = (14,6))
#plt.subplot(1,2,2)
#plotspeed(meas,sim,platooninfo,curplatoon[1])
#plt.title('IDM with lane changing dynamics (RMSE = '+str(round(rmse,1))+')')
#plt.plot(LCtimes,y,'*')
#plt.ylim(0,35)
#plt.legend(['Measurements', 'Simulation after calibration','Lane Change Time'])
#
#sim[curplatoon[1]] = meas[curplatoon[1]].copy()
#p = results_nor[0][vehind][0]
#leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,sim)
#
#obj2 = objfn2(p,model,modeladjsys,modeladj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args2)
#rmse2 = convert_to_rmse(obj2,platooninfo,curplatoon)
#print('without relaxation rmse is '+str(rmse2))
#
#y2 = []
#for i in LCtimes:
#    y2.append(sim[curplatoon[1]][i-t_nstar,3])
#
#plt.subplot(1,2,1)
#plotspeed(meas,sim,platooninfo,curplatoon[1])
#plt.plot(LCtimes,y2,'*')
#plt.ylim(0,35)
#plt.title('IDM (RMSE = '+str(round(rmse2,1))+')')
#plt.legend(['Measurements', 'Simulation after calibration','Lane Change Time'])
#plt.savefig('eg4.png',dpi=200)

#%% #same thing as above block but this is going to be for newell which has time delay.
#        #i choose vehicles 185 and 164
#        #just plot the spaces. speed is a little wonky because the headway has kinks in it basically
#        #if you really wanted to you could enforce a smoothness condition in the way you choose the gamma constant.
#        #This is straightforward to work out and I took a pic of an example
#results = nLC_2r # choose the results to look at
#results_nor = nLC_nor  #results with no LC
#
#sim = copy.deepcopy(meas) #initialize simulation
######change stuff here##########
#objfn = TTobjfn_obj #put in the relevant objective function
#objfn2 = TTobjfn_obj
#model = IDM_b3
#modeladjsys = IDMadjsys_b3
#modeladj = IDMadj_b3
#infofn = makeleadfolinfo_r3
#args = (True,4,True)
#args2 = (False,2)
##############################
##########pick the vehicle by changing vehind
##vehind = 1 #this is the first example
#vehind = 185 #this one can be nice
#curplatoon = LClist[vehind]
#print('Results for vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s). Relaxation info is ')
#leadinfo,folinfo,rinfo = infofn(curplatoon,platooninfo,sim)
##leadinfo,folinfo,rinfo= makeleadfolinfo(curplatoon,platooninfo,sim) #debug
#print(rinfo)
#print(' with relaxation found '+str(results[0][vehind][1])+' with no relaxation found '+str(results_nor[0][vehind][1]))
#
##t_nstar,t_n, T_nm1, T_n = platooninfo[curplatoon[1]][0:4]
#
#p = results[0][vehind][0]
#obj = objfn(p,model,modeladjsys,modeladj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
##obj = objfn(p[0:2],model,modeladjsys,modeladj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,False,2) #debug
#rmse = convert_to_rmse(obj,platooninfo,curplatoon,dim=1,delay = p[0])
#print('with relaxation rmse is '+str(rmse))
#
#re_diff(sim,platooninfo,curplatoon,delay=p[0])
#
#t_nstar = platooninfo[curplatoon[1]][0]
#LCtimes = []
#y = []
#delay = math.floor(p[0]/.1) #need this for newell
#for i in rinfo[0]:
#    LCtimes.append(i[0]+delay-1) #so you need -1 because rinfo is the time the vehicle changes, we want to mark the last time you have the current leader
#    y.append(sim[curplatoon[1]][i[0]+delay-1-t_nstar,2]) #this might give you out of bounds sometimes so be careful
#
#plt.close('all')
#plt.figure(figsize = (14,6))
#plt.subplot(1,2,2)
#plotdist(meas,sim,platooninfo,curplatoon[1],delay=p[0])
#plt.title('Newell with lane changing dynamics (RMSE = '+str(round(rmse,1))+')')
#plt.plot(LCtimes,y,'*')
##plt.ylim(330,1360)
##plt.xlim(1850,2231)
##plt.ylim(0,35)
#plt.legend(['Measurements', 'Simulation after calibration','Lane Change Time'])
#
#######
##plt.subplot(1,2,1)
##plotdist(meas,sim,platooninfo,curplatoon[1],delay=p[0])
#
#sim[curplatoon[1]] = meas[curplatoon[1]].copy()
#p = results_nor[0][vehind][0]
#leadinfo,folinfo,rinfo2 = makeleadfolinfo(curplatoon,platooninfo,sim)
#
#obj2 = objfn2(p,model,modeladjsys,modeladj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo2,*args2)
#rmse2 = convert_to_rmse(obj2,platooninfo,curplatoon,dim=1,delay = p[0])
#print('without relaxation rmse is '+str(rmse2))
#
#re_diff(sim,platooninfo,curplatoon,delay=p[0])
#
#LCtimes = []
#y = []
#delay = math.floor(p[0]/.1) #need this for newell
#for i in rinfo[0]:
#    LCtimes.append(i[0]+delay-1)#so you need -1 because rinfo is the time the vehicle changes, we want to mark the last time you have the current leader
#    y.append(sim[curplatoon[1]][i[0]+delay-1-t_nstar,2]) #this might give you out of bounds sometimes so be careful
#
#plt.subplot(1,2,1)
#plotdist(meas,sim,platooninfo,curplatoon[1],delay=p[0])
#plt.plot(LCtimes,y,'*')
##plt.ylim(330,1360)
##plt.xlim(1850,2231)
##plt.ylim(0,35)
#plt.title('Newell (RMSE = '+str(round(rmse2,1))+')')
#plt.legend(['Measurements', 'Simulation after calibration','Lane Change Time'])
#plt.savefig('eg5.png',dpi=200)

#%% show off the smoothness condition for newell

#        #i choose vehicles 185 and 164
#        #just plot the spaces. speed is a little wonky because the headway has kinks in it basically
#        #if you really wanted to you could enforce a smoothness condition in the way you choose the gamma constant.
#        #This is straightforward to work out and I took a pic of an example
#results = nLC_2r # choose the results to look at
#results_nor = nLC_nor  #results with no LC
#
#sim = copy.deepcopy(meas) #initialize simulation
######change stuff here##########
#
#infofn = makeleadfolinfo_r3
#p = [1.5,60,5,5]
#mybounds = [(0,5),(0,200),(.1,75),(.1,75)]
##############################
##########pick the vehicle by changing vehind
##vehind = 1 #this is the first example
#vehind = 185 #this one can be nice
#curplatoon = LClist[vehind]
#print('Results for vehicle '+str(curplatoon[1])+' which has '+str(len(platooninfo[curplatoon[1]][4]))+' different leader(s). Relaxation info is ')
#leadinfo,folinfo,rinfo = infofn(curplatoon,platooninfo,sim)
##leadinfo,folinfo,rinfo= makeleadfolinfo(curplatoon,platooninfo,sim) #debug
#print(rinfo)
#print(' with relaxation found '+str(results[0][vehind][1])+' with no relaxation found '+str(results_nor[0][vehind][1]))
#
##t_nstar,t_n, T_nm1, T_n = platooninfo[curplatoon[1]][0:4]
#
#bfgs = sc.fmin_l_bfgs_b(TTobjfn_obj,p,TTobjfn_fder,(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4,True,True),0,mybounds)
##obj = objfn(p[0:2],model,modeladjsys,modeladj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,False,2) #debug
#rmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon,dim=1,delay = bfgs[0][0])
#print('with relaxation rmse is '+str(rmse))
#
#re_diff(sim,platooninfo,curplatoon,delay=bfgs[0][0])
#
#t_nstar = platooninfo[curplatoon[1]][0]
#LCtimes = []
#y = []
#delay = math.floor(bfgs[0][0]/.1) #need this for newell
#for i in rinfo[0]:
#    LCtimes.append(i[0]+delay-1) #so you need -1 because rinfo is the time the vehicle changes, we want to mark the last time you have the current leader
#    y.append(sim[curplatoon[1]][i[0]+delay-1-t_nstar,3]) #this might give you out of bounds sometimes so be careful
#
#plt.close('all')
#plt.figure(figsize = (14,6))
#plt.subplot(1,2,2)
#plotspeed(meas,sim,platooninfo,curplatoon[1],delay=bfgs[0][0])
#plt.title('Enforcing smoothness condition (RMSE = '+str(round(rmse,1))+')')
#plt.plot(LCtimes,y,'*')
#plt.ylim(15,34)
#plt.xlim(1970,2070)
##plt.ylim(0,35)
#plt.legend(['Measurements', 'Simulation after calibration','Lane Change Time'])
#
#######
##plt.subplot(1,2,1)
##plotdist(meas,sim,platooninfo,curplatoon[1],delay=p[0])
#
#sim[curplatoon[1]] = meas[curplatoon[1]].copy()
#leadinfo,folinfo,rinfo2 = makeleadfolinfo(curplatoon,platooninfo,sim)
#leadinfo,folinfo,rinfo = infofn(curplatoon,platooninfo,sim)
#bfgs2 = sc.fmin_l_bfgs_b(TTobjfn_obj,p,TTobjfn_fder,(None, None, None, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True,4,True),0,mybounds)
#rmse2 = convert_to_rmse(bfgs2[1],platooninfo,curplatoon,dim=1,delay = bfgs2[0][0])
#print('without relaxation rmse is '+str(rmse2))
#
#re_diff(sim,platooninfo,curplatoon,delay=bfgs2[0][0])
#
#LCtimes = []
#y = []
#delay = math.floor(bfgs2[0][0]/.1) #need this for newell
#for i in rinfo[0]:
#    LCtimes.append(i[0]+delay-1)#so you need -1 because rinfo is the time the vehicle changes, we want to mark the last time you have the current leader
#    y.append(sim[curplatoon[1]][i[0]+delay-1-t_nstar,3]) #this might give you out of bounds sometimes so be careful
#
#plt.subplot(1,2,1)
#plotspeed(meas,sim,platooninfo,curplatoon[1],delay=bfgs2[0][0])
#plt.plot(LCtimes,y,'*')
#plt.ylim(15,34)
#plt.xlim(1970,2070)
##plt.ylim(0,35)
#plt.title('Original choice of relaxation constant (RMSE = '+str(round(rmse2,1))+')')
#plt.legend(['Measurements', 'Simulation after calibration','Lane Change Time'])
#plt.savefig('smooth.png',dpi=200)
