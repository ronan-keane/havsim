# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:02:13 2018

@author: rlk268

shows why the relaxation phenomenon is important
needs meas and platooninfo defined as global variables

first two sections are for the first two tables in relax paper. 
Then the bottom has stuff for the other tables. Note that this is pretty disorganized unfortunately. 
Meant mainly for the sake of reproducability of the results in the paper, and not really meant to be reused in the future. 
"""
#this first block is for the initial tables motivating the relaxation phenomenon
##so one way you could make this better is by considering different vehicles involved in the lane change differently. i.e. categorize different vehicles into different sets of statistics. 
##also could do something where you look at the avg speed/headway before and after averaged instead of just a single point. 
##this is beyond the scope of the current study though to be honest. 
##basically in the future you would want to show that anything you consider (treating different vehicles different, having anticipation, post-relaxation) are things you can observe in the data. 
#
#vehlist = meas.keys() #vehicle list 
#dataind = [2,3,4,5,6,7,8] 
#
#interval = 30 
#
#sprev = [] #holds all the previous headways
#alls = [] #holds all the next headways 
#allsmalls = [] #holds all the smaller headways
#allbigs = [] #holds all the bigger headways
#
#vprev = [] #same thing as above but for velocity
#allv = []
#allsmallv = [] #this is for when the headway gets smaller
#allbigv = [] #this is for when the headway gets bigger (cuz sometimes the LC results in vehicles having a larger headway)
#
#lanechange = [] #list of vehicle, time of change
#
#for i in vehlist: 
#    curveh = meas[i]
#    t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
#    curveh = curveh[t_n-t_nstar:T_nm1-t_nstar+1,:] # only get simulation 
#    prevlead = curveh[0,dataind[2]]
#    for j in range(len(curveh)):
#        curlead = curveh[j,dataind[2]]
#        if curlead != prevlead: #if leader changes
#            
##            time = t_n+j #this is the current frameID
#            
#            curlowint = interval
#            curhighint = interval
#            lowintcap = j #number of intervals we can have on lower side
#            highintcap = T_nm1-t_n-j+1 #number of intervals we can have on upper side
#            
#            if lowintcap < curlowint: #in case we can't have the desired number of intervals
#                curlowint = lowintcap
#            if highintcap < curhighint: #in case we can't have the desired number of intervals
#                curhighint = highintcap
#                
#            #need to add something so we don't get multiple lane changes in this part
#            prevmeas = curveh[j-curlowint:j,:] #all the previous measurements (before the LC)
#            test = np.unique(prevmeas[:,dataind[2]])
#            if len(test)>1:
#                prevmeas = prevmeas[prevmeas[:,dataind[2]]==prevlead]
#                curlowint = len(prevmeas)
#            postmeas = curveh[j:j+curhighint,:] #all the pmeasurements after LC 
#            test = np.unique(postmeas[:,dataind[2]]==curlead)
#            if len(test)>1:
#                postmeas = postmeas[postmeas[:,dataind[2]]==curlead]
#                curhighint = len(postmeas)
#            
#            prevlead = np.zeros(curlowint)
#            prevleadlen = np.zeros(curlowint)
#            postlead = np.zeros(curhighint)
#            postleadlen = np.zeros(curhighint)
#            
#            for k in range(len(prevmeas)):
#                leadid = prevmeas[k,dataind[2]] #leadid
#                time = int(prevmeas[k,1]) #time
#                leadt_nstar = platooninfo[leadid][0]
#                leadmeas = meas[leadid][time-leadt_nstar,:]
#                prevlead[k] = leadmeas[dataind[0]]
#                prevleadlen[k] = leadmeas[dataind[4]]
#                
#            for k in range(len(postmeas)):
#                leadid = postmeas[k,dataind[2]] #leadid
#                time = int(postmeas[k,1]) #time
#                leadt_nstar = platooninfo[leadid][0]
#                leadmeas = meas[leadid][time-leadt_nstar,:]
#                postlead[k] = leadmeas[dataind[0]]
#                postleadlen[k] = leadmeas[dataind[4]]
#                
#                
#            curvprev = prevmeas[:,dataind[1]]
#            curv = postmeas[:,dataind[1]]
#            
#            cursprev = prevlead - prevmeas[:,dataind[0]] - prevleadlen
#            curs = postlead - postmeas[:,dataind[0]] - postleadlen
#            
#            sprev.append(cursprev)
#            alls.append(curs)
#            vprev.append(curvprev)
#            allv.append(curv)
#            
#            if cursprev[-1] > curs[0]: #if the headway decreases after the lane change
#                allsmalls.append(curs)
#                allsmallv.append(curv)
#                allbigs.append([])
#                allbigv.append([])
#            else: 
#                allbigs.append(curs)
#                allbigv.append(curv)
#                allsmalls.append([])
#                allsmallv.append([])
#        prevlead = curlead #update prev lead. this is with the forloop in j not with the if statement
#        
#        
##%% after running the above section you can plot the results and compute some other stuff
#count = 0
#avgbefore = []
#avgafter = []
#changebefore = []
#changeafter = []
#
#avgbeforev = []
#avgafterv = []
#changebeforev = []
#changeafterv = []
#
#avgbefore1 = []
#avgafter1 = []
#changebefore1 = []
#changeafter1 = []
#
#avgbeforev1 = []
#avgafterv1 = []
#changebeforev1 = []
#changeafterv1 = []
#
##you need to figure out why some of the sprev/ alls entries are messed up. Pretty sure it's because we are getting multiple lane changes in a short amount of time. 
##fix that. 
#for i in range(len(allsmalls)):
#    if len(allsmalls[i]) == 0:
#        continue
#    count += 1
#    avgbefore.append(np.mean(sprev[i])) #append the average of previous headway
#    avgafter.append(np.mean(allsmalls[i]))
#    changebefore.append(sprev[i][-1]-sprev[i][0]) #this is how much the headway changes right before
#    changeafter.append(allsmalls[i][-1]-allsmalls[i][0]) #this is how much the headway changes right after
#    
#    avgbeforev.append(np.mean(vprev[i])) #append the average of previous headway
#    avgafterv.append(np.mean(allsmallv[i]))
#    changebeforev.append(vprev[i][-1]-vprev[i][0]) #this is how much the headway changes right before
#    changeafterv.append(allsmallv[i][-1]-allsmallv[i][0]) #this is how much the headway changes right after
#            
#for i in range(len(allbigs)):
#    if len(allbigs[i]) == 0:
#        continue
#    count += 1
#    avgbefore1.append(np.mean(sprev[i])) #append the average of previous headway
#    avgafter1.append(np.mean(allbigs[i]))
#    changebefore1.append(sprev[i][-1]-sprev[i][0]) #this is how much the headway changes right before
#    changeafter1.append(allbigs[i][-1]-allbigs[i][0]) #this is how much the headway changes right after
#    
#    avgbeforev1.append(np.mean(vprev[i])) #append the average of previous headway
#    avgafterv1.append(np.mean(allbigv[i]))
#    changebeforev1.append(vprev[i][-1]-vprev[i][0]) #this is how much the headway changes right before
#    changeafterv1.append(allbigv[i][-1]-allbigv[i][0]) #this is how much the headway changes right after                
                


#%% random testing I think
#test the strategy for relaxation phenomenon 
#possible test vehicles: 1321.0, 1330, 1352, 1375,1380,1385,1391,1395
#for i in platooninfo.keys():
#    if len(platooninfo[i][4])>2:
#        print(i)
#995's last lane change at 3158 shows a good example of the lane change causing problems. 
#mergelist = []
#for i in meas.keys():
#    if 7 in meas[i][:,7] and platooninfo[i][0]!=platooninfo[i][1]: #if vehicle is on the on-ramp
#        mergelist.append(i)
#        
#mergelist2 = []
#for i in meas.keys():
#    t_nstar, t_n = platooninfo[i][0:2]
#    if 7 in meas[i][:,7] and platooninfo[i][0]!=platooninfo[i][1] and len(np.unique(meas[i][:t_n-t_nstar,4]))>1: #if vehicle is on the on-ramp
#        mergelist2.append(i)
#        
#mergelist3 =[]
#for i in meas.keys(): 
#    t_nstar, t_n = platooninfo[i][0:2]
#    if 7 in meas[i][:,7] and platooninfo[i][0]!=platooninfo[i][1] and len(np.unique(meas[i][:t_n-t_nstar,4]))==1:
#        mergelist3.append(i)
    
    #%% random notes
#LC_nor2 #baseline for lane changing noLC2 #no lane changing. 

#LC_posr2, lC_negr2, LC_r2, LC_2r, #these are the results for OVM  #corresponds to makeleadfolinfo_r, makeleadfolinfo_r2, makeleadfolinfo_r3, makeleadfolinfo_r3
    #but the last one needs the objective function for 2 parameter relaxation 
    
#then theres the merging results as well 
    
 #%%%   #this is for making tables for main results
#from calibration import * 
#
def LC_rmse(meas,sim,curplatoon,t_nstar, T_nm1,time,ntime,interval,h=.1):
    T_nm1 = min(T_nm1,ntime) #T_nm1 we are treating as the cutoff, and this will be either the next lane change time, or the end of simulation; whichever comes first.
    if time+interval-1>T_nm1:
        interval = T_nm1-time+1
    
    loss = sim[curplatoon[1]][time-t_nstar:time-t_nstar+interval,2] - meas[curplatoon[1]][time-t_nstar:time-t_nstar+interval,2]
    loss = np.square(loss)
#    loss = np.sum(loss)
    
    rmse = np.mean(loss)**.5
    
    
    return rmse
    

def results_helper(meas,platooninfo,platoonlist,makeleadfolinfo,results, platoonobjfn_obj, model, *args, interval = 100):
    #meas - measurments
    #platooninfo - platooninfo corresponding to measurements 
    #platoonlist - vehicles we want to consider
    #makeleadfolinfo - input the function we should use based on the strategy 
    #result - optimization result for everything in platoonlist
    #platoonobjfn_obj - the function to evaluate the objective
    #model - the model we are using 
    # *args - any extra arguments we need to pass into platoonobjfn_obj
    
    overall = np.mean(results[2]) #overall rmse
    
    #want to categorize lane changes into these 4 situations and get the rmse around the lane change only for that situation 
    sit1 = [] #follower initiates LC; gamma is positive
    sit2 = [] #follower initiates LC; gamma is negative
    sit3 = [] #current leader initiates LC (gamma is negative)
    sit4 = [] #new leader initiates LC (gamma is positive)
    
    sim = copy.deepcopy(meas) #for current strategy we always just calibrate to the measurements 
    for i in range(len(platoonlist)):  #iterate over vehicles 
        curplatoon = platoonlist[i] 
        t_nstar, t_n, T_nm1,T_n = platooninfo[curplatoon[1]][0:4]
        leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon, platooninfo,sim)
        unused,unused, rinfo2 = makeleadfolinfo_r3(curplatoon,platooninfo,sim)
        curp = results[0][i][0] #current p 
        
        obj = platoonobjfn_obj(curp,model, None, None, meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo, *args)
        
        if obj != results[0][i][1]: 
#            print('Warning! Check that the below numbers are equivalent to the necessary precision!') 
#            print(str(obj)+' '+str(results[0][i][1]))
            #this is a good sanity check. Also note that when BFGS terminates because of a line search failure, these two numbers will often be 
            #very close but not equivalent. This is a rather obnoxious property of the implementation which doesn't seem intentional but perhaps it is. 
        
        #for each lane change we will look at the time up to interval, T_nm1, or ntime. ntime is the time of the next lane change. This prevents us 
        #from double counting lane changes which are close to each other. 
        
        timelist = []
        for j in rinfo2[0]:
            timelist.append(j[0])
        timelist.append(T_nm1)
        
        for z in range(len(rinfo2[0])):
            j = rinfo2[0][z]
            time = j[0]
            ntime = timelist[z+1]
            
            if sim[curplatoon[1]][time-t_nstar,7] != sim[curplatoon[1]][time-t_nstar-1,7]:
                if j[1] > 0 :
                    currmse = LC_rmse(meas,sim,curplatoon,t_nstar,T_nm1,time,ntime,interval)
                    sit1.append(currmse)
                else: 
                    currmse = LC_rmse(meas,sim,curplatoon,t_nstar,T_nm1,time,ntime,interval)
                    sit2.append(currmse)
            elif j[1] > 0:
                currmse = LC_rmse(meas,sim,curplatoon,t_nstar,T_nm1,time,ntime,interval)
                sit4.append(currmse)
            else: 
                currmse = LC_rmse(meas,sim,curplatoon,t_nstar,T_nm1,time,ntime,interval)
                sit3.append(currmse)
#                newlead = sim[curplatoon[1]][time-t_nstar,4]
#                oldlead = sim[curplatoon[1]][time-t_nstar-1,4]
#                newt_nstar = platooninfo[newlead][0]
#                if sim[curplatoon[1]][time-t_nstar,7] != sim[curplatoon[1]][time-t_nstar-1,7]
        
        
        
        sim[curplatoon[1]] = meas[curplatoon[1]].copy() #reset the simulation for the next vehicle 
    
    return overall, sit1, sit2, sit3, sit4

LClist = []
for i in meas.keys():
    if len(platooninfo[i][4])>1:
        LClist.append([[],i])
#need to pass in different things to look at different model results
#out = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r,LC_posr2,platoonobjfn_obj,OVM, True,6)  
#
#out2 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r2,LC_negr2,platoonobjfn_obj,OVM, True,6)
#
#out3 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r3,LC_r2,platoonobjfn_obj,OVM, True,6)
#
#out4 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r3,LC_2r,platoonobjfn_obj2,OVM, True,7)
#
#out5 = results_helper(meas,platooninfo,LClist,makeleadfolinfo,LC_nor2,platoonobjfn_obj,OVM)
        
#out = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r,iLC_posr,platoonobjfn_obj,IDM_b3, True,6)  
#
#out2 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r2,iLC_negr,platoonobjfn_obj,IDM_b3, True,6)
#
#out3 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r3,iLC_r,platoonobjfn_obj,IDM_b3, True,6)
#
#out4 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r3,iLC_2r,platoonobjfn_obj2,IDM_b3, True,7)
#
#out5 = results_helper(meas,platooninfo,LClist,makeleadfolinfo,iLC_nor,platoonobjfn_obj,IDM_b3)

out = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r,nLC_posr,TTobjfn_obj,None, True,3)  

out2 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r2,nLC_negr,TTobjfn_obj,None, True,3)

out3 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r3,nLC_r,TTobjfn_obj,None, True,3)

out4 = results_helper(meas,platooninfo,LClist,makeleadfolinfo_r3,nLC_2r,TTobjfn_obj,None, True,4,True)

out5 = results_helper(meas,platooninfo,LClist,makeleadfolinfo,nLC_nor,TTobjfn_obj,None)


#%%
# look at vehicles which experience both positive and negative lane changes to better illustrate benefit of 2 parameter relax 

LClist = []
for i in meas.keys():
    if len(platooninfo[i][4])>1:
        LClist.append([[],i])
        
LClist2 = [] #true if entry in LClist has 2 or more lane changes, with both positive and negative lane changes; false otherwise
for i in LClist:
    unused,unused,rinfo = makeleadfolinfo_r3(i,platooninfo,meas)
    pos = False
    neg = False
    for j in rinfo[0]:
        if j[1] >0: 
            pos = True
        if j[1] < 0:
            neg = True
    if len(rinfo[0])> 1 and pos and neg: #2 or more lane changes, also has both negative and positive lane change
        LClist2.append(True)
    else:
        LClist2.append(False)
        
rmse2 = []
#results = [LC_posr2, LC_negr2, LC_r2, LC_2r, LC_nor2] #change this line to look at different model results 
#results = [iLC_posr, iLC_negr, iLC_r, iLC_2r, iLC_nor] #change this line to look at different model results 
results = [nLC_posr, nLC_negr, nLC_r, nLC_2r, nLC_nor] #change this line to look at different model results 
for i in results: 
    temp = []
    for j in range(len(LClist2)): #all vehicles that experiences lane changes 
        if LClist2[j]:#if they experience both positive and negative lane changes 
            temp.append(i[2][j]) #append to temp
    rmse2.append(np.mean(temp)) #average rmse only for those vehicles 


        
#%% other thing we want to do is look at the no lane changing results and filter out everything in the HOV lane cuz that lane skews the results. 
#note that some noLC results are in postercontent
#one rather odd thing is that for newell the HOV lane actually has higher RMSE? Not sure why this is, you'd expect it to be the reverse. 
noLClist = []
for i in meas.keys():
    if len(platooninfo[i][4])==1:
        noLClist.append([[],i])
        
noLClist2 = [] #True if vehicle is NOT in HOV lane (HOV vehicles are quite different since there isnt much congestion in there )
for i in noLClist:
    t_nstar,t_n,T_nm1,T_n = platooninfo[i[1]][0:4]
    lanelist = np.unique(meas[i[1]][t_n-t_nstar:T_nm1-t_nstar,7])
    if 1 in lanelist: #change this and you can look at different lanes 
        noLClist2.append(False)
    else: 
        noLClist2.append(True)
        
noLC_rmse = []
for i in range(len(noLClist)):
    if noLClist2[i]: #have not or no not depending on what you want 
        noLC_rmse.append(nnoLC[2][i]) #change this line to look at different model results 
print(np.mean(noLC_rmse))
print(len(noLC_rmse))

#%% #look at average relaxation parameter values 
#you should check that the relaxation is actually used for positive/negative relax otherwise you will get thrown off due to initial guesses
#no function here; need to manually choose platoonlist, results, the makeleadfolinfo function, and what gamma you want to check for. 
platoonlist = LClist
results = nLC_posr
para = []
for i in range(len(results[2])):
    unused, unused, rinfo = makeleadfolinfo_r(platoonlist[i],platooninfo,meas) #use the appropriate rinfo function
    for j in rinfo[0]: 
        if j[1] > 0: #check for positive or negative relax
            para.append(results[0][i][0][-1])
            break
print(np.mean(para))
#%%#this section of the code you need to run to get the neccesary lists to analyze the merger results
#from calibration import * 
#
##this section of the code you need to run to get the neccesary lists to analyze the merger results 
#sim = copy.deepcopy(meas)
#mergelist = []
#merge_from_lane = 7 
#merge_lane = 6
#for i in meas.keys():
#    curveh = i
#    t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
#    lanelist = np.unique(sim[curveh][:t_n-t_nstar,7])
#    if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][t_n-t_nstar,7]==merge_lane:
#        mergelist.append([[],i])
#        
#        
#mergeLClist = [] #this is going to be all vehicles that merge according to the normal LC rule, but we don't apply the LC there. 
#    #basically this will give a baseline for having no merging rule for vehicles 
#for i in meas.keys():
#    unused, unused, rinfo = makeleadfolinfo_r3([[],i],platooninfo,meas)
#    unused,unused,rinfo2 = makeleadfolinfo_r6([[],i],platooninfo,meas)
#    if len(rinfo[0])>0:
#        if len(rinfo2[0])==0:
#            mergeLClist.append([[],i])
#        elif rinfo[0][0] != rinfo2[0][0]:
#            mergeLClist.append([[],i])
#            
##merge_nor, merge_r, merge_2r,mergeLC_r, mergeLC_2r, LC_r2, LC_2r #all results that need to be loaded to do the analysis 
#            
#noLClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])==1:
#        noLClist.append([[],i])
#        
#LClist = []
#for i in meas.keys():
#    if len(platooninfo[i][4])>1:
#        LClist.append([[],i])
#
#mergelist2 = [] #True if a vehicle in LClist has one of the lane changes as a merger #this is the bool for LClist
#for i in LClist: 
#    if i in mergeLClist: 
#        mergelist2.append(True)
#    else:
#        mergelist2.append(False)
#        
#mergelistbool = np.ones(len(mergelist),dtype='Bool')
  #%%      
##input is one of the result outputs, list of platoons the results are for, and a list of booleans of the same length as the list of platoons
## If the boolean is true, for the corresponding result index, 
#  #we will compute the simulation corresponding to the calibrated parameters, identify the merger in the simulation, and compute the 
#  #rmse for the merger. we assume there is only a single merger. 
#  #in opt 1, we are considering vehicles from the mergelist (whose simulation starts when they merge)
#  #in opt 2, we are considering vehicles from the mergeLClist (whose simulations starts before they merge)
#  
#from calibration import * 
#  
#def LC_rmse(meas,sim,curplatoon,t_nstar, T_nm1,time,ntime,interval,h=.1):
#    T_nm1 = min(T_nm1,ntime) #T_nm1 we are treating as the cutoff, and this will be either the next lane change time, or the end of simulation; whichever comes first.
#    if time+interval-1>T_nm1:
#        interval = T_nm1-time+1
#    
#    loss = sim[curplatoon[1]][time-t_nstar:time-t_nstar+interval,2] - meas[curplatoon[1]][time-t_nstar:time-t_nstar+interval,2]
#    loss = np.square(loss)
##    loss = np.sum(loss)
#    
#    rmse = np.mean(loss)**.5
#    
#    
#    return rmse
#  
#def merge_helper(meas,platooninfo,platoonlist, resultsbool,makeleadfolinfo,results, platoonobjfn_obj, model, *args, opt = 1, interval = 100):
#    sim = copy.deepcopy(meas)
#    
#    sit1 = []
#    sit2 = []
#    for i in range(len(resultsbool)):
#        if resultsbool[i]:
#            curp = results[0][i][0]
#            curplatoon = platoonlist[i]
#            leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon, platooninfo, meas)
#            
#            t_nstar,t_n, T_nm1, T_n = platooninfo[curplatoon[1]][0:4]
#            
#            obj = platoonobjfn_obj(curp,model,None,None,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args) #do the simulation 
#            
#            if obj != results[0][i][1]: #sanity check; if these aren't equal something was passed incorrectly. 
#                print('something is wrong')
#            
#            ########forget this#########
##            timelist = []
##            curmergetime = float('inf')
##            for j in range(t_n, T_nm1+1): #the times we are going to check for a merger
##                if j ==0 and t_n-t_nstar==0: 
##                    continue #avoid indexing error
##                prevlane = sim[curplatoon[1]][j-t_nstar-1,7]
##                curlane = sim[curplatoon[1]][j-t_nstar,7]
##                if prevlane==7 and curlane ==6: #definition of merger
##                    curmergetime = j #time when the merger occurs for the current vehicle 
##                    break #if you wanted to you could check for multiple mergers but we know each vehicle is only going to merge once in this case 
##            if curmergetime == float('inf'):
##                print('no merger detected. something is wrong') #sanity check
#            ##############################
#            if opt==1: 
#                unused,unused,rinfo2 =  makeleadfolinfo_r4(curplatoon,platooninfo,meas)
#                mergeinfo = rinfo2[0][0] #by definition this is the rinfo for the merger in option 1
#                
#                currmse = LC_rmse(meas,sim,curplatoon,t_nstar,T_nm1,mergeinfo[0],T_nm1,interval) #compute RMSE around the merger 
#            elif opt==2: 
#                unused,unused,rinfo2 = makeleadfolinfo_r3(curplatoon,platooninfo,meas)
#                unused,unused,rinfo3 = makeleadfolinfo_r6(curplatoon,platooninfo,meas)
#                
#                for j in rinfo2[0]:
#                    if j not in rinfo3[0]:
#                        mergeinfo = j
#                
#                currmse = LC_rmse(meas,sim,curplatoon,t_nstar,T_nm1,mergeinfo[0],T_nm1,interval)
#                #here the mergeinfo is going to be given by the entry in rinfo2[0] that is not present in rinfo3[0]. 
#                
#            
#            if mergeinfo[1] > 0: #if relaxation constant is positive assign it to sit 1
#                sit1.append(currmse)
#            else: 
#                sit2.append(currmse)
#                
#            sim[curplatoon[1]] = meas[curplatoon[1]].copy() #reset simulation for next vehicle 
#                
#    return sit1, sit2
## 1 p relax
#out01 = merge_helper(meas,platooninfo,mergelist,mergelistbool,makeleadfolinfo_r4,merge_r,platoonobjfn_obj,OVM,True,6,opt=1)
#                
#out02 = merge_helper(meas,platooninfo,LClist,mergelist2,makeleadfolinfo_r3,LC_r2,platoonobjfn_obj,OVM,True,6,opt=2)
#
##results for 2 p relax 
#out03 = merge_helper(meas,platooninfo,mergelist,mergelistbool,makeleadfolinfo_r4,merge_2r,platoonobjfn_obj2,OVM,True,7,opt=1)
#                
#out04 = merge_helper(meas,platooninfo,LClist,mergelist2,makeleadfolinfo_r3,LC_2r,platoonobjfn_obj2,OVM,True,7,opt=2)
#
##results for no relax 
#out05 = merge_helper(meas,platooninfo,mergelist,mergelistbool,makeleadfolinfo,merge_nor,platoonobjfn_obj,OVM,False,5,opt=1)
#               
#out06 = merge_helper(meas,platooninfo,LClist,mergelist2,makeleadfolinfo,LC_nor2,platoonobjfn_obj,OVM,False,5,opt=2)
#    
##see paperplots for interpretation on what all the output means and the different categories. 

#%% last section will make simple table to show accuracy for different number of lane changes. 
#LC_r2, LC_2r, LC_nor2
results = nLC_2r

LClist = []
for i in meas.keys():
    if len(platooninfo[i][4])>1: #at least 2 leaders = at least 1 lane change
        LClist.append([[],i])
        
indlist = []
for i in meas.keys():
    temp = len(platooninfo[i][4]) 
    if temp ==0 or temp==1: #at least 2 leaders = at least 1 lane change
        continue
    else: 
        leadinfo,folinfo,rinfo = makeleadfolinfo_r3([[],i],platooninfo,meas)
        temp = len(rinfo[0]) #number of lane changes
        indlist.append(temp-1) #1 lane changes gets mapped to 0 index
out = [[] for i in range(4)] #categorize into 1, 2, 3, and 4 or more lane changes
for i in range(len(results[0])):
    temp = indlist[i]
    if temp >3:
        temp = 3
    out[temp].append(results[2][i])

print(np.mean(out[0]))
print(np.mean(out[1]))
print(np.mean(out[2]))
print(np.mean(out[3]))
  

