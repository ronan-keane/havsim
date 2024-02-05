
"""
@author: rlk268@cornell.edu
"""
#meas2 = {}
#platooninfo2 = {}
#count = 0
#for i in meas.keys():
#    meas2[i] = meas[i].copy()
#    platooninfo2[i] = platooninfo[i].copy()
#    count +=1
#    if count > 20: 
#        break
    
#platoonplot(meas2,None,platooninfo2,platoon=[], lane=2,opacity =.1, colorCode= True, speed_limit = [10,35]) 
#%%
    
from havsim.plotting import platoonplot

def plotformat(sim, auxinfo, roadinfo, starttimeind = 0, endtimeind = 3000, density = 2, indlist = [], specialind = 21):
    #get output from simulation into a format we can plot using plotting functions
    
    #starttimeind = first time to be plotted 
    #endtimeind = last time to be plotted 
    
    #density = k plots every kth vehicle, indlist = [keys] plots keys only. 
    #specialind does not do anything
    #it handles the circular road by creating the wrap around as new vehicles. 
    
    L = roadinfo[0]
    platooninfo = {} #need platooninfo 0 - 4 : observation times
    meas = {} #need columns 1, 2,3, 7
    idcount = 0
    speciallist = []
    
    if indlist == []:
        uselist = list(sim.keys())[::density]
    else:
        uselist = indlist
    for i in uselist:
        cur = sim[i]
        
#        if i == specialind:
#            speciallist.append(idcount)
        
        #initialize output for current vehicle
        curtime = 0
        prevx = -1
        tlist = []
        xlist = []
        vlist = []
        platooninfo[idcount] = [curtime, curtime, None, None]
        for counter, j in enumerate(cur): #for each vehicle
            if counter < starttimeind:
                continue 
            
            if j[0] < prevx: #if distance decreases its because we wrapped around - store measurements in new vehicle 
                
                endtime = counter #get end time and update platooninfo
                platooninfo[idcount][2:] = [endtime-1,endtime-1]
                #update meas
                meas[idcount] = np.zeros((endtime-curtime,8))
                meas[idcount][:,1] = tlist
                meas[idcount][:,2] = xlist
                meas[idcount][:,3] = vlist
                #lane just set always to 1
                meas[idcount][:,7] = 1
                
                #reset iteration
                idcount += 1
                curtime = endtime
                prevx = j[0]
                tlist = [curtime]
                xlist = [j[0]]
                vlist = [j[1]]
                platooninfo[idcount] = [curtime, curtime, None, None]
                continue
                
            tlist.append(counter)
            xlist.append(j[0])
            vlist.append(j[1])
            prevx = j[0]
            
            if counter >= endtimeind:
                break
            
        #also need to finish current once for loop ends 
        endtime = counter #get end time and update platooninfo
        platooninfo[idcount][2:] = [endtime,endtime]
        #update meas
        meas[idcount] = np.zeros((endtime-curtime+1,8))
        meas[idcount][:,1] = tlist
        meas[idcount][:,2] = xlist
        meas[idcount][:,3] = vlist
        #lane just set always to 1
        meas[idcount][:,7] = 1
        
        idcount += 1
            

    return meas, platooninfo

bestind = np.arange(1,40,2)
bestind = np.arange(0,41,1)
meas, platooninfo = plotformat(sim,auxinfo,roadinfo, endtimeind = math.inf, indlist = bestind, specialind = 21)

#meas2 = {}
#platooninfo2 = {}
#count = 0
#for i in meas.keys():
#    meas2[i] = meas[i].copy()
#    platooninfo2[i] = platooninfo[i].copy()
#    count +=1
#    if count > 20: 
#        break

platoonplot(meas,None,platooninfo,platoon=[], lane=1, colorcode= True, speed_limit = [0,25]) 
plt.ylim(0,roadinfo[0]) #dunno why ylimit is messed up 