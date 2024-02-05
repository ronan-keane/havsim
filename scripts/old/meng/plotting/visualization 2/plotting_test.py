# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:14:48 2019

@author: rlk268
"""
#from calibration import *

#%% for reproducing data 
#get data for testing purposes 
# meas, platooninfo, platoonlist = makeplatoonlist(data,n=3,lane=2,vehs=[649,1130])
#initialize simulation
# vehlist = []
# [vehlist.extend(i[1:]) for i in platoonlist]
#sim = copy.deepcopy(meas)
#
##calibrate vehicles in platoonlist
#plist = [[40,1,1,3,10,25],[60,1,1,3,10,5], [80,1,15,1,1,5]]
#mybounds = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
#test = calibrate_tnc2(plist,mybounds,meas,platooninfo,platoonlist,makeleadfolinfo_r3,platoonobjfn_objder,None,IDM_b3,IDMadjsys_b3,IDMadj_b3,
#                      *(True,6),cutoff=4,cutoff2=5.5,order=1,dim=2,budget=1)
#
##load results into sim
#sim = obj_helper(test[0],IDM_b3,IDMadjsys_b3,IDMadj_b3,meas,sim,platooninfo,platoonlist,makeleadfolinfo_r3,platoonobjfn_obj,(True,6))
##obj = SEobj_pervehicle(meas,sim,platooninfo,platoonlist[0]) #sanity check 
#
# with open('plottingtesting.pkl','wb') as f:
#    pickle.dump([meas,platooninfo,platoonlist,sim],f)
    

#%%
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm 


#with open('/home/rlk268/Downloads/hav-sim-master(1)/hav-sim-master/visualization/plottingtesting.pkl','rb') as f: 
#    meas, platooninfo, platoonlist, sim = pickle.load(f)
#
#vehlist = []
#[vehlist.extend(i[1:]) for i in platoonlist]
#
#from havsim.plotting import platoonplot, animatevhd_list, animatetraj

"""
TO DO 
color code the trajectories in platoonplot based on the current speed of vehicles
add new keyword argument to control whether or not to color code the trajectories. There should also be a keyword that can set the maximum/minimum 
speeds, and if this keyword is None, the function should find the minimum/maximum speeds 
You should also add a colorbar which shows what speeds the color correspond to. You can use plt.colobar() to do this. 

Matplotlib has some default colormaps (from matplotlib import cm), you should use 
cm.RdYlBu 
as the colormap the function uses.

I also want you to add the feature where you can click on a trajectory to display the vehicle ID. the function optplot already has this feature added. 
To do this, when you plot things pass in keyword 'picker = 5'. e.g. plt.plot([1,2,3],[1,2,3],picker=5)
You will also need to keep reference to the figure you are using (fig = plt.figure()). Then add in an extra line 
fig.canvas.callbacks.connect('pick_event', on_pick) 
What happens is when you click within 5 pixels of an artist in fig, it triggers a 'pick_event' which then calls the function you define called on_pick

Then in on_pick, it should look something like 
def on_pick(event): 
    ax = event.artist.axes #get current axes of the event 
    curind = ax.lines.index(event.artist) #get the index of the artist on the axis which has been triggered 
    
    #recolor the selected artist and all other artists associated with the vehicle ID so you can see what line you clicked on 
    #change the title to say what vehicle you selected. 
    
Note that you will need to keep track of which artists correspond to which vehicles 

%%%%%%%%%%%%%%UPDATED TO DO %%%%%%%%%%%%%%%%%
make sure you have palettable installed

opacity doesn't work with colorcode=True; make it so the trajectories outside of the given lane either have opacity applied (if it's possible to do so), or have those trajectories outside 
the given lane not shown if opacity < 1 and colorcode = True

There is a bug with plotcolorlines where the trajectories are getting cut at weird places, and being colored wrong. It seems like every line given to plotcolorlines is being colorcoded 
based on itself, and when vehicles change lanes the trajectories in different lanes are being treated as new trajecotries, so they don't get connected with the past trajectory, and 
the line is colored wrong as well. 
See the below example: 
	
platoonplot(meas,None,platooninfo,platoon=[[],672], lane=2,opacity =1, colorCode= True) #why is there a weird gap in some of the trajectories? Like this one
plt.figure()
plt.plot(meas[672][:,3])

Shown is the trajectory for a single vehicle, there shouldn't be a gap in it. Also note that the speeds are shown in the second plot, so you can see only the very beginning of the trajectory should be 
red (because its around 20 ft/s) and after that beginning it should be either yellow or green. There shouldn't be a second red colored region. 


Lastly, the speed_limit keyword doesn't seem to be being used at all. Refer to the .pdf for the functionality of that keyword. 


"""
platoonplot(meas,None,platooninfo,platoon=vehlist, lane=2,opacity =.1, colorCode= True) #opacity doesn't work
platoonplot(meas,None,platooninfo,platoon=vehlist, lane=2,opacity =.1, colorCode= False) #opacity works

# platoonplot(meas,None,platooninfo,platoon=[[],672], lane=2,opacity =1, colorCode= True) #why is there a weird gap in some of the trajectories? Like this one
# plt.figure()
# plt.plot(meas[672][:,3]) #you can see there is not supposed to be any gap in speeds, and it doesn't make sense why it is red right after as well.
# plt.show()
#some sort of bug with the plotcolorlines function, or some sort of problem caused by the function chopping up lines when a vehicle changes lanes.

#also there is another bug with how the speed_limit is being calculated. 
#compute what the correct max/min should be. results -> [0, 81.87], that is what should automatically be used if speed_limit is not explicitly passed. 
# mymin = 1e10
# mymax = 0
# for i in vehlist:
#     curmin = min(meas[i][:,3])
#     curmax = max(meas[i][:,3])
#     if mymin > curmin:
#         mymin = curmin
#     if mymax < curmax:
#         mymax = curmax

platoonplot(meas,None,platooninfo,platoon=vehlist, lane=2,opacity =.1, colorCode= True, speed_limit = [10,35]) #can't explicitly set speed_limit to being between (10,35)
#%%

"""
TO DO 
The below function animates a trajectory in the speed-headway plane. I want you to make the following improvements to it - 

-Currently the function accepts my_id as a float which corresponds to the vehicle ID to animate. I want to be able to pass in a list 
of vehicle IDs and have all trajectories animated together. 

-Also make it so you can specify a starting time for the animation, and an ending time for the entire animation. 

e.g. animatevhd(sim,None,platooninfo,[898,905,909,916,920],effective_headway=False,rp=15,show_sim=False,start = 2700, end = 3100)
should animate the vehicles 898, 905, 909, 916, and 920 in the same axis from time = 2700 to time = 3100. 

%%%%%%%%%%%UPDATED TO DO%%%%%%%%%%%%%
swap out black dots with numbers for the vehicle IDs
make it so you can plot just 1 set of data (currently both meas and sim must be plotted)
"""
animatevhd_list(meas,sim,platooninfo,[898, 905, 909, 916, 920],effective_headway=True,rp=15,show_sim=False, usestart = 2900, useend=3000) #show measurements and simulation


#animatevhd_list(meas,sim,platooninfo,[638, 622, 654],effective_headway=True,rp=15,show_sim=False) 
#animatevhd_list(meas,sim,platooninfo,[898, 905, 909],effective_headway=True,rp=15,show_sim=False, start = 2900, end=3000) #show measurements and simulation
#
#animatevhd_list(meas,None,platooninfo,[898, 905, 909],effective_headway=True,rp=15,show_sim=False, start = 2900, end=3000) #show measurements and simulation


#animatevhd(meas,sim,platooninfo,705,effective_headway=True,rp=15,show_sim=False)  #show only measurements, make sure effective_headway is working 

#%%

"""
TO DO 
animatetraj will animate the positions of vehicles, so it is like if you have a bird's eye view on the road looking down at the vehicles. 
Make the following improvements/features: 
    
-The function can get very slow for a very big platoon. For example, if you try to plot  animatetraj(meas,platooninfo) it will be pretty laggy. 
I believe the reason why this is because the way it currently works is using artistanimation instead of funcanimation, 
see https://matplotlib.org/3.1.1/api/animation_api.html
In artistanimation, everytime it updates it is redrawing all the objects. If you used funcanimation you could make it so it just needs to update the positions
of all the current artists shown, and I believe this will make it faster. 

-Currently the vehicles are represented by black dots. I want the vehicles to be a colored rectangle with black numbers inside showing the vehicle ID. See the .pdf


%%%%%%%%UPDATED TO DO%%%%%%%
make it so the usetime keyword works to give the times you want to animate. Usetime should be a list of all frames to animate. 
colorbar is getting cutoff in the figure; make it so you can see the entire colorbar. 
"""
vehlist = []
[vehlist.extend(i[1:]) for i in platoonlist]
animatetraj(meas,platooninfo,vehlist,usetime=list(range(2600,2700))) #the usetime keyword no longer works, fix this

# animatetraj(meas,platooninfo) #plot all vehicles 



#%%
# For Test

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# platoonplot(meas,None,platooninfo,platoon=vehlist, lane=2,opacity =0, colorCode= True) #replace meas with sim you can see what the simulation looks like
# anim = animatevhd_list(meas,sim,platooninfo,[898, 905, 909, 916, 920],effective_headway=False,rp=15,show_sim=True, start=2700, end=3000) #show measurements and simulation
# anim.save("All_vehicles.mp4", writer=writer)


# anim = animatevhd_list(meas,sim,platooninfo,[898, 905, 909, 916, 920],effective_headway=False,rp=15,show_sim=True, start=2700, end=3000) #show measurements and simulation
# platoonplot(meas,None,platooninfo,platoon=vehlist, lane=2,opacity =0, colorCode= False) #replace meas with sim you can see what the simulation looks like
