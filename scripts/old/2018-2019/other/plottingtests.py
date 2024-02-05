
"""
@author: rlk268@cornell.edu
"""

"""

TO DO
some features of selectoscillation missing; most notably removing vehicles. The way its initialized is still buggy and there is some sort of problem there. 

Upon inspection of the sortveh3 function, I've determined that the issue is basically due to when we are adding all the newfolveh, they have different leaders. Then we are comparing them against 
different lead vehicles, and this can be a problem. 
Solution would be to, at the end of each iteration, form a sort of "convex hull" that is basically the last trajectory pieced together with the other things. 

Other problem is the code to assign leftovers has some issues with logic, seeing as the 5 and 7 are getting put to the very end, when this shouldn't happen. 
"""

#speeds,X,Y,meanspeeds = meanspeedplot(data,100,20,lane=2,use_avg='harm')
#main call signature 
times,x,lane,veh = selectoscillation(data,50,20,lane=3,use_avg='mean') #this function is still buggy I think, something wrong with initialization, sometimes labelling messed up 
    
#times,x,lane,veh = selectoscillation(data1974,50,160,lane=1,use_avg='mean', region_shape=None)

#times,x,lane,veh = selectoscillation(highd,50*2.5,20,lane=2,use_avg='mean', region_shape=None)


#%%
#############
test2 = [[(5562.474476963611, 1476.8050669428), (6311.045414797408, 164.0527611552), (7203.064516129032, 164.0527611552), (6454.493578295235, 1476.8050669428)]] 
#note there used to be a bug with this shape but it should be fixed; see selectvehID comments for more details. 
######################
test3 = [224.0, 194.0, 244.0, 240.0, 249.0, 255.0, 260.0, 267.0, 257.0]
#correct order is [224.0, 194.0, 240.0, 244, 249.0, 255.0, 260.0, 257, 267]
#for testing purposes
#selectvehID(data,times,x,lane,veh,test2,test3) #[378.0, 381.0, 391.0, 397.0, 3360.0, 394.0] #[224.0, 194.0, 244.0, 240.0, 249.0, 255.0, 260.0, 267.0, 257.0]
    
#meas, platooninfo, platoonlist = makeplatoonlist(data,n=5,lane=2,vehs=[582,1146])
#meas, platooninfo, platoonlist = makeplatoonlist(data,n=5,lane=2,vehs=[649,1130])
#problem is because 267 follows 240 only in lane 3, not in lane 2, so when we compute the distance we are getting Nan because we only look at the distance in lane 2. 

################
#wtplot(meas,1013)
    
#this passes successfully 
#test22 = sortveh3(test3,2,meas,platooninfo) #[542,561,565,566] #[378.0, 381.0, 391.0, 397.0, 3360.0,394.0]