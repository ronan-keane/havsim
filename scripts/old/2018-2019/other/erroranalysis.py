# -*- coding: utf-8 -*-
"""
look at the differences between reextracted (true) data compared to the raw and reconstructed data
These functions are called in the script test.py
erroranalysis %%%%%%%%%%%%%
inputs:
accepts rawdata,truedata, data as inputs (from loadngsim)
opt is a string, either 'raw' or 'true', which controls whether to use raw or reconstructed data
bin variables control size of bins for each category. (units are in feet)
distbin - local y value (portion of road)
lenbin - vehicle length 
speedbin - speed of vehicle 
errorbin - magnitude of error

outputs: out (error for each measurement is sorted into which dist/len/speed/err bin it fits into)
trueunique (list of unique vehicle IDs in true data)
outtraj (dict with vehicle ID as key. value is list of tuples with (frame, err) for specific vehicle ID)

ploterror %%%%%%%%%%%%%%%%%%%%%%%%
inputs: 
out (from erroranalysis)
distbinplot - index of distance bin to plot
lenbinplot - index of length bin to plot
speedbinplot - index of speed bin to plot

output:
plot of error with x axis as bin indices, y axis as frequency (number of observations) for specific dist/len/speed

ploterrortraj %%%%%%%%%%%%%%%%%%%%%
inputs: 
outtraj (from erroranalysis)
vehicle ID (scalar, selected from trueunique)
outputs: 
plot of error over time with x axis as frame id, y axis as magnitude of error (ft)
    
    

#NGSim study area 
#based on the picture from "data analysis" document from cambridge system analytics report...other option is to use the videos but this is more complicated 
#give x and y pixel values for the center for each camera range - these pixel coordinates are from an orthorectified image. so what causes the discrepancy? 
#estimate the local y ranges for each camera shown in brackets (multiply length by 3.42920)
#first try...camera 1 clearly off (end to end)
#camera - (x pixel value, y pixel value) : computed local_y value {rescaled (if applicable)}
#1 - (269, 948) / (245, 800):  -497 / 17.2
#2 - (222, 635): 588.5
#3 - (212, 577): 790.3
#4 - (201, 513): 1013.0
#5 - (191, 445): 1248.7
#6 - (177, 351): 1574.6
#7 - (128,41)

#second try  (6.835726) (brackets to brackets using coifman data...still not quite right)
#1 - (109, 442) / (105, 408): -200 / 34.3
#2 - (95, 326): 599.0 
#3 - (91, 297): 799.1
#4 - (86, 266): 1013.8
#5 - (81, 232): 1248.7
#6 - (73, 185): 1574.6
#7 - (69, 162): 1734.2

#third try  (brackts to brackets corresponding to range of 0-1650) (5.83363) {rescaled to 0-1757.5}
#1 - (109, 442) / (105, 408): 0 / 199.7 {0 / 212.7}
#2 - (95, 326): 681.6 {726}
#3 - (91, 297): 852.4 {907.9}
#4 - (86, 266): 1035.6 {1103.1}
#5 - (81, 232): 1236.1 {1316.6}
#6 - (73, 185): 1514.22 {1612.6}
#7 - (69, 162): 1650 {1757.5}

#other way to get these is to go through video frame by frame, find frames where detected positions swap from camera to camera, then get local y estimate of camera coverage
#note when the detected positions actually switch cameras will depend on the vehicle. time consuming only need to do this if above estimates are not good..third try no curly braces plausible
#remember li was tracking back of vehicles, so third try has close agreement when adding minimum veh length to #5 and maximum veh length to #6

#note: frame ID numbering starts from 1 in the data. x pixel 348 on cam 1 is roughly 0.0. Any vehicle detected further upstream than 348 gets a position of 0. 
#assuming each pixel of video data is .5 feet leads to the following estimates for camera ranges:
#includes reference vehicles and frames used when making measurements. 'Detected at x' means detected positions swaps cameras in the NGSim processed video footage at local y value of x
#camera - (x pixel upstream, x pixel downstream) - (local_y upstream, local_y downstream) (veh_id@frame_id detected at local_y) 
#1 - (348, 828) - (0.0, 240.0)  (476@1765,1766 - 546@1755,1756)
#2 - (0, 932) - (239.0/239.5, 705.0/705.5) (514@1800 detected at 249.5, 519@1801 detected at 252, 476@1803 detected at 259.5)
#3 - (0, 340) - (701.5/702.0/705.5/705.0, 871.5/875.5) (1024@3139 detected at 712, 1029@3143 detected at 718.5,  1045@3171 detected at 719.5, 1056@3174 detected at 716.5)         
#4 - (0, 360) - (869.5, 1049.5) (862@2780 detected at 881.5, 870@2781 detected at 883.5, 903@2893 detected at 885.0)
#5 - (0, 388) - (1047.0/1049.0, 1253.0/1255.0 ) (1347@4239 detected at 1056.0, 1344@4240 detected at 1060.0,  1350@4251 detected at 1063.0)
#6 - (0, 532) - (1242.5, 1508.5) (1497@4843 detected at 1256.0, 1510@4842 detected at 1252.0, 1543@4903 detected at 1265.0) 
#7 - (0, - ) - (1507.0/1507.5, ) (1480@4939 detected at 1519.0, 1500@5020 detected at 1525.5, 1491@5105 detected at 1521.5)

#min/max local y for reconstructed data: (164.04, 1496.27) 
#min/max local y for raw data: (0, 1757.49)
#min/max local y for true data: (1248.7, 1574.6) = 325.9 
"""

import numpy as np
import matplotlib.pyplot as plt

from ..havsim.calibration.algs import makeplatoonlist

def erroranalysis(rawdata, truedata, data, opt, distbin, lenbin, speedbin, errbin):
    
    maxdist = 1750 #maximum local y value
    maxlen = 80 #maximum vehicle length
    maxspeed = 105 #max speed 
    maxerr = 50 #upper bound. if err is greater than max err, it will get put into the last bin
    
    if opt=='raw': #which dataset will we use, row or reconstructed? data stored in different columns depending on source
        mydata = rawdata
        yind = 5
        lenind = 8 
        speedind = 11
        laneind = 13
    else: 
        mydata = data
        yind = 3
        lenind = 6
        speedind = 4
        laneind = 2

    trueunique = np.unique(truedata[:,0]) #vehicle IDs for true measurements 
    #take the maximum value, divide into equally sized bins, rounding the result up 
    #have a 2 at end because error can be positive or negative
    errbins = -(-maxerr // errbin)
    dim = (-(-maxdist // distbin), -(-maxlen // lenbin), -(-maxspeed // speedbin), errbins, 2) #double slash operator denotes mod...can be used to divide and then round up or down
    
    out = np.zeros(dim) #initialize output
    outtraj = {} 
    
    for i in trueunique: 
        trueveh = truedata[truedata[:,0]==i] #frames for vehicle from true data
        dataveh = mydata[mydata[:,0]==i] #frames for vehicle from choosen data source
        frames = trueveh[:,1]
        maxframe = max(dataveh[:,1])
        
        vehlen = trueveh[0,8] #vehicle length 
        errtraj = []
        
        for j in range(len(frames)): #for each frame for specific vehicle
            if frames[j]>maxframe: #prevent out of bounds error when processing reconstructed data
                break #break if reconstructed data does not contain frame number
        
            curtrue = trueveh[j,:]
            curdata = dataveh[dataveh[:,1]==frames[j]]
            
            
            err = curdata[0,yind]-curtrue[5] #error in local y measurement
            dist = curdata[0,yind] #local y value
            speed = curdata[0,speedind] #speed value
            errtraj.append(frames[j])
            errtraj.append(err)
            
            if err< 0 :
                mysign = 1 #need to put 1 if error is negative
                curerrbin = int(-err // errbin) #need negative here as well 
                
            else: 
                mysign = 0 #otherwise just leave it as 0
                curerrbin = int(err // errbin)
                
            curdistbin = int(dist // distbin)
            curlenbin = int(vehlen // lenbin)
            curspeedbin = int(speed // speedbin)
            
            if curerrbin > errbins-1: #if error is too big we put it into the last bin
                    curerrbin = errbins-1
            
            out[curdistbin, curlenbin, curspeedbin, curerrbin, mysign] += 1 #sort error into appropriate bin
            #may want to add additional bin for lane changing...but note there arent many trucks and motorcycles 
        
        
        outtraj[str(i)] = errtraj
    return out,trueunique, outtraj

def ploterror(out, distbinplot=26,lenbinplot = 0, speedbinplot = 3):
    #numbers correspond to which bin the data is from 
    #bin values = binplot*bin through (binplot+1)*bin
    #eg distbin = 10. distbinplot = 30. function plots everything with local y vals between 300 and 309.99
    
    plotdata = np.append(out[distbinplot, lenbinplot, speedbinplot,::-1,1], out[distbinplot, lenbinplot, speedbinplot,:,0], axis = 0)
    dim = len(out[distbinplot, lenbinplot, speedbinplot,::-1,1])
    plotx = np.append(list(range(-dim,0)), list(range(dim)),axis=0)
    
    #note in positive smallest bin, x axis is zero, in negative smallest bin, x axis is -1
    #x axis corresponds to bin number
    plt.plot(plotx,plotdata)
    plt.show()

    return 

def ploterrortraj(outtraj,vehid):
    #plot evolution of error over time for a specific vehicle
    plotdata = outtraj[vehid]
    plt.plot(plotdata[::2],plotdata[1::2])
    
    return

def analyzeerror(data, rawdata, truedata, use_data='raw'):
    #just want to see rmse for rawdata and reconstructed data
    
    #this is going to be RMSE averaged over each vehicle; this is equivalent to vehicles observed shorter times will be weighted more heavily. 
    ans = []
    
    if use_data == 'raw':
        meas1, platooninfo1 = makeplatoonlist(rawdata,1,False)
    else: 
        meas1, platooninfo1 = makeplatoonlist(data,1,False)
        
    meas,platooninfo = makeplatoonlist(truedata,1,False)
    
    for i in meas.keys(): 
        try: 
            meas1[i]
        except KeyError:
            continue #this must be a key associated with a motorcycle (the different data handles the motorcycles differently)
            
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        t_nstar1, t_n1, T_nm11, T_n1 = platooninfo1[i][0:4]
        
        start = max([t_nstar, t_nstar1])
        end = min([T_n, T_n1])
        curerror = np.square(meas[i][start-t_nstar:end-t_nstar+1,2]-meas1[i][start-t_nstar1:end-t_nstar1+1,2])
        ans.append(np.mean(curerror))
        
        
    ans = np.mean(ans)**.5     
    
    return ans


ans = analyzeerror(data,rawdata,truedata,0)

rawans = analyzeerror(data,rawdata,truedata)

#%% call on some of the above functions and make some observations 
###################################################################################
#out,vehIDlist,outtraj = erroranalysis(rawdata, truedata, data, 'raw', 50, 40, 10, 1)
##get error for every measurement, sorted based on local y value, vehicle length, and speed
##dist, length, speed, error

#ploterror(out, 32,0,2)
#plot error for specfic combination 

#ploterrortraj(outtraj,str(vehIDlist[909])) #909 is an example of vehicle with very large error, corresponding to vehicle ID 1463, time 8:05 in camera 6 video
#plot trajectory of error 

###################################################################################
#any combinations we find that don't look normal
#binsize = 'raw',50,15,10,3 
#plotsize = 30,0,3 
#plotsize = 27,4,3

#binsize = 'raw',50,15,10,1
#plotsize = 27,4,3

#binsize = 'raww',50,15,10,1
#plotsize = 24,0,3
#plotsize = 29,0,5

#seems to suggest that trucks do have some error that look bimodal (because of the tracking error issue when they enter camera?)
#definitely seems that trucks have some weird stuff going on, but in general the error can be argued as being normal 
#these plots tend to get bumpy, espsecially for those plots with smaller total measurments. 
#stuff gets bumpier when speed is decreased as well 

#for errortraj, plots tend to be upside-down U shaped for reconstructed, and they tend to change directions several times. relatively smooth 
#for raw data, plots tend to be pretty much the same, no real noticable differences 
####################################################################################





    

    
                

        