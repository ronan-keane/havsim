# -*- coding: utf-8 -*-
"""
This file can load in the US101 original data, treiterer and myers 1974 dataset, and highD dataset.
Data format -

            0 - vehicle ID
            1 - time
            2- position,
            3 - speed,
            4 - leader,
            5 - follower.
            6 - length,
            7 - lane,
            8 - acceleration

Created on Tue Jul 23 17:51:26 2019

@author: rlk268
"""
import re
import numpy as np
import csv
import pickle

def load101():
    #load original US101 data
    #only first 15 minutes are loaded but you can load the rest as well if you want.
    numentries = 18 #there are 18 columns in the data
    data = []
    file = open('trajectories-0750am-0805am.txt', 'r') #raw data from fhwa, change the filename here if you want to see the other time intervals

    for line in file:
        for s in re.findall(r'\d+(?:\.\d+)?',line):  #find all ints and floats in the string
            data.append(float(s)) #append eveyrthing into a long array as floats

    l = len(data)
    data = np.asarray(data)
    data = np.reshape(data, (int(l/numentries),numentries))

    #if desired you can redifferentiate the data or clean/filter it. Note that this data is fairly inaccurate. Can refer to julien montein's dissertation for e.g. of filtering

    #we don't want every single entry
    data = data[:,[0,1,5,11,14,15,8,13,12]]

    return data


def load1974():
    #load the treiterer and myers 1974 dataset
    #created by benjamin coifman based on the figures from the 1974 paper, can refer to his 2018 paper and website for further details
    #note there is no information about the lengths of the vehicles.

    with open('coifman2018data.csv',newline='') as f:
        readdata = csv.reader(f)
        numentries = 18
        rows=279767
#        rows = sum(1 for i in readdata)
#        readdata = csv.reader(f)

        data = np.zeros((rows,numentries))
        counter = 0
        for row in readdata:
            data[counter,:] = [float(i) for i in row]
            counter+=1


        data = data[:,[0,1,5,11,14,15,8,13,12]]

    return data

def loadhighd(datanum = 1, drivingdir = 1):
    #load the highD dataset
    #datanum = 1 - which data to use, int from 1 to 60
    #drivingdir = 1 - which driving direction (opposite sides of the road on divided highway), you must choose a driving direction, so there are 120 datasets in total.

    #get path of desired data
    path = 'C:/Users/rlk268/OneDrive - Cornell University/important misc/datasets/highD-dataset-v1.0'
    if datanum < 10:
        pathnum = '/'
        pathnum = pathnum+str(0)+str(datanum)
    else:
        pathnum = '/'
        pathnum = pathnum+str(datanum)
    metapath = path+pathnum+'_tracksMeta.csv'
    datapath = path+pathnum+'_tracks.csv'

    #first we need to figure out which vehicles are the ones
    #we will create a dictionary which returns True if vehicle is in the desired driving direction and False otherwise.
    dirdict = {} #driving direction dictionary
    #populate dictionary
    with open(metapath, newline='') as f:
        readdata = csv.reader(f)
        next(readdata) #skip first row
        rows = 0 #count number of rows the final data will have
        for row in readdata: #iterate over rows
            if float(row[7]) == drivingdir: #if driving direction is correct
                dirdict[float(row[0])]  = True
                rows += float(row[5])
            else:
                dirdict[float(row[0])]  = False


    #load in data
    with open(datapath, newline='') as f:
        numentries = 25 #data originally has 25 columns although we reduce this to 9
        readdata = csv.reader(f)
        next(readdata) #skip first row

        data = np.zeros((int(rows),numentries))
        counter = 0
        for row in readdata:
            if dirdict[float(row[1])]: #if vehicle is in correct driving direction
                data[counter,:] = [float(i) for i in row]
                counter += 1


    data = data[:,[1,0,2,6,16,17,4,24,8]] #put data in format we want

    if drivingdir ==1:
        data[:,2] = -data[:,2]
        data[:,3] = -data[:,3]

    return data

#data101 = load101()

#highd = loadhighd()

#data1974 = load1974()

#with open('data2019.pkl','wb') as f:
#    pickle.dump([data101, highd, data1974], f)

#with open('data2019.pkl','rb') as f:
#    data101, highd, data1974 = pickle.load(f)

highd = loadhighd(datanum = 12, drivingdir = 1)

#print(np.unique(highd[:,7]))

#highd datasets with congestion - 25, 26