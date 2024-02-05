
"""
All calibration related functions which set up optimization problems, including doing the actual simulation of vehicle trajectories.


    \\ TO DO
	HIGH PRIORITY
		-being able to use autodifferentiation so we don't have to rely on adjoint calculation
		-want to rewrite the platoonobjfn to use python lists and be vectorized - should have similar construction to simulation code
        should use auxinfo/modelinfo and take up less memory as well ; will need to modify plotting functions to work with
        this updated format
            -many of the problems in general features/QOL can be solved by using the new simulation code in place of current calibration code
        -We have written code for LL and delay models which has not been tested
        -should refactor calibration code to be cleaner, including a polished api for users
        -create test script that can be used to check for bugs in the calibration code


    general features/QOL
        -support for using relaxation phenomenon on a platoon, where we would like to update the relaxation amount during simulation.
            -Also need to support relaxation for delay
        -need to implement solving for delay models

    \\ TO DO


@author: rlk268@cornell.edu
"""

import numpy as np
import math

from collections import deque
import scipy.interpolate as sci
import scipy.optimize as sc
from havsim.old import models

from havsim.helper import r_constant


def r_constant2(currinfo, frames, T_n, rp, h = .1):
    #old strategy for overwriting gamma values

    #given a list of times and gamma constants (rinfo for a specific vehicle = currinfo) as well as frames (t_n and T_nm1 for that specific vehicle) and the relaxation constant (rp). h is the timestep (.1 for NGSim)
    #we will make the relaxation amounts for the vehicle over the length of its trajectory
    #rinfo is precomputed in makeleadfolinfo_r. then during the objective evaluation/simulation, we compute these times.
    #note that we may need to alter the pre computed gammas inside of rinfo; that is because if you switch mutliple lanes in a short time, you may move to what looks like only a marginally shorter headway,
    #but really you are still experiencing the relaxation from the lane change you just took
    out = np.zeros(T_n-frames[0]+1) #initialize relaxation amount for the time between t_n and T_n
    out2 = np.zeros(T_n-frames[0]+1)
    if len(currinfo)==0:
        return out, out2 #if currinfo is empty we don't have to do anything else
    maxind = frames[1]-frames[0]+1 #this is the maximum index we are supposed to put values into because the time between T_nm1 and T_n is not simulated. Plus 1 because of the way slices work.
    if rp<h: #if relaxation is too small for some reason
        rp = h #this is the smallest rp can be
#    if rp<h: #if relaxation is smaller than the smallest it can be #deprecated
#        return out, out2 #there will be no relaxation

    mylen = math.ceil(rp/h)-1 #this is how many nonzero entries will be in r each time we have the relaxation constant
    r = np.linspace(1-h/rp,1-h/rp*(mylen),mylen) #here are the relaxation constants. these are determined only by the relaxation constant. this gets multipled by the 'gamma' which is the change in headway immediately after the LC

    for i in range(len(currinfo)): #frames[1]-frames[0]+1 is the length of the simulation; this makes it so it will be all zeros between T_nm1 and T_n
        entry = currinfo[i] #the current entry for the relaxation phenomenon
        curind = entry[0]-frames[0] #current time is entry[0]; we start at frames[0] so this is the current index
        curgamma = out[curind]*out2[curind] #the current value in the relaxation constant

        #can change the absolute value places
        if curgamma > abs(entry[1]): #if we're still being relaxed we might use the current constant instead of the gamma computed in makeleadfolinfo_r
            usegamma = curgamma
        else:
            usegamma = entry[1]

#        usegamma = curgamma + entry[1] #alternate strategy #note that this doesn't work because of how the adjoint calculation is handled.
        #it doesn't seem to really help anyway.

        if curind+mylen > maxind: #in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)
            out[curind:maxind] = r[0:maxind-curind]
            out2[curind:maxind] = usegamma
        else: #this is the normal case
            out[curind:curind+mylen] = r
            out2[curind:curind+mylen] = usegamma
    return out, out2


def r_constant3(currinfo, frames, T_n, rp, rp_n, adj = True, h = .1):
    #this is for 2 parameter relaxation. rp_n is the parameter for negative relaxation

    posr = []
    negr = []

    for i in currinfo:
        if i[1]>0:
            posr.append(i)
        else:
            negr.append(i)

    relax1, relaxadjpos = r_constant(posr,frames,T_n,rp,adj,h)
    relax2, relaxadjneg = r_constant(negr,frames,T_n,rp_n,adj,h)

    relax = relax1+relax2

    return relax, relaxadjpos,relaxadjneg

def rinfo_newell(currinfo, frames, lead, rp, rp_n, relax2,h=.1):
    #enforce the "smoothness condition" which will make the speed of newell less spiky
    #this is only for newell i.e. TTobjfn
    if not relax2: #if only 1 parameter set it equal to rp_n
        rp = rp_n

    for j in range(len(currinfo)):
        i = currinfo[j]
        if i[0] < frames[0]+2: #need this to use the strategy
            continue
        if i[1] > 0:
            c = rp
        else:
            c = rp_n
        tjm1 = i[0]-frames[0]-2
        tj = tjm1+1
        tjp1 = tjm1+2
        rhs = h/2*(lead[tjm1,3]+lead[tjp1,3])+lead[tj,2]+lead[tjp1,6]-lead[tjp1,2]-lead[tj,6]
        cn = 2*c/(2*c-h)
        currinfo[j][1] = cn*rhs
    return currinfo

def rescaledobjfn_obj(p, rescale, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = False, m=5, dim = 2, h = .1, datalen = 9):
	#can pass in numpy array rescale which are relative magnitudes for different parameters. In my experience this hasn't seemed to help much.
    p = np.multiply(p,1/rescale)
    obj = platoonobjfn_obj(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = use_r, m=m, dim = dim, h = h, datalen = datalen)
    return obj

def platoonobjfn_obj(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = False, m=5, dim = 2, h = .1, datalen = 9):
    #inputs -
    #p - list of parameter values for the chosen model
    #model - function that calls the model used.
    #modeladjsys - function that computes the adjoint variables for the model used
    #modeladj - function that computes the gradient given the adjoint variables and model output
    #meas - measurements in dictionary format where key is the vehicle ID
    #sim - simulation in dictionary format where key is the vehicle ID
    #platooninfo - contains information about each vehicle, key is vehicle ID
    #platoons - the platoon to be simulated. Can either be in format [[], vehid1, vehid2, vehid3, ....] or [[time0, time1, time2,. ...], [vehid1, vehid2,...],[vehid1, vehid2, vehid3,...]]
    #leadinfo - output from makeleadfolinfo, needed to form lead trajectory
    #folinfo - output frmo makeleadfolinfo
    #rinfo - output from makeleadfolinfo
    #use_r = False - choose whether or not to use relaxation. Note that even if this is True you also need to make sure that the rinfo is correct and pass it in.
    #the relaxation parameters should be at the end of p (e.g. p[-1] for 1 parameter, p[-2], p[-1] are positive, negative relaxation values respectively )
    #m = 5 - number of parameters, len(p) is not necssarily the same as m because m only
    #dim = 2 - dimension of model, so we regard second order model as dim 2 and dim 1 otherwise.
    #h = .1 - time step used for simulation. Currently we have to have h the same as the discretization used for data, if you don't want this you can use sci.interp1d
    #datalen = 9 - this is just always 9.

    #outputs -
    #obj, grad - objective, gradient

    #versions of program -
    #a 2 at the end indicates it is for the 2 parameter relaxation phenomenon. Otherwise it is the 1 parameter version.
    #_obj - objective only, _der - gradient only, _objder - returns both objective and gradient



    #note that I'm pretty sure you could get rid of the make leader and follower parts being inside this function if you just did it outside.

    lead = {} #dictionary where we put the relevant lead vehicle information
    obj = 0
    n = len(platoons)

    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = p[m*i:m*(i+1)]
        curvehid = platoons[i] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n, T_nm1]

        relax,unused = r_constant(rinfo[i],frames,T_n,curp[-1],False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation


        leadlen = lead[i][:,6] #get lead length information
        IC = platooninfo[curvehid][5:7] #get initial conditions
#        IC = sim[curvehid][t_n-t_nstar,dataind[0:2]]
        simtraj, reg = euler(curp, frames, IC, model, lead[i], leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1

        #reg keeps track of the model regime. we don't actually need it if we just want to evaluate the obj fn.

        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly?
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        loss = simtraj[:T_nm1+2-t_n,1] - curmeas[t_n-t_nstar:T_nm1+2-t_nstar,2] #calculate difference between simulation and measurments up til the first entry of shifted end (not inclusive)
        loss = np.square(loss) #squared error in distance is the loss function we are using. will need to add ability to choose loss function later
#        obj += sum(loss)
#        obj += sci.simps(loss,None,dx=h,axis=0,even='first')
        obj += sum(loss)*h #weighted by h


    return obj

def platoonobjfn_obj2(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = False, m=5, dim = 2, h = .1, datalen = 9):
    #note that I'm pretty sure you could get rid of the make leader and follower parts being inside this function if you just did it outside.

    lead = {} #dictionary where we put the relevant lead vehicle information
    obj = 0
    n = len(platoons)

    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = p[m*i:m*(i+1)]
        curvehid = platoons[i+1] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n, T_nm1]

        relax,unused,unused = r_constant3(rinfo[i],frames,T_n,curp[-2],curp[-1],False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation


        leadlen = lead[i][:,6] #get lead length information
        IC = platooninfo[curvehid][5:7] #get initial conditions
#        IC = sim[curvehid][t_n-t_nstar,dataind[0:2]]
        simtraj, reg = euler(curp, frames, IC, model, lead[i], leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1

        #reg keeps track of the model regime. we don't actually need it if we just want to evaluate the obj fn.

        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly? #again, no
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        loss = simtraj[:T_nm1+2-t_n,1] - curmeas[t_n-t_nstar:T_nm1+2-t_nstar,2] #calculate difference between simulation and measurments up til the first entry of shifted end (not inclusive)
        loss = np.square(loss) #squared error in distance is the loss function we are using. will need to add ability to choose loss function later
#        obj += sum(loss)
#        obj += sci.simps(loss,None,dx=h,axis=0,even='first')
        obj += sum(loss)*h #weighted by h


    return obj

def platoonobjfn_obj_b(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, bounds, use_r = False, m=5, dim = 2, h = .1, datalen = 9):
    #this is for nelder mead

    lead = {} #dictionary where we put the relevant lead vehicle information
    obj = 0
    n = len(platoons)

    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = p[m*i:m*(i+1)]
        curvehid = platoons[i+1] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n, T_nm1]

        relax,unused = r_constant(rinfo[i],frames,T_n,curp[-1],False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation


        leadlen = lead[i][:,6] #get lead length information
        IC = platooninfo[curvehid][5:7] #get initial conditions
#        IC = sim[curvehid][t_n-t_nstar,dataind[0:2]]
        simtraj, reg = euler(curp, frames, IC, model, lead[i], leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1

        #reg keeps track of the model regime. we don't actually need it if we just want to evaluate the obj fn.

        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly? #again, no
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        loss = simtraj[:T_nm1+2-t_n,1] - curmeas[t_n-t_nstar:T_nm1+2-t_nstar,2] #calculate difference between simulation and measurments up til the first entry of shifted end (not inclusive)
        loss = np.square(loss) #squared error in distance is the loss function we are using. will need to add ability to choose loss function later
#        obj += sum(loss)
#        obj += sci.simps(loss,None,dx=h,axis=0,even='first')
        obj += sum(loss)*h #weighted by h

    for i in range(len(p)): #add a penalty function to enforce bounds
        if p[i] < bounds[i][0] or p[i]> bounds[i][1]:
            obj += 1e8
            break
    return obj



def platoonobjfn_der(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, dim = 2, h = .1, datalen = 9):

    #first we need to simualte all the vehicles
    lead = {} #dictionary where we put the relevant lead vehicle information
    lam = {}
    extra = {} #this will hold the regime, the r, and gamma
    n = len(platoons)
    grad = np.zeros(n*m)
    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = p[m*i:m*(i+1)] #parameters for current vehicle
        curvehid = platoons[i] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n,T_nm1]

        relax,relaxadj = r_constant(rinfo[i],frames,T_n,curp[-1],True,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
#        print('vehicle '+str(curvehid)) #for debugging
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = platooninfo[curleadid][0] #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
        leadlen = lead[i][:,6] #get lead length information #note faster if we just pass it in directly?
        IC = platooninfo[curvehid][5:7] #get initial conditions  #again, faster if we just pass it in directly?
        simtraj, reg = euler(curp,frames, IC, model, lead[i],leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1
        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly?
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        inmodel = np.zeros(T_n-t_n+1)
        inmodel[:T_nm1-t_n+1] = reg
        extra[i] = [inmodel, relax, relaxadj]

    for i in range(n,0,-1): #go backwards over the vehicles for gradient
        curp = p[m*(i-1):m*i]
        curvehid = platoons[i-1] #current vehicle in the platoon
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4]
        frames = [t_n, T_n]
        inmodel,relax,relaxadj = extra[i-1][0:3]

        #we need the below things because of coupling term between leader and follower

        havefol = np.zeros(T_n-t_n+1) #this says whether the vehicle in question has a following vehicle (in the platoon)
        curfol = np.zeros((T_n+1-t_n,datalen)) #this is where the follower measurements go
        lamp1 = np.zeros((T_n+1-t_n,dim))
        vehlen = sim[curvehid][0,6]
        folp = [] #initialize
        folrelax = np.zeros(T_n-t_n+1)

        for j in folinfo[i-1]: #need i-1 because platoons has a special entry as its 0 index but folinfo and leadinfo don't have this

            #we can implement relaxation phenomenon by finding the times we need to apply it in makeleadfolinfo

            curfolid = j[0] #get current follower
            folind = platoons.index(curfolid) #this is the index of the current follower
            folp = p[m*(folind):m*(folind+1)]
            folt_nstar, folt_n = platooninfo[curfolid][0:2] #first time follower is observed
            curfol[j[1]-t_n:j[2]+1-t_n,:] = sim[curfolid][j[1]-folt_nstar:j[2]+1-folt_nstar,:] #load in the relevant portion of follower trajectory from the simulation
            #add in a part here to handle the adjoint variables
            lamp1[j[1]-t_n:j[2]+1-t_n,:] = lam[curfolid][j[1]-folt_n:j[2]+1-folt_n,[1,2]]
            havefol[j[1]-t_n:j[2]+1-t_n] = extra[folind-1][0][j[1]-folt_n:j[2]+1-folt_n]
            folrelax[j[1]-t_n:j[2]+1-t_n] = extra[folind-1][1][j[1]-folt_n:j[2]+1-folt_n]


        #note that lead[i-1], leadlen, curfol, havefol, inmodel, and lambdas are all defined on times t_n to T_n.
        #sim[i] and meas[i] are both defined on t_nstar to T_n. Of course we don't really care about the times t_nstar to t_n; we don't do anything in those times
#        simtraj = sim[curvehid] #simulation
#        curlead = lead[i-1] #leader
#        vehstar = meas[curvehid] #measurements
        leadlen = lead[i-1][:,6] #leader's length
        #note that shiftloss is dependent on both the loss function as well as the dimension of the model
        shiftloss = 2*(sim[curvehid][-1,2]-meas[curvehid][-1,2])/(T_n-T_nm1) #this computes the derivative of loss in the shifted end, then weights it by the required amount
        #now we can compute the adjoint variables for the current vehicle.
        lam[curvehid] = euleradj2(curp, frames, modeladjsys, lead[i-1], leadlen, meas[curvehid], sim[curvehid], curfol, inmodel, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss)

#        calculate the gradient now
        lang = np.zeros((T_nm1-t_n+1,m)) #lagrangian
        offset = t_n-t_nstar #offset for indexing lead and sim
#        cursim = sim[curvehid][offset:,dataind[0:2]]
#        curlead = lead[i-1][:,dataind[0:2]]
#        curlam = lam[curvehid][:,[1,2]]

        for j in range(T_nm1-t_n+1):
            lang[j,:] = modeladj(sim[curvehid][j+offset,2:4], lead[i-1][j,2:4],curp,leadlen[j],lam[curvehid][j,[1,2]], inmodel[j], relax[j], relaxadj[j], use_r)
#            lang[j,:] = modeladj(cursim[j,:], curlead[j,:],curp,leadlen[j],curlam[j,:])
#        temp = sci.simps(lang,None,h,axis=0,even = 'first') #this is less accurate than using a riemann sum because of the euler integration scheme
        temp = np.sum(lang,0)*h
        grad[m*(i-1):m*i] = temp


    return grad
def platoonobjfn_der2(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, dim = 2, h = .1, datalen = 9):
    #2 parameter version for the relaxation phenomenon; derivative of objective only

    lead = {} #dictionary where we put the relevant lead vehicle information
    lam = {}
    extra = {} #this will hold the regime, the r, and gamma
    n = len(platoons) #fix this in this funciton and the above
    grad = np.zeros(n*m)
    obj = 0
    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = p[m*i:m*(i+1)] #parameters for current vehicle
        curvehid = platoons[i+1] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n,T_nm1]

        relax,relaxadjpos,relaxadjneg = r_constant3(rinfo[i],frames,T_n,curp[-2],curp[-1],True,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
#        print('vehicle '+str(curvehid)) #for debugging
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = platooninfo[curleadid][0] #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
        leadlen = lead[i][:,6] #get lead length information #note faster if we just pass it in directly?
        IC = platooninfo[curvehid][5:7] #get initial conditions  #again, faster if we just pass it in directly?
        simtraj, reg = euler(curp,frames, IC, model, lead[i],leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1
        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly?
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        inmodel = np.zeros(T_n-t_n+1)
        inmodel[:T_nm1-t_n+1] = reg


        extra[i] = [inmodel, relax, relaxadjpos,relaxadjneg]


    for i in range(n,0,-1): #go backwards over the vehicles for gradient
        curp = p[m*(i-1):m*i]
        curvehid = platoons[i] #current vehicle in the platoon
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4]
        frames = [t_n, T_n]
        inmodel,relax,relaxadjpos,relaxadjneg = extra[i-1][0:4]

        #we need the below things because of coupling term between leader and follower

        havefol = np.zeros(T_n-t_n+1) #this says whether the vehicle in question has a following vehicle (in the platoon)
        curfol = np.zeros((T_n+1-t_n,datalen)) #this is where the follower measurements go
        lamp1 = np.zeros((T_n+1-t_n,dim))
        vehlen = sim[curvehid][0,6]
        folp = [] #initialize
        folrelax = np.zeros(T_n-t_n+1)

        for j in folinfo[i-1]: #need i-1 because platoons has a special entry as its 0 index but folinfo and leadinfo don't have this

            #we can implement relaxation phenomenon by finding the times we need to apply it in makeleadfolinfo

            curfolid = j[0] #get current follower
            folind = platoons.index(curfolid) #this is the index of the current follower
            folp = p[m*(folind-1):m*folind]
            folt_nstar, folt_n = platooninfo[curfolid][0:2] #first time follower is observed
            curfol[j[1]-t_n:j[2]+1-t_n,:] = sim[curfolid][j[1]-folt_nstar:j[2]+1-folt_nstar,:] #load in the relevant portion of follower trajectory from the simulation
            #add in a part here to handle the adjoint variables
            lamp1[j[1]-t_n:j[2]+1-t_n,:] = lam[curfolid][j[1]-folt_n:j[2]+1-folt_n,[1,2]]
            havefol[j[1]-t_n:j[2]+1-t_n] = extra[folind-1][0][j[1]-folt_n:j[2]+1-folt_n]
            folrelax[j[1]-t_n:j[2]+1-t_n] = extra[folind-1][1][j[1]-folt_n:j[2]+1-folt_n]


        #note that lead[i-1], leadlen, curfol, havefol, inmodel, and lambdas are all defined on times t_n to T_n.
        #sim[i] and meas[i] are both defined on t_nstar to T_n. Of course we don't really care about the times t_nstar to t_n; we don't do anything in those times
#        simtraj = sim[curvehid] #simulation
#        curlead = lead[i-1] #leader
#        vehstar = meas[curvehid] #measurements
        leadlen = lead[i-1][:,6] #leader's length
        #note that shiftloss is dependent on both the loss function as well as the dimension of the model
        shiftloss = 2*(sim[curvehid][-1,2]-meas[curvehid][-1,2])/(T_n-T_nm1) #this computes the derivative of loss in the shifted end, then weights it by the required amount
        #now we can compute the adjoint variables for the current vehicle.
        lam[curvehid] = euleradj(curp, frames, modeladjsys, lead[i-1], leadlen, meas[curvehid], sim[curvehid], curfol, inmodel, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss)

#        calculate the gradient now
        lang = np.zeros((T_nm1-t_n+1,m)) #lagrangian
        offset = t_n-t_nstar #offset for indexing lead and sim
#        cursim = sim[curvehid][offset:,dataind[0:2]]
#        curlead = lead[i-1][:,dataind[0:2]]
#        curlam = lam[curvehid][:,[1,2]]

        for j in range(T_nm1-t_n+1):
            lang[j,:] = modeladj(sim[curvehid][j+offset,2:4], lead[i-1][j,2:4],curp,leadlen[j],lam[curvehid][j,[1,2]], inmodel[j], relax[j], relaxadjpos[j], relaxadjneg[j],use_r)
#            lang[j,:] = modeladj(cursim[j,:], curlead[j,:],curp,leadlen[j],curlam[j,:])
#        temp = sci.simps(lang,None,h,axis=0,even = 'first')
        temp = np.sum(lang,0)*h
        grad[m*(i-1):m*i] = temp


    return grad

def rescaledobjfn_objder(p, rescale, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = False, m=5, dim = 2, h = .1, datalen = 9):
    p = np.multiply(p,1/rescale)
    obj, grad = platoonobjfn_objder(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = use_r, m=m, dim = dim, h = h, datalen = datalen)
    return obj, grad

def platoonobjfn_objder(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, dim = 2, h = .1, datalen = 9):
    #this is here because l-bfgs-b wants the function to return both obj and gradient

    lead = {} #dictionary where we put the relevant lead vehicle information
    lam = {}
    extra = {} #this will hold the regime, the r, and gamma
    n = len(platoons) #fix this in this funciton and the above
    grad = np.zeros(n*m)
    obj = 0
    pdict = {i: p[m*i:m*(i+1)] for i in range(n)}
    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = pdict[i] #parameters for current vehicle
        curvehid = platoons[i] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n,T_nm1]

        relax,relaxadj = r_constant(rinfo[i],frames,T_n,curp[-1],True,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
#        print('vehicle '+str(curvehid)) #for debugging
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = platooninfo[curleadid][0] #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
        leadlen = lead[i][:,6] #get lead length information #note faster if we just pass it in directly?
        IC = platooninfo[curvehid][5:7] #get initial conditions  #again, faster if we just pass it in directly?
        simtraj, reg = euler(curp,frames, IC, model, lead[i],leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1
        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly?
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        loss = simtraj[:T_nm1+2-t_n,1] - curmeas[t_n-t_nstar:T_nm1+2-t_nstar,2] #calculate difference between simulation and measurments up til the first entry of shifted end (not inclusive)
        loss = np.square(loss) #squared error in distance is the loss function we are using. will need to add ability to choose loss function later
#        obj += sci.simps(loss,None,dx=h,axis=0,even='first')
        obj += sum(loss)*h

        inmodel = np.zeros(T_n-t_n+1)
        inmodel[:T_nm1-t_n+1] = reg


        extra[i] = [inmodel, relax, relaxadj]


    for i in range(n-1,-1,-1): #go backwards over the vehicles for gradient
        curp = pdict[i]
        curvehid = platoons[i] #current vehicle in the platoon
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4]
        frames = [t_n, T_n]
        inmodel,relax,relaxadj = extra[i][0:3]

        #we need the below things because of coupling term between leader and follower

        havefol = np.zeros(T_n-t_n+1) #this says whether the vehicle in question has a following vehicle (in the platoon)
        curfol = np.zeros((T_n+1-t_n,datalen)) #this is where the follower measurements go
        lamp1 = np.zeros((T_n+1-t_n,dim))
        vehlen = sim[curvehid][0,6]
        folplist = np.zeros(T_n+1-t_n) #initialize
        folrelax = np.zeros(T_n-t_n+1)

        for j in folinfo[i]: #need i-1 because platoons has a special entry as its 0 index but folinfo and leadinfo don't have this

            #we can implement relaxation phenomenon by finding the times we need to apply it in makeleadfolinfo

            curfolid = j[0] #get current follower
            folind = platoons.index(curfolid) #this is the index of the current follower
            folt_nstar, folt_n = platooninfo[curfolid][0:2] #first time follower is observed
            curfol[j[1]-t_n:j[2]+1-t_n,:] = sim[curfolid][j[1]-folt_nstar:j[2]+1-folt_nstar,:] #load in the relevant portion of follower trajectory from the simulation
            #add in a part here to handle the adjoint variables
            lamp1[j[1]-t_n:j[2]+1-t_n,:] = lam[curfolid][j[1]-folt_n:j[2]+1-folt_n,[1,2]]
            havefol[j[1]-t_n:j[2]+1-t_n] = extra[folind][0][j[1]-folt_n:j[2]+1-folt_n]
            folplist[j[1]-t_n:j[2]+1-t_n] = folind
            folrelax[j[1]-t_n:j[2]+1-t_n] = extra[folind][1][j[1]-folt_n:j[2]+1-folt_n]


        #note that lead[i-1], leadlen, curfol, havefol, inmodel, and lambdas are all defined on times t_n to T_n.
        #sim[i] and meas[i] are both defined on t_nstar to T_n. Of course we don't really care about the times t_nstar to t_n; we don't do anything in those times
#        simtraj = sim[curvehid] #simulation
#        curlead = lead[i-1] #leader
#        vehstar = meas[curvehid] #measurements
        leadlen = lead[i][:,6] #leader's length
        #note that shiftloss is dependent on both the loss function as well as the dimension of the model
        shiftloss = 2*(sim[curvehid][-1,2]-meas[curvehid][-1,2])/(T_n-T_nm1) #this computes the derivative of loss in the shifted end, then weights it by the required amount
        #now we can compute the adjoint variables for the current vehicle.
        lam[curvehid] = euleradj(curp, frames, modeladjsys, lead[i], leadlen, meas[curvehid], sim[curvehid], curfol, inmodel,
           havefol, vehlen, lamp1, folplist, relax, folrelax, shiftloss, pdict)

#        calculate the gradient now
        lang = np.zeros((T_nm1-t_n+1,m)) #lagrangian
        offset = t_n-t_nstar #offset for indexing lead and sim
#        cursim = sim[curvehid][offset:,dataind[0:2]]
#        curlead = lead[i-1][:,dataind[0:2]]
#        curlam = lam[curvehid][:,[1,2]]

        for j in range(T_nm1-t_n+1):
            lang[j,:] = modeladj(sim[curvehid][j+offset,2:4], lead[i][j,2:4],curp,leadlen[j],lam[curvehid][j,[1,2]], inmodel[j], relax[j], relaxadj[j], use_r)
#            lang[j,:] = modeladj(cursim[j,:], curlead[j,:],curp,leadlen[j],curlam[j,:])
#        temp = sci.simps(lang,None,h,axis=0,even = 'first')
        temp = np.sum(lang,0)*h
        grad[m*i:m*(i+1)] = temp


    return obj, grad

def platoonobjfn_objder2(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, dim = 2, h = .1, datalen = 9):
    #2 parameter version for the relaxation phenomenon
    #this is here because l-bfgs-b wants the function to return both obj and gradient

    lead = {} #dictionary where we put the relevant lead vehicle information
    lam = {}
    extra = {} #this will hold the regime, the r, and gamma
    n = len(platoons) #fix this in this funciton and the above
    grad = np.zeros(n*m)
    obj = 0
    for i in range(n): #iterate over all vehicles in the platoon
        #first get the lead trajectory and length for the whole length of the vehicle
        #then we can use euler function to get the simulated trajectory
        #lastly, we want to apply the shifted trajectory strategy to finish the trajectory
        #we can then evaluate the objective function
        curp = p[m*i:m*(i+1)] #parameters for current vehicle
        curvehid = platoons[i+1] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n,T_nm1]

        relax,relaxadjpos,relaxadjneg = r_constant3(rinfo[i],frames,T_n,curp[-2],curp[-1],True,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only

        lead[i] = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
#        print('vehicle '+str(curvehid)) #for debugging
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = platooninfo[curleadid][0] #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
        leadlen = lead[i][:,6] #get lead length information #note faster if we just pass it in directly?
        IC = platooninfo[curvehid][5:7] #get initial conditions  #again, faster if we just pass it in directly?
        simtraj, reg = euler(curp,frames, IC, model, lead[i],leadlen, relax, dim, h) #get the simulated trajectory between t_n and T_nm1
        curmeas = meas[curvehid] #current measurements #faster if we just pass it in directly?
        if T_n > T_nm1: #if we need to do the shifted end
            shiftedtraj = shifted_end(simtraj,curmeas,t_nstar,t_n,T_nm1,T_n)
            simtraj = np.append(simtraj,shiftedtraj,0) #add the shifted end onto the simulated trajectory
        #need to add in the simulated trajectory into sim
#        sim[curvehid] = meas[curvehid].copy()
        sim[curvehid][t_n-t_nstar:,2:4] = simtraj[:,1:3]

        loss = simtraj[:T_nm1+2-t_n,1] - curmeas[t_n-t_nstar:T_nm1+2-t_nstar,2] #calculate difference between simulation and measurments up til the first entry of shifted end (not inclusive)
        loss = np.square(loss) #squared error in distance is the loss function we are using. will need to add ability to choose loss function later
#        obj += sci.simps(loss,None,dx=h,axis=0,even='first')
        obj += sum(loss)*h

        inmodel = np.zeros(T_n-t_n+1)
        inmodel[:T_nm1-t_n+1] = reg


        extra[i] = [inmodel, relax, relaxadjpos,relaxadjneg]


    for i in range(n,0,-1): #go backwards over the vehicles for gradient
        curp = p[m*(i-1):m*i]
        curvehid = platoons[i] #current vehicle in the platoon
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4]
        frames = [t_n, T_n]
        inmodel,relax,relaxadjpos,relaxadjneg = extra[i-1][0:4]

        #we need the below things because of coupling term between leader and follower

        havefol = np.zeros(T_n-t_n+1) #this says whether the vehicle in question has a following vehicle (in the platoon)
        curfol = np.zeros((T_n+1-t_n,datalen)) #this is where the follower measurements go
        lamp1 = np.zeros((T_n+1-t_n,dim))
        vehlen = sim[curvehid][0,6]
        folp = [] #initialize
        folrelax = np.zeros(T_n-t_n+1)

        for j in folinfo[i-1]: #need i-1 because platoons has a special entry as its 0 index but folinfo and leadinfo don't have this

            #we can implement relaxation phenomenon by finding the times we need to apply it in makeleadfolinfo

            curfolid = j[0] #get current follower
            folind = platoons.index(curfolid) #this is the index of the current follower
            folp = p[m*(folind-1):m*folind]
            folt_nstar, folt_n = platooninfo[curfolid][0:2] #first time follower is observed
            curfol[j[1]-t_n:j[2]+1-t_n,:] = sim[curfolid][j[1]-folt_nstar:j[2]+1-folt_nstar,:] #load in the relevant portion of follower trajectory from the simulation
            #add in a part here to handle the adjoint variables
            lamp1[j[1]-t_n:j[2]+1-t_n,:] = lam[curfolid][j[1]-folt_n:j[2]+1-folt_n,[1,2]]
            havefol[j[1]-t_n:j[2]+1-t_n] = extra[folind-1][0][j[1]-folt_n:j[2]+1-folt_n]
            folrelax[j[1]-t_n:j[2]+1-t_n] = extra[folind-1][1][j[1]-folt_n:j[2]+1-folt_n]


        #note that lead[i-1], leadlen, curfol, havefol, inmodel, and lambdas are all defined on times t_n to T_n.
        #sim[i] and meas[i] are both defined on t_nstar to T_n. Of course we don't really care about the times t_nstar to t_n; we don't do anything in those times
#        simtraj = sim[curvehid] #simulation
#        curlead = lead[i-1] #leader
#        vehstar = meas[curvehid] #measurements
        leadlen = lead[i-1][:,6] #leader's length
        #note that shiftloss is dependent on both the loss function as well as the dimension of the model
        shiftloss = 2*(sim[curvehid][-1,2]-meas[curvehid][-1,2])/(T_n-T_nm1) #this computes the derivative of loss in the shifted end, then weights it by the required amount
        #now we can compute the adjoint variables for the current vehicle.
        lam[curvehid] = euleradj(curp, frames, modeladjsys, lead[i-1], leadlen, meas[curvehid], sim[curvehid], curfol, inmodel, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss)

#        calculate the gradient now
        lang = np.zeros((T_nm1-t_n+1,m)) #lagrangian
        offset = t_n-t_nstar #offset for indexing lead and sim
#        cursim = sim[curvehid][offset:,dataind[0:2]]
#        curlead = lead[i-1][:,dataind[0:2]]
#        curlam = lam[curvehid][:,[1,2]]

        for j in range(T_nm1-t_n+1):
            lang[j,:] = modeladj(sim[curvehid][j+offset,2:4], lead[i-1][j,2:4],curp,leadlen[j],lam[curvehid][j,[1,2]], inmodel[j], relax[j], relaxadjpos[j], relaxadjneg[j],use_r)
#            lang[j,:] = modeladj(cursim[j,:], curlead[j,:],curp,leadlen[j],curlam[j,:])
#        temp = sci.simps(lang,None,h,axis=0,even = 'first')
        temp = np.sum(lang,0)*h
        grad[m*(i-1):m*i] = temp


    return obj, grad

def TTobjfn_obj(p, unused, unused2, unused3, meas, sim, platooninfo, platoons, leadinfo, unused4, rinfo, use_r = False, m =2, relax2 = False, smooth=False, dim = 0, h = .1, datalen = 9):
    #trajectory translation i.e. newell model
    #we will use finite difference method to calculate the gradient
    #you could probably use the adjoint method if you want

    #this is only going to be for a single vehicle since vehicles are uncoupled in newell model.
    #really you might still want some sort of platoon feature because of circular dependency

    #2 parameters; p[0] = tau (time shift), p[1] = d (space shift)
    #if smooth is true, do special smoothing strategy for relaxation, this is explained in relaxation phenomenon paper

    t_nstar, t_n, T_nm1, T_n = platooninfo[platoons[1]][0:4]

    curmeas = meas[platoons[1]]

    lead = np.zeros((T_n+1-t_n,datalen)) #lead vehicle trajectory
    for j in leadinfo[0]:
        curleadid = j[0] #current leader ID
        leadt_nstar = platooninfo[curleadid][0] #t_nstar for the current lead, put into int
        lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation

    offset = math.ceil(p[0]/h)
    offsetend = math.floor(p[0]/h)
    if T_nm1 + offsetend >= T_n:  #whether the delay is enough to define entire trajectory or will we need shifted end?
        end = T_n
    else:
        end = T_nm1 + offsetend
    start = t_n+offset
    times = list(range(start, end+1)) #times the  (interpolated) vehicle trajectory will eventually be defined on.
    vehtimes = np.asarray(range(t_n,T_nm1+1))+p[0]/h #these are the times that the raw newell trajectory will be defined on
    diff =len(vehtimes) - len(times) #this has to be positive
    frames = [t_n, T_nm1]

    if smooth:
        rinfo[0] = rinfo_newell(rinfo[0],frames,lead,p[-2],p[-1],relax2,h)

    if not relax2: #1 parameter relax #either choose 1 or 2 parameter relax (or if you pass rinfo as empty it doens't matter)
        relax, unused = r_constant(rinfo[0],frames, T_n, p[-1],False,h)
    else: #2 parameter relax
        relax, unused,unused = r_constant3(rinfo[0],frames, T_n, p[-2],p[-1],False,h)

    leadtraj = lead[:T_nm1+1-t_n,2] + relax[:T_nm1-t_n+1] - lead[:T_nm1+1-t_n,6] #lead trajectory  #need to add the length of the vehicle into there.

    inttraj = sci.interp1d(vehtimes,leadtraj-p[1],assume_sorted = True)

    meastraj = curmeas[start-t_nstar:end-t_nstar+1,2] #measurements during simulation

    if start -1 < t_nstar: #special case when time delay is 0 because of the way we interpolate
        times1 =  list(range(start,T_n+1))
        intmeas = sci.interp1d(times1,curmeas[start-t_nstar:,2])
    else:
        times1 = list(range(start-1,T_n+1))
        intmeas = sci.interp1d(times1,curmeas[start-1-t_nstar:,2])

    if diff == 0: #special case when diff is 0 because of the way slices work
        intmeastraj = intmeas(vehtimes)
        loss2 = intmeastraj - leadtraj+p[1]
    else:
        intmeastraj = intmeas(vehtimes[:-diff]) #interpolate measurements to be defined on the raw newell trajectory times
        loss2 = intmeastraj - leadtraj[:-diff]+p[1]
    #only taking up to -diff ensures that intmeastraj and vehtraj will always have the same number of measuremnets.
    #also need to do something for convert to rmse

    vehtraj = inttraj(times) #interpolated vehicle trajectory
    loss = vehtraj - meastraj
    loss = np.square(loss)
    loss = sum(loss)*h

    loss2 = np.square(loss2)
    loss2 = sum(loss2)*h
    loss = (loss+loss2)/2

    if T_n > end: #need to get a shifted end traj in this case. otherwise nothing needs to be done.
        shiftedtraj = shifted_end(vehtraj,curmeas,t_nstar,start,end,T_n,h,dim)
        vehtraj = np.append(vehtraj,shiftedtraj[:,1],0)

    sim[platoons[1]][start-t_nstar:,2] = vehtraj
    sim[platoons[1]][:start-t_nstar,2] = meas[platoons[1]][:start-t_nstar,2].copy() #need this because the time interval we overwrite for the simulation
    #changes based on what the delay is. So if we don't reset this section back to the measurments, will get weird stuff happening before the simulation.

    #if you want to plot the stuff you are going to need to redifferentiate at the end

    return loss

def LLobjfn_obj(p, unused, unused2, unused3, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = False, m = 6, relax2 = False, dim = 0, h = .1, datalen = 9, am = 13.3):
    #LL model -
    # p = [delay, shift, max speed, eps = rate of change of eta, eta t = minimum eta reached, eta T = maximum eta reached  ]
    #note that it might be better to formulate this as a DE since we have issues with lane changing with this model.
    #for now we will go ahead and finish this though.

    lead = {}
    obj = 0
    n = len(platoons[1:])
    eta = {} #store eta in here
    extra = {} #needed to store regime

    for i in range(n):
        curp = p[m*i:m*(i+1)]
        delay = p[0]
        datadelay = delay/h
        curvehid = platoons[i+1] #current vehicle ID
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4] #grab relevant times for current vehicle
        frames = [t_n, T_nm1]

        relax,unused = r_constant(rinfo[i],frames,T_n,curp[-1],False,h) #get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.

        lead[i] = np.zeros((T_nm1+1-t_n,datalen)) #initialize the lead vehicle trajectory

        LClist = deque([])
        for j in leadinfo[i]:
            curleadid = j[0] #current leader ID
            leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
            LClist.append(j[1]) #append time of lane change
        LClist.append(None)

        lead[i] = lead[i] + relax
        leadint = sci.interp1d(lead[i][:,1], lead[i][:,2])

        curmeas = meas[curvehid]


        simlen = math.floor((frames[1]-frames[0])*h/delay)+1
        simtimes = np.linspace(t_n+datadelay,t_n+simlen*datadelay,simlen+1)
        simtraj = np.empty((simlen+1,3))
        simtraj[:,0] = simtimes
        reg = np.empty(simlen+1)
        eta[i] = np.ones(simlen+1)

        #initialize
        curtime = simtimes[0]
        prev = curmeas[t_n-t_nstar,2]
        prevspeed = curmeas[t_n-t_nstar,3]
        cureta = eta[i][0]
        #initialize
        count = 0
        etaactive = False
        curLC = LClist.popleft()

        for j in range(simlen+1):
            #timestep for the model is handled by LLobj
			#don't like how many variables we have here, but basically:
				#newx, newspeed are outputs. curreg keeps track of model regime
				#eta and etaactive are needed to manage the eta parameter in the model
				#count, curLC, LClist keep track of whether or not we can give an estimate of acceleration (needed to trigger eta effect in model)
            newx, newspeed, curreg, eta[i], count, etaactive,curLC, LCList = models.LLobj(p,simtimes,curtime,prev,prevspeed,count,eta[i],cureta,etaactive,am,leadint,curLC,LClist,h, datadelay)

            #update iteration
            simtraj[j,1] = newx
            simtraj[j,2] = newspeed
            reg[j] =curreg

            curtime = simtimes[j+1]
            prev = newx
            prevspeed = newspeed
            cureta = eta[i][j+1]



        extra[i] = reg #need to store this, if doing adjoint method I htink you would need to store relax as well and the derivative of relax as well.

        #need to do interpolation- simulation (which has delayed, noninteger times) onto the measurement times. And then the measurement onto the simulation times.
        #then we compute the objective over both. The theory is that this is a good idea because we don't want the objective to become biased towards weird oscllatory solutions
        #with long delays that end up having error cancelation when we go to compute the objective.
        delayt_n = math.ceiling(simtraj[0,0])
        delayT_nm1 = math.floor(simtraj[-1,0])
        if delayT_nm1 > T_n:
            delayT_nm1 = T_n

        measint = sci.interp1d(curmeas[:,1],curmeas[:,2]) #need to interpolate measurements onto sim times
        simint = sci.interp1d(simtraj[:,0],simtraj[:,1]) #need to interpolate simulation onto meas times
        simintspeed = sci.interp1d(simtraj[:,0],simtraj[:,2])
        inttimes = range(delayt_n, delayT_nm1+1) #these are the times that simulation gets interpolated; measurements gets interpolated on simtimes
        intlen = len(inttimes) #this is the number of simulated measurements after interpolation; note that the simlen is actually simlen + 1

        intmeas = np.zeros((intlen,))
        for count,j in enumerate(simtimes):
            intmeas[count] = measint(j) #measurements interpolated onto simtimes #note these don't need to be for loops fix this

        offset = delayt_n - t_nstar
        for count,j in enumerate(inttimes): #simulation interpolated back onto the maesurments times
            sim[curvehid][offset+count,2] = simint(j)
            sim[curvehid][offset+count,2] = simintspeed(j)

        #calculate objective
        cursimobj = simtraj[:,1] - intmeas[:,1]
        cursimobj = np.square(cursimobj)
        cursimobj = np.sum(cursimobj)

        curmeasobj = sim[curvehid][offset:offset+intlen,2] - curmeas[offset:offset+intlen,2]
        curmeasobj = np.square(curmeasobj)
        curmeasobj = np.sum(curmeasobj)

        curobj = (curmeasobj+cursimobj)*intlen/(intlen+simlen+1)
        obj += curobj

        #get shifted end
        if T_n > delayT_nm1:
            shiftedtraj = shifted_end(sim[curvehid][offset:offset+intlen,[1,2,3]], curmeas, t_nstar,inttimes[0],inttimes[-1],T_n,h,dim=1)
            sim[curvehid][offset+intlen:,[2,3]] = shiftedtraj[:,[1,2]]

    return obj


def delayobjfn_objder(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r = False, m=5, dim = 2, h=.1, datalen=9):
    #equivalent of platoonobjfn, but this is for DDE models
    lead = {}
    obj = 0
    n = len(platoons[1:])

    for i in range(n):

        curp = p[m*i:m*(i+1)]
        curvehid = platoons[i+1]
        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4]
        delay = p[0] #first parameter should always be the delay
#        if T_nm1+delay > T_n:
#            frames = [t_n+delay, T_n]
#        else:
#            frames = [t_n+delay, T_nm1+delay]
        frames = [t_n, T_nm1]

        relax, relaxadj = r_constant(rinfo[i], frames,T_n, curp[-1], True,h)

        lead[i] = np.zeros((T_nm1+1-t_n,datalen))

        curmeas = meas[curvehid]

        for j in leadinfo[i]:
            curleadid = j[0]
            leadt_nstar = platooninfo[curleadid][0] #t_nstar for the current lead, put into int
            lead[i][j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation

        leadlen = lead[i][:,6]

        simtraj, reg = eulerdelay(curp,frames,model,lead[i],leadlen,relax,curmeas,dim,h)

        delayt_n = math.floor(simtraj[0,0])+1
        delayT_nm1 = math.floor(simtraj[-1,0])
        if delayT_nm1 > T_n:
            delayT_nm1 = T_n #sometimes your last simualted point might be past the last known measurements because of the delay. This typically should not happen though.

        inttimes = range(delayt_n,delayT_nm1+1) #interpolated times - simtraj has discretization of delay and we will interpolate back onto the data's discretization
        intlen = len(inttimes)
        simlen = len(simtraj)
#        intsimtraj = np.zeros(intlen,3)

        #we calculate loss over the simulated discretization and the data discretization. it's weighted so that the objective value can be directly compared
        #to an objective calculated over the data discretization only. If you wanted to then convert from squared error to RMSE, you would divide by the intlen and take the square root.

        simint = sci.interp1d(simtraj[:,0],simtraj[:,1]) #used to interpolate simulation onto inttimes
        simintspeed = sci.interp1d(simtraj[:,0],simtraj[:,2])
        measint = sci.interp1d(curmeas[:,1],curmeas[:,2]) #interpolate measurements onto simtimes

        #interpolate the measurements onto simulated discretiaztion
        intmeas = np.zeros(intlen,2)
        intmeas[:,0] = simtraj[:,0] #times
        for count,j in enumerate(simtraj[:,0]): #note this doesn't need to be a for loop
            intmeas[count,1] = measint(j)
        #inteprolate simulation onto the meas discretization and put it into sim
        offset = delayt_n - t_nstar
        for count,j in enumerate(inttimes): #this doesn't need to be a for loop fix this
            sim[curvehid][offset+count,2] = simint(j)
            sim[curvehid][offset+count,3] = simintspeed(j)
        #calculate objective over both discretizations and weight the final objective so that RMSE can be calculated based on intlen
        cursimobj = simtraj[:,1] - intmeas[:,1]
        cursimobj = np.square(cursimobj)
        cursimobj = np.sum(cursimobj)

        curmeasobj = sim[curvehid][offset:offset+intlen,2] - curmeas[offset:offset+intlen,2]
        curmeasobj = np.square(curmeasobj)
        curmeasobj = np.sum(curmeasobj)

        curobj = (curmeasobj + cursimobj)*intlen/(intlen+simlen)
        obj += curobj

        #get shifted end or else you need to deal with boundary in some other way
        shiftedtraj = shifted_end(sim[curvehid][offset:offset+intlen,[1,2,3]], curmeas, t_nstar, t_n, inttimes[-1], T_n, h = h, dim = dim )

        sim[curvehid][offset+intlen:,[2,3]] = shiftedtraj[:,[1,2]]

    return obj

def platoonobjfn_fder(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, dim = 2, h = .1, datalen = 9):
    #this calculates the gradient using finite differences.
    grad = sc.approx_fprime(p,platoonobjfn_obj,1e-8,model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo,use_r,m,dim,h,datalen)
    return grad

def platoonobjfn_fder2(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, dim = 2, h = .1, datalen = 9):
    #this calculates the gradient using finite differences.
    grad = sc.approx_fprime(p,platoonobjfn_obj2,1e-8,model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo,use_r,m,dim,h,datalen)
    return grad

def TTobjfn_fder(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo, use_r=False, m = 5, relax2 = False, smooth=False, dim = 1, h = .1, datalen = 9):
    #this calculates the gradient using finite differences.
    grad = sc.approx_fprime(p,TTobjfn_obj,1e-8,model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo,rinfo,use_r,m,relax2,smooth,dim,h,datalen)
    return grad



def shifted_end(simtraj, meas, t_nstar, t_n, T_nm1, T_n, h = .1, dim=2):
    #model can be either dim = 1 (position output only) or dim = 2 (position + speed output)
    #t_n and T_nm1 refer to the starting and ending times of the simulation, so to use a model with delay input the delayed values.
    dim = int(dim)
    shiftedtraj = np.zeros((T_n-T_nm1,dim+1)) #for now we are just going to assume the output should have 3 columns (time, position, speed, respectively)
    time = range(T_nm1+1,T_n+1)

    if dim ==2: #this is for a second order ode model
        shiftedtraj[:,0] = time
        shiftedtraj[0,1] = simtraj[-1,1]+h*simtraj[-1,2] #position at next timestep from the velocity the vehicle currently has
        shiftedtraj[0,2] = meas[T_nm1+1-t_nstar,3] #speed in the shifted trajectory is actually the same since we are just shifted position
        if T_n-T_nm1 ==1: #in this case there is nothing else to do but this should be pretty rare
            return shiftedtraj
        shift = shiftedtraj[0,1] - meas[T_nm1+1-t_nstar,2] #difference between last position of simulation and measurements
        shiftedtraj[1:,[1,2]] = meas[T_nm1+2-t_nstar:,2:4] #load in the end trajectory taken directly from the data
        shiftedtraj[1:,1] = shiftedtraj[1:,1] + np.repeat(shift, T_n-1-T_nm1) #now shift the position by the required amount

    elif dim==1: #dimension 1 models
        #note you might to modify shiftedtraj to fill in the last speed value but this would require shifted_end to output simtraj as well as shiftedtraj
        #so I have not done this for now and the entry simtraj[-1,2] is empty.
        #note also that this is very similar to what is being done for dim==0

        #simtraj[-1,2] = meas[T_nm1-t_nstar,3] #this is what the speed is supposed to be but will require outputting simtraj
        shiftedtraj[:,0] = time
        shift = simtraj[-1,2] - meas[T_nm1-t_nstar,2]
        shiftedtraj[:,[1,2]] = meas[T_nm1-t_nstar+1:,[2,3]]
        shiftedtraj[:,[1,2]] = shiftedtraj[:,[1,2]] + np.repeat(shift,T_n-T_nm1)

        pass
    else: #for now this is only for newell model, dim = 0
        shiftedtraj[:,0] = time
        shift = simtraj[-1]- meas[T_nm1-t_nstar,2] #the shift; note that simtraj[-1] is only going to make sense for the newell model case
        shiftedtraj[:,1] = meas[T_nm1+1-t_nstar:,2] + shift


    return shiftedtraj


def euler(p,frames,IC,func,lead, leadlen, r, dim = 2, h = .1,*args):
    #############
#    this is where the main computational time is spent.
    #can we make this faster by passing things in directly? i.e. not having to do the [:,dataind[0:2]] part and np.flip part?
    #can we use cython or something to make the for loop faster?
    #############

	#p - parameters
	# frames - list of len 2, frames[0] is initial time of simulation and frames[1] last time
	#IC - initial conditions
	#func - defines model
	#lead - lead trajectory
	#leadlen - length of leader
	#r - relaxation, list
	#dim = 2 - dimension of model (first or second order DE)
	# h = .1 - timestep of data

    #only currently intended to work for ODE case
    #should be able to handle second or first order model
    #only meant to do 1 trajectory at a time
    #note that the time (frames) for lead need to match the times for the trajectory we are simulating


#    frames = np.zeros((1,2)) #get relevant frames
#    frames[0,1] = lead[-1,1]
#    frames[0,0] = lead[0,1]
    lead = lead[:,2:4]

    t = range(int(frames[0]),int(frames[1]+1))
    a = frames[1]-frames[0]+1


#    a,b = lead.shape
    simtraj = np.empty((a,3)) #initialize solution
    reg = np.empty(a)

    simtraj[:,0] = t #input frames = times
    simtraj[0,[1,2]] = IC #initial conditions


    if dim ==2: #dim 2 model gives speed and acceleration output from func.
        for i in range(len(t)-1):
            out, regime = func(simtraj[i,[1,2]],lead[i,:],p,leadlen[i],r[i])  #run the model, gets as output the derivative and the current regime
            simtraj[i+1,[1,2]] = simtraj[i,[1,2]] + h*out   #forward euler formula where func computes derivative
            reg[i] = regime #regime is defined this way with i.
            #print(func(out[i,[1,2]],lead[i,:],p))

    elif dim ==1: #dim 1 model gives speed output from func.
        for i in range(len(t)-1):
            out, regime = func(simtraj[i,[1,2]],lead[i,:],p,leadlen[i],r[i])  #run the model, gets as output the derivative and the current regime
            simtraj[i+1,1] = simtraj[i,1] + h*out[0]   #forward euler formula where func computes derivative
            simtraj[i,2] = out[0] #first order model gives speed output only
            reg[i] = regime #regime is defined this way with i.
            #print(func(out[i,[1,2]],lead[i,:],p))
    reg[-1] = regime #this is what the last regime value should be for the adjoint method. Anything else will make the gradient less accurate

    return simtraj, reg



def euleradj(p,frames, func,lead,leadlen, vehstar, simtraj, curfol, inmodel, havefol, vehlen, lamp1, folplist, relax, folrelax, shiftloss, pdict, dim = 2, h = -.1,*args):
    #############
#    this is where the main computational time is spent.
    #can we use cython or something to make the for loop faster?
    #############

    #p - parameters
	# frames - list of len 2, frames[0] is initial time of simulation and frames[1] last time
	#func - defines model
	#lead - lead trajectory
	#leadlen - length of leader
	#vehstar - measurements
	#simtraj - simulation
	#curfol - follower of vehicle
	#inmodel - regime of model, list
	#havefol - boolean records whether or not there is a leader
	#vehlen - length of vehicle
	#lamp1 - adjoint system of follower
	#folp - index of following vehicle at current time
	#relax - relaxation, list
	#folrelax - relaxation of follower
	#shiftloss - for when we are using the shifted end startegy for boundary
	#dim = 2 - dimension of model (first or second order DE)
	# h = .1 - timestep of data

    lead = lead[:,2:4] #get only the position and speed because that's what we care about
    #lead from leadinfo is on times t_n-t_nstar:T_nm1-t_nstar+1 (using t_nstar of follower, obv need leadt_nstar when getting it from leader)
    #so we are going to need to pad the lead trajectory with zeros for this to work
#    simtraj = simtraj[:,1:] this is when you use the output of euler
    simtraj = simtraj[:,2:4]
    #need the simtraj for t_n to T_n
    vehstar = vehstar[:,2:4]
    #need the vehstar for t_n to T_n
    curfol = curfol[:,2:4]

#    if h < 0: #get relevant frames for solution
    t = range(int(frames[1]),int(frames[0]-1),-1)  #time t is going backwards here
    #flipping everything should be pretty fast so don't need to worry about it too much
    lead = np.flip(lead,0)   #need to flip lead to match t since we go backwards in time for adjoint system
    simtraj = np.flip(simtraj,0)
    vehstar = np.flip(vehstar,0)
    leadlen = np.flip(leadlen,0)
    curfol = np.flip(curfol,0)
    inmodel = np.flip(inmodel,0)
    havefol = np.flip(havefol,0)
    lamp1 = np.flip(lamp1,0)
    relax = np.flip(relax)
    folrelax = np.flip(folrelax)
    folplist = np.flip(folplist)
#    else:
#        t = range(int(frames[0,0]),int(frames[0,1]+1))



#    a,b = lead.shape #note we got only the speed and position of lead, and lead is padded with zeros so this is ok
    a = int(frames[1]-frames[0]+1)
    lam = np.empty((a,dim+1)) #initialize solution

    lam[:,0] = t #input frames = times
    lam[0,[1,2]] = [0,0] #initial conditions



    for i in range(len(t)-1):
        lam[i+1,[1,2]] = lam[i,[1,2]] + h*func(simtraj[i,:],lead[i,:],p,leadlen[i],vehstar[i,:],lam[i,[1,2]], curfol[i,:], inmodel[i], havefol[i],
           vehlen, lamp1[i,:], pdict[folplist[i]], relax[i],folrelax[i],shiftloss)    #forward euler formula where func computes derivative
        #print(func(out[i,[1,2]],lead[i,:],p))
    lam = np.flip(lam,0)
    return lam

def euleradj2(p,frames, func,lead,leadlen, vehstar, simtraj, curfol, inmodel, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss, dim = 2, h = -.1,*args):
    #############
# old version with bugged folp it's here for debugging
    #############

    #p - parameters
	# frames - list of len 2, frames[0] is initial time of simulation and frames[1] last time
	#func - defines model
	#lead - lead trajectory
	#leadlen - length of leader
	#vehstar - measurements
	#simtraj - simulation
	#curfol - follower of vehicle
	#inmodel - regime of model, list
	#havefol - boolean records whether or not there is a leader
	#vehlen - length of vehicle
	#lamp1 - adjoint system of follower
	#folp - index of following vehicle at current time
	#relax - relaxation, list
	#folrelax - relaxation of follower
	#shiftloss - for when we are using the shifted end startegy for boundary
	#dim = 2 - dimension of model (first or second order DE)
	# h = .1 - timestep of data

    lead = lead[:,2:4] #get only the position and speed because that's what we care about
    #lead from leadinfo is on times t_n-t_nstar:T_nm1-t_nstar+1 (using t_nstar of follower, obv need leadt_nstar when getting it from leader)
    #so we are going to need to pad the lead trajectory with zeros for this to work
#    simtraj = simtraj[:,1:] this is when you use the output of euler
    simtraj = simtraj[:,2:4]
    #need the simtraj for t_n to T_n
    vehstar = vehstar[:,2:4]
    #need the vehstar for t_n to T_n
    curfol = curfol[:,2:4]

#    if h < 0: #get relevant frames for solution
    t = range(int(frames[1]),int(frames[0]-1),-1)  #time t is going backwards here
    #flipping everything should be pretty fast so don't need to worry about it too much
    lead = np.flip(lead,0)   #need to flip lead to match t since we go backwards in time for adjoint system
    simtraj = np.flip(simtraj,0)
    vehstar = np.flip(vehstar,0)
    leadlen = np.flip(leadlen,0)
    curfol = np.flip(curfol,0)
    inmodel = np.flip(inmodel,0)
    havefol = np.flip(havefol,0)
    lamp1 = np.flip(lamp1,0)
    relax = np.flip(relax)
    folrelax = np.flip(folrelax)
#    else:
#        t = range(int(frames[0,0]),int(frames[0,1]+1))



#    a,b = lead.shape #note we got only the speed and position of lead, and lead is padded with zeros so this is ok
    a = int(frames[1]-frames[0]+1)
    lam = np.empty((a,dim+1)) #initialize solution

    lam[:,0] = t #input frames = times
    lam[0,[1,2]] = [0,0] #initial conditions



    for i in range(len(t)-1):
        lam[i+1,[1,2]] = lam[i,[1,2]] + h*func(simtraj[i,:],lead[i,:],p,leadlen[i],vehstar[i,:],lam[i,[1,2]], curfol[i,:], inmodel[i], havefol[i],
           vehlen, lamp1[i,:], folp, relax[i],folrelax[i],shiftloss)    #forward euler formula where func computes derivative
        #print(func(out[i,[1,2]],lead[i,:],p))
    lam = np.flip(lam,0)
    return lam

def eulerdelay(p, frames, func, lead, leadlen, r, curmeas, dim=2, h=.1, *args):
	#does simulation by numerically integrating a DDE model

    delay = p[0], t_n = frames[0]
    T_n = curmeas[-1,1]
    datadelay = delay/h #delay is measured in seconds, in data we have frames with spacing h
    times = curmeas[:,1]
    pos = curmeas[:,2]
    speed = curmeas[:,3]

    timeslead = lead[:,1]
    poslead = lead[:,2]
    speedlead = lead[:,3]

    #when doing the simulation we need to interpolate:
    #the vehicle trajectory, the lead trajectory, as well as the relaxation amount and lead length.
    #note that interpolating both the lead trajectory and lead length is equivalent to interpolating the space headway
    posint = sci.interp1d(times,pos) #read as position interpolation
    speedint = sci.interp1d(times,speed) #speed interpolation
    posintlead = sci.interp1d(timeslead,poslead)
    speedintlead = sci.interp1d(timeslead,speedlead)
    leadlenint = sci.interp1d(timeslead,leadlen)
    relaxint = sci.interp1d(timeslead,r)


    IC = [posint(t_n+datadelay), speedint(t_n+datadelay)]

    #deprecated code block
#    if frames[1] == curinfo[3]: #no shifted end
#        simlen = math.floor((frames[1]-frames[0])*h)+1
#    else:
#        simlen = curinfo[2] - curinfo[1] + 1 #simulation is from t_n+delay to T_nm1 + delay so we have the ``normal" number of discrete simulated points
#        #which is T_nm1 - t_n + 1

    simlen = math.floor((frames[1]-frames[0])*h/delay)+1 #really this is the number of simulated points so the actual simtraj has a len of 1 + this (extra point is the initial point)
    simtimes = np.linspace(t_n+datadelay,t_n+simlen*datadelay,simlen+1)
    #this is what the times would be for a timestep that isn't equal to the delay
    #start = t_n+datadelay, end = min(T_n, T_nm1+datadelay)
    #simlen = math.floor((end-start)/dt)+1

    simtraj = np.empty((simlen,3))
    reg = np.empty(simlen)
    simtraj[0,1:] = IC
    simtraj[:,0] = simtimes
    reg = np.empty(simlen)
    #initialize
    delayt = t_n
    curt = t_n+datadelay
    prevlead = [posintlead(delayt), speedintlead(delayt)]
    prevveh = [posint(delayt), speedint(delayt)]
    curveh = IC
    prevrelax = relaxint(delayt)
    prevleadlen = leadlenint(delayt)

    if dim==2:
        for i in range(simlen):
            #do a timestep
            out, regime = func(curveh, prevveh, prevlead, prevleadlen, prevrelax)
            simtraj[i+1,[1,2]] = curveh + delay*out
            reg[i] = regime

            #update iteration
            delayt = simtimes[i]
            curt = simtimes[i+1]
            prevlead = [posintlead(delayt), speedintlead(delayt)]
            prevveh = curveh
            curveh = simtraj[i+1,[1,2]]
            prevrelax = relaxint(delayt)
            prevleadlen = leadlenint(delayt)

    elif dim ==1:
        for i in range(simlen):
            out, regime = func(p,curveh, prevveh, prevlead, prevleadlen, prevrelax)
            simtraj[i+1,1] = simtraj[i,1] + delay*out[0]
            simtraj[i,2] = out[0]
            reg[i] = regime

            delayt = simtimes[i]
            curt = simtimes[i+1]
            prevlead = [posintlead(delayt), speedintlead(delayt)]
            prevveh = curveh
            curveh = simtraj[i+1,[1,2]]
            prevrelax = relaxint(delayt)
            prevleadlen = leadlenint(delayt)

    return simtraj, reg



