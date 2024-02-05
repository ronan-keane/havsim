
"""
@author: rlk268@cornell.edu
"""

import time
import copy
import scipy.optimize as sc
import numpy as np

from havsim import helper
from havsim.old import opt

def calibrate_custom(plist,bounds,meas,platooninfo,platoonlist,makeleadfolinfo, custom, platoonobjfn, platoonobjfn_der, platoonobjfn_hess, model,modeladjsys, modeladj,
                     linesearch, kwargs, *args, cutoff=7.5, delay = False, dim = 2 ):
    #this can be used for any of my custom optimization programs; SPSA, pgrad_descent, and SQP. You can choose the keyword options for them, as well as pick
    #your desired linesearch (backtrack, weak/strong wolfe, watchdog, nonmonotone backtracking or nonmonotone wolfe) for pgrad_descent and SQP.
    #they have some other nifty features as well, like using barzilei borwein scaling and safeguarding the newton step.
    #In general, the quasi-newton methods
    #will probably give better performance for most problems though.

    sim = copy.deepcopy(meas)
    attempts = len(plist)

    out = []
    times = []
    rmse = []

    for i in platoonlist:
        counter = 0
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)

        start = time.time()
        bfgs = custom(opt.platoonobjfn_obj,platoonobjfn_der, platoonobjfn_hess, plist[counter],bounds,linesearch,
                      (model,modeladjsys,modeladj,meas,sim,platooninfo,i,leadinfo,folinfo,rinfo, *args), **kwargs)
        end = time.time()
        curtime = end-start

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim, delay = bfgs[0][0])

        while currmse>cutoff and counter<attempts-1:
            counter += 1

            start = time.time()
            rebfgs = custom(opt.platoonobjfn_obj,platoonobjfn_der, platoonobjfn_hess, plist[counter],bounds,linesearch,
                      (model,modeladjsys,modeladj,meas,sim,platooninfo,i,leadinfo,folinfo,rinfo, *args), **kwargs)
            end = time.time()
            curtime += end-start

            if rebfgs[1]<bfgs[1]:
                rebfgs[2]['gradeval'] += bfgs[2]['gradeval']
                rebfgs[2]['iter'] += bfgs[2]['iter']
                rebfgs[2]['objeval'] += bfgs[2]['objeval']
                bfgs = rebfgs
            else:
                bfgs[2]['gradeval'] += rebfgs[2]['gradeval']
                bfgs[2]['iter'] += rebfgs[2]['iter']
                bfgs[2]['objeval'] += rebfgs[2]['objeval']

            if not delay:
                currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim)
            else:
                currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim, delay = bfgs[0][0])
        bfgs = (*bfgs,counter+1)
        out.append(bfgs)
        times.append(curtime)
        rmse.append(currmse)

        for j in i[1:]: #reset simulation to measurements for next platoon
            sim[j] = meas[j].copy()

    return out, times, rmse

def calibrate_tnc(plist,bounds, meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn, platoonobjfn_der, model, modeladjsys, modeladj, *args, cutoff=7.5, order = 0, delay = False, dim = 2, objder = True):
    #call this to use the truncated newton method.
    #same call signature as bfgs, the difference is TNC has a funky system for how it returns the answers.


    #r is for relaxation
    #eventually want to be able to choose all the options from this function and have a wrapper where you pick the optimization algorithm
    #note that this isn't meant for calibrating platoons currently because of the way the rmse function works
    sim = copy.deepcopy(meas)
    attempts = len(plist)

    out = []
    times = []
    rmse = []

    for i in platoonlist:
        counter = 0
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)

        start = time.time()
        bfgs = sc.fmin_tnc(platoonobjfn,plist[counter],platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,bounds)
        end = time.time()
        curtime = end-start

        if objder: #TNC doesn't return the objective value at the returned point so we have to do it. #also note that the function usually returns both obj and grad
            obj, grad = platoonobjfn(bfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)
        else:
            obj = platoonobjfn(bfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)

        bfgs = list(bfgs) #need to convert to list for next step
        bfgs.append(obj) #append value
        bfgs[1] += 1 #add 1 to function count

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim, delay = bfgs[0][0])

        while currmse>cutoff and counter<attempts-1:
            counter += 1

            start = time.time()
            rebfgs = sc.fmin_tnc(platoonobjfn,plist[counter],platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,bounds)
            end = time.time()
            curtime += end-start

            if objder:
                obj, grad = platoonobjfn(rebfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)
            else:
                obj = platoonobjfn(rebfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)

            rebfgs = list(rebfgs) #see above
            rebfgs.append(obj)
            rebfgs[1] += 1

            if rebfgs[-1]<bfgs[-1]: #if new solution is better
                rebfgs[1] += bfgs[1] #update function evaluation count
                bfgs = rebfgs #this will load in new parameters and objective value
            else: #otherwise just update function evaluation count
                bfgs[1] += rebfgs[1]

        if not delay: #get rmse for best solution
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim, delay = bfgs[0][0])
        bfgs = (*bfgs,counter+1) #record number of times we did the optimization, and save in all the required values
        out.append(bfgs)
        times.append(curtime)
        rmse.append(currmse)

        if order ==0:
            for j in i[1:]: #reset simulation to measurements for next platoon. you can change this part as desired for the desired calibration strategy.
                sim[j] = meas[j].copy()

    return out, times, rmse

def calibrate_tnc2(plist,bounds, meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn, platoonobjfn_der, model, modeladjsys, modeladj, *args,
                   cutoff=7.5, cutoff2 = 4.5, order = 0, delay = False, dim = 2, budget = 1,reguess = True,objder=True, **kwargs):
    #more up to date version which uses some heuristics to calibrate platoons efficiently

    #DOCUMENTATION
    #plist - list where entries are lists of parameters representing initial guesses for a single vehicle (m parameters per guess/vehicle, multiple guesses can be provided)
    #bounds - list of tuples where each tuple is length two representing the lower and upper bounds for that parameter  (m tuples each of length 2)
    #platooninfo - same as everywhere else
    #platoonlist - list of platoons which will be calibrated in the order provided
    #makeleadfolinfo - function which makes the leader, follower, and relaxation information
    #platoonobjfn - main function for objective/gradient
    #platoonobjfn_der - secondary function for objective/gradient, if platoonobjfn is platoonobjfn_objder then platoonobjfn_der can be None
    #model
    #modeladjsys
    #modeladj
    #*args - extra arguments are passed into platoonobjfn
    #cutoff = 7.5 - calibration is repeated if rmse is above this number
    #cutoff2 = 4.5 - for a calibration of platoons, an individual vehicle's calibration is repeated if its rmse is above this number
    #order = 0 - if order is 0, the calibrated trajectory 'sim' will be reset to be equal to meas after the calibration. this represents the case where
    #calibraiton is done to the measurements always, so the order is '0' because the calibraiton can be done in parallel.
    #delay = False - whether or not model has time delay
    #dim = 2 - order of model (second order = model gives an acceleration)
    #budget = 1 - number of guesses to use
    #reguess = True - if reguess is False, then when we do another optimization run vehicles which are below cutoff (which would repeat their previous guess)
    #stay at their optimized parameter values.  If true, vehicles will return to whichever guess was just used for them.
    #objder = True - if True, platoonobjfn will return both objective and gradient, otherwise it returns only objective. platoonobjfn_der always returns only gradient.
    #**kwargs - keyword arguments for the optimization algorithm


    #r is for relaxation
    #eventually want to be able to choose all the options from this function and have a wrapper where you pick the optimization algorithm
    #note that this isn't meant for calibrating platoons currently because of the way the rmse function works
    sim = copy.deepcopy(meas)
#    attempts = len(plist)

    out = []
    times = []
    rmse = []
    m = args[1] #number of parameters for each vehicle
    nguesses = len(plist) #number of guesses provided


    for i in platoonlist:
        counter = 0
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)
        #first new part is here - need to adjust the length of bounds and length of parameters based on the size of the platoon
        nveh = len(i)

        curp = np.tile(plist[counter],(nveh,1)) #first guess is always the provided guess for all vehicles.
        usebounds = np.tile(bounds,(nveh,1))

        countguesses = [1 for j in range(nveh)] #each vehicle will be tried with the first guess


        start = time.time()
        bfgs = sc.fmin_tnc(platoonobjfn,curp,platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,usebounds, **kwargs)
        end = time.time()
        curtime = end-start
        #note after doing the optimization we actually need to evaluate the objective when using the tnc algorithm because we need the sim and objective function value, not just the parameters
        if objder: #TNC doesn't return the objective value at the returned point so we have to do it. #also note that the function usually returns both obj and grad
            obj, grad = platoonobjfn(bfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)
        else:
            obj = platoonobjfn(bfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)

        #depending on the algorithm you may or may not want to reevaluate the objective at this point. If the linesearch fails, the simulation recorded in sim does not necessarily correspond
        #to the solution that the algorithm reports. l-bfgs-b has this problem, where if the linesearch fails the reported objective (and what is in sim) is much worse than
        #what is actually achieved by the reported parameters.
        #I don't know whether or not TNC has this problem, but in calibrate_tnc the objective is always recomputed so the issue can never arise.

        #next new part is here. we will get individual
        obj = helper.SEobj_pervehicle(meas,sim,platooninfo,i) #list of (individual) objective functions. This is needed for next step
        rmselist = []
        for j in range(nveh): #convert each individual objective to individual rmse
            temp = helper.convert_to_rmse(obj[j],platooninfo,[i[j]])
            rmselist.append(temp)

        bfgs = list(bfgs) #need to convert to list for next step
        bfgs.append(sum(obj)) #append value
#        bfgs[1] += 1 #add 1 to function count

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim, delay = bfgs[0][0])

        while currmse>cutoff and counter<budget-1:
            counter += 1
            #this is how we are going to handle the guesses for an arbitrarily sized platoon of vehicles

            psame = [True for j in range(nveh)] #are the parameter guesses the same as the one just used

            for j in range(nveh): #update guesses based on individual rmse
                if rmselist[j] > cutoff2: #individual rmse is high enough that we would like to try a new guess for the vehicle
                    psame[j] = False #false because this vehicle is having its guess changed
                    if countguesses[j] < nguesses: #if still have provided guesses to try
                        curp[m*j:m*(j+1)] = plist[countguesses[j]] #then we can try that guess
                        countguesses[j] += 1 #remember that we've tried this new guess
                    else: #otherwise we need to generate a new guess
                        newp = []
                        for z in range(nveh): #generate new guess
                            newp.append(np.random.uniform(bounds[z][0],bounds[z][1])) #uniform random inside box bounds
                        curp[m*j:m*(j+1)] = newp
                elif not reguess: #set to the optimized value if reguess is false and we are below the cutoff
                    curp[m*j:m*(j+1)] = bfgs[0][m*j:m*(j+1)]
            while np.all(psame): #if no changes to guess happened we will force some changes.
                for j in range(nveh):
                    randn = np.random.uniform()
                    if randn<.5:  #this is the random condition for forcing a new guess potential for modifying this; for example weight higher rmse more strongly to try new guess?
                        #if the condition is fulfilled, we update the guesses in exactly the same way as above. we keep iterating in the while loop until at least one guess is updated.
                        if countguesses[j] < nguesses: #if still have provided guesses to try
                            curp[m*j:m*(j+1)] = plist[countguesses[j]] #then we can try that guess
                            countguesses[j] += 1 #remember that we've tried this new guess
                        else: #otherwise we need to generate a new guess
                            newp = []
                            for z in range(m): #generate new guess
                                newp.append(np.random.uniform(bounds[z][0],bounds[z][1]))
                            curp[m*j:m*(j+1)] = newp
                        psame[j] = False
            #it's arguably slightly weird that we are choosing the new guesses randomly. The idea is we don't necessarily want to keep forcing new guesses
            #for vehicles if already have very good fits, but we also need something to handle the situation when everything is below cutoff2.
            #if you don't like this behavior then just put cutoff2 is 0 so all guesses will always be updated.

            start = time.time()
            rebfgs = sc.fmin_tnc(platoonobjfn,curp,platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,usebounds, **kwargs)
            end = time.time()
            curtime += end-start

            if objder:
                obj, grad = platoonobjfn(rebfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)
            else:
                obj = platoonobjfn(rebfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)

            obj = helper.SEobj_pervehicle(meas,sim,platooninfo,i)
            rmselist = []
            for j in range(nveh): #convert each individual objective to individual rmse
                temp = helper.convert_to_rmse(obj[j],platooninfo,[i[j]])
                rmselist.append(temp)

            rebfgs = list(rebfgs) #see above
            rebfgs.append(sum(obj))
#            rebfgs[1] += 1

            if rebfgs[-1]<bfgs[-1]: #if new solution is better
                rebfgs[1] += bfgs[1] #update function evaluation count
                bfgs = rebfgs #this will load in new parameters and objective value
            else: #otherwise just update function evaluation count
                bfgs[1] += rebfgs[1]

        if not delay: #get rmse for best solution
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[-1],platooninfo,i,dim=dim, delay = bfgs[0][0])
        bfgs = (*bfgs,counter+1) #record number of times we did the optimization, and save in all the required values
        out.append(bfgs)
        times.append(curtime)
        rmse.append(currmse)

        if order ==0:
            for j in i: #reset simulation to measurements for next platoon. you can change this part as desired for the desired calibration strategy.
                sim[j] = meas[j].copy()
        elif order ==1: #need to make sure simulation loaded in is the best one found
            if objder:
                obj, grad = platoonobjfn(bfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)
            else:
                obj = platoonobjfn(bfgs[0],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)

    return out, times, rmse

def calibrate_bfgs(plist,bounds, meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn, platoonobjfn_der, model, modeladjsys, modeladj, *args,
                   cutoff=7.5, order = 0, delay = False, dim = 2):
    #this is going to become the main function to call for calibrating using bfgs.


    #r is for relaxation
    #eventually want to be able to choose all the options from this function and have a wrapper where you pick the optimization algorithm
    #note that this isn't meant for calibrating platoons currently because of the way the rmse function works
    sim = copy.deepcopy(meas)
    attempts = len(plist)

    out = []
    times = []
    rmse = []

    for i in platoonlist:
        counter = 0
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)

        start = time.time()
        bfgs = sc.fmin_l_bfgs_b(platoonobjfn,plist[counter],platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,bounds)
        end = time.time()
        curtime = end-start

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim, delay = bfgs[0][0])

        while currmse>cutoff and counter<attempts-1:
            counter += 1

            start = time.time()
            rebfgs = sc.fmin_l_bfgs_b(platoonobjfn,plist[counter],platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,bounds)
            end = time.time()
            curtime += end-start

            if rebfgs[1]<bfgs[1]:
                rebfgs[2]['funcalls'] += bfgs[2]['funcalls']
                rebfgs[2]['nit'] += bfgs[2]['nit']
                bfgs = rebfgs
            else:
                bfgs[2]['funcalls'] += rebfgs[2]['funcalls']
                bfgs[2]['nit'] += rebfgs[2]['nit']

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim, delay = bfgs[0][0])
        bfgs = (*bfgs,counter+1)
        out.append(bfgs)
        times.append(curtime)
        rmse.append(currmse)

        if order ==0:
            for j in i[1:]: #reset simulation to measurements for next platoon
                sim[j] = meas[j].copy()

    return out, times, rmse

def calibrate_bfgs2(plist,bounds, meas,platooninfo,platoonlist,makeleadfolinfo,platoonobjfn, platoonobjfn_der, model, modeladjsys, modeladj, *args,
                   cutoff=7.5, cutoff2 = 4.5, order = 0, delay = False, dim = 2, budget = 1, **kwargs):
    #more up to date version which uses some heuristics to calibrate platoons efficiently

    #inputs
    #plist - list of parameters, where each entry is an initial guess for the parameters of the model (so it's a list of lists, because an initial guess for parameters
    #is a list of numbers)
    #bounds - list of tuples, where each tuple are the lower and upper bounds of that particular parameter
    #meas - dictionary where the key is the vehicle ID and value is a matrix of the data. see loadngsim for the exact specifications of the data. each row is corresponding
    #to an observation and there are multiple columns for the data recorded at each observation.
    #platooninfo - dictionary where key is the vehicle ID and value is a list which contains information about the vehicle's trajectory
    #makeleadfolinfo - you need to choose what options you want for the relaxation phenomenon (if any) and you do this by passing in one of the makeleadfolinfo functions.
    #platoonobjfn - returns either objective, or objective and gradient
    #platoonobjfn_der - returns only gradient, or it can be None in which case platoonobjfn needs to return both objective and gradient
    #model - function name for the model used
    #modeladjsys - function name for adjoint system of the model
    #model adj - function name for the function which computes the gradient after the adjoint system is computed
    #*args - these are extra arguments passed to the platoonobjfn and platoonobjfn_der functions
    #cutoff = 7.5 - calibration is repeated if rmse is above this number
    #cutoff2 = 4.5 - for a calibration of platoons, an individual vehicle's calibration is repeated if its rmse is above this number
    #order = 0 - if order is 0, the calibrated trajectory 'sim' will be reset to be equal to meas after the calibration. this represents the case where
    #calibraiton is done to the measurements always, so the order is '0' because the calibraiton can be done in parallel.
    #delay = False - whether or not model has time delay
    #dim = 2 - order of model (second order = model gives an acceleration)
    #budget = 1 - number of guesses to use
    #**kwargs - keyword arguments for the optimization algorithm

    #this is going to become the main function to call for calibrating using bfgs.


    #r is for relaxation
    #eventually want to be able to choose all the options from this function and have a wrapper where you pick the optimization algorithm
    #this is for calibrating an arbitrary model using bfgs algorithm. calibration of platoons is handled by updating guesses where we try up to budget total guesses
    #and if an individual vehicle's RMSE is below cutoff2 we won't update it.

    sim = copy.deepcopy(meas)
    attempts = len(plist)

    out = []
    times = []
    rmse = []
    m = args[1]
    nguesses = len(plist)

    for i in platoonlist:
        counter = 0
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)

        nveh = len(i)-1

        curp = np.tile(plist[counter],nveh)
        usebounds = np.tile(bounds,(nveh,1))

        countguesses = [1 for j in range(nveh)]

        maxfun = max(200,20*m*nveh) #going to put a cap on bfgs iterations. CG has a cap which is half of this.

        start = time.time()
        bfgs = sc.fmin_l_bfgs_b(platoonobjfn,curp,platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,usebounds, maxfun = maxfun,**kwargs)
        end = time.time()
        curtime = end-start

        obj = helper.SEobj_pervehicle(meas,sim,platooninfo,i) #list of (individual) objective functions. This is needed for next step
        rmselist = []
        for j in range(nveh): #convert each individual objective to individual rmse
            temp = helper.convert_to_rmse(obj[j],platooninfo,[[],i[j+1]])
            rmselist.append(temp)

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim, delay = bfgs[0][0])

        while currmse>cutoff and counter<budget-1:
            counter += 1

            oldp = curp.copy() #the guess we just used

            for j in range(nveh): #update guesses based on individual rmse
                if rmselist[j] > cutoff2: #individual rmse is high enough that we would like to try a new guess for the vehicle
                    if countguesses[j] < nguesses: #if still have provided guesses to try
                        curp[m*j:m*(j+1)] = plist[countguesses[j]] #then we can try that guess
                        countguesses[j] += 1 #remember that we've tried this new guess
                    else: #otherwise we need to generate a new guess
                        newp = []
                        for z in range(nveh): #generate new guess
                            newp.append(np.random.uniform(bounds[z][0],bounds[z][1])) #uniform random inside box bounds
                        curp[m*j:m*(j+1)] = newp
            while np.all(np.array(curp)==np.array(oldp)): #if no changes to guess happened we will force some changes.
                for j in range(nveh):
                    randn = np.random.uniform()
                    if randn<.5:  #this is the random condition for forcing a new guess ; potential for modifying this; for example weight higher rmse more strongly to try new guess?
                        #if the condition is fulfilled, we update the guesses in exactly the same way as above. we keep iterating in the while loop until at least one guess is updated.
                        if countguesses[j] < nguesses: #if still have provided guesses to try
                            curp[m*j:m*(j+1)] = plist[countguesses[j]] #then we can try that guess
                            countguesses[j] += 1 #remember that we've tried this new guess
                        else: #otherwise we need to generate a new guess
                            newp = []
                            for z in range(nveh): #generate new guess
                                newp.append(np.random.uniform(bounds[z][0],bounds[z][1]))
                            curp[m*j:m*(j+1)] = newp
            #it's arguably slightly weird that we are choosing the new guesses randomly. The idea is we don't necessarily want to keep forcing new guesses
            #for vehicles if already have very good fits, but we also need something to handle the situation when everything is below cutoff2.
            #if you don't like this behavior then just put cutoff2 is 0 so all guesses will always be updated.



            start = time.time()
            rebfgs = sc.fmin_l_bfgs_b(platoonobjfn,curp,platoonobjfn_der,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),0,usebounds, maxfun = maxfun, **kwargs)
            end = time.time()
            curtime += end-start

            obj = helper.SEobj_pervehicle(meas,sim,platooninfo,i)
            rmselist = []
            for j in range(nveh): #convert each individual objective to individual rmse
                temp = helper.convert_to_rmse(obj[j],platooninfo,[[],i[j+1]])
                rmselist.append(temp)

            if rebfgs[1]<bfgs[1]:
                rebfgs[2]['funcalls'] += bfgs[2]['funcalls']
                rebfgs[2]['nit'] += bfgs[2]['nit']
                bfgs = rebfgs
            else:
                bfgs[2]['funcalls'] += rebfgs[2]['funcalls']
                bfgs[2]['nit'] += rebfgs[2]['nit']

        if not delay:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim)
        else:
            currmse = helper.convert_to_rmse(bfgs[1],platooninfo,i,dim=dim, delay = bfgs[0][0])
        bfgs = (*bfgs,counter+1)
        out.append(bfgs)
        times.append(curtime)
        rmse.append(currmse)

        if order ==0:
            for j in i[1:]: #reset simulation to measurements for next platoon
                sim[j] = meas[j].copy()

    return out, times, rmse

def calibrate_nlopt_helper(alg, pguess, bounds, meas, sim, platooninfo, platoon, makeleadfolinfo, platoonobjfn, platoonobjfn_der,
                           model, modeladjsys, modeladj, *args, evalper = 20 ):
    #this is going to take an specified platoon and calibrate it using one of the NLopt algorithms.

    #refer to calibrate_bfgs2 for the most up to date documentation of what all the parameters are.
    #note that nlopt doesn't support grad and obj being returned at the same time (this is not ideal for using automatic differentiation or the adjoint method because its slower)
    #so this means that platoonobjfn_der needs to only return the grad.

    N = len(pguess) #total number of parameters
    m = args[1] #number of parameters per vehicle
    opt = nlopt.opt(alg,N)
    lb = []; ub = [];
    for i in bounds:
        lb.append(i[0]); ub.append(i[1])

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    if evalper is not None:  #option to set max function evaluations

        maxfun = max(200,evalper*N) #
        opt.set_maxeval(maxfun)

    leadinfo, folinfo, rinfo = makeleadfolinfo(platoon, platooninfo, sim) #note that this needs to be done with sim and not meas

    count = 0
    countgrad = 0

    def nlopt_fun(p, grad):
        nonlocal count
        nonlocal countgrad
        if len(grad) > 0 :
            newgrad = platoonobjfn_der(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoon, leadinfo, folinfo, rinfo, *args)
            grad[:] = newgrad
            countgrad += 1
            return grad
        obj = platoonobjfn(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoon, leadinfo, folinfo, rinfo, *args)
        count += 1
        return obj

    opt.set_min_objective(nlopt_fun)

    ans = opt.optimize(pguess) #returns number of parameters

    return ans, count, countgrad

def calibrate_nlopt(alg, pguess, mybounds, meas, platooninfo, platoonlist, makeleadfolinfo, platoonobjfn, platoonobjfn_der,
                           model, modeladjsys, modeladj, *args, evalper = 20, order = 0, dim = 2):
    #calibrate using nlopt. note that some of the nlopt algs currently don't work. unsure why.
    #also this is not going to work for models with delay ATM so no newell!

    out = []
    times = []
    rmse = []
    sim = copy.deepcopy(meas)

    for i in platoonlist:
        n = len(i)-1
        m = args[1]
        p = np.tile(pguess, n)
        bounds = np.tile(mybounds,(n,1))

        start = time.time()
        ans, count, countgrad = calibrate_nlopt_helper(alg, p, bounds, meas, sim, platooninfo, i, makeleadfolinfo, platoonobjfn, platoonobjfn_der,
                           model, modeladjsys, modeladj, *args, evalper = evalper )
        end = time.time()
        dt = end-start

        times.append(dt)
        output = [ans]
        output.append(count); output.append(countgrad)

        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,meas)
        obj = platoonobjfn(ans,model,modeladjsys,modeladj,meas,sim,platooninfo,i,leadinfo,folinfo,rinfo,*args)
        output.append(obj)

        currmse = helper.convert_to_rmse(obj,platooninfo,i,dim=dim)
        rmse.append(currmse)
        out.append(output)

        if order == 0: #if order is 0 we reset the simulation to measurements, pass in anything else and calibratino is done sequentially in order given.
            for j in i[1:]:
                sim[j] = meas[j].copy()

    return out, times, rmse


def calibrate_GA(bounds, meas,platooninfo,platoonlist,makeleadfolinfo, platoonobjfn,platoonobjfn_der,model,modeladjsys,modeladj, *args, order = 0, **kwargs):
    #eventually want to be able to choose all the options from this function and have a wrapper where you pick the optimization algorithm
    #note that this isn't meant for calibrating platoons currently because of the way the rmse function works
    sim = copy.deepcopy(meas)

    out = []
    times = []
    rmse = []

    for i in platoonlist:
        nveh = len(i)
        usebounds = np.tile(bounds,(nveh,1))
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)

        start = time.time()
        GA = sc.differential_evolution(platoonobjfn,usebounds,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo, *args), **kwargs)
        end = time.time()
        curtime = end-start

        currmse = helper.convert_to_rmse(GA['fun'],platooninfo,i)
        out.append(GA)
        times.append(curtime)
        rmse.append(currmse)

        if order ==0:
            for j in i: #reset simulation to measurements for next platoon
                sim[j] = meas[j].copy()
        elif order ==1:
            obj = platoonobjfn(GA['x'],model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args)

    return out, times, rmse

def calibrate_NM(p,meas,platooninfo,platoonlist,makeleadfolinfo, platoonobjfn,platoonobjfn_der,model,modeladjsys,modeladj, *args, **kwargs):
    #this can only do 1 guess at a time currently
    sim = copy.deepcopy(meas)

    out = []
    times = []
    rmse = []

    for i in platoonlist:
        leadinfo,folinfo,rinfo = makeleadfolinfo(i,platooninfo,sim)

        start = time.time()
        NM = sc.minimize(platoonobjfn,p,(model, modeladjsys, modeladj, meas, sim, platooninfo, i, leadinfo, folinfo,rinfo,*args),'Nelder-Mead',options = {'maxfev':10000})
        end = time.time()
        curtime = end-start

        currmse = helper.convert_to_rmse(NM['fun'],platooninfo,i)
        out.append(NM)
        times.append(curtime)
        rmse.append(currmse)

        for j in i[1:]: #reset simulation to measurements for next platoon
            sim[j] = meas[j].copy()

    return out, times, rmse

def calibrate_check_realistic(meas,platooninfo,platoonlist,makeleadfolinfo,plist,platoonobjfn,model,modeladjsys,modeladj,*args):
    #goes through results of calibration on a dataset and returns whether each calibrated trajectory is realistic or not.
    sim = copy.deepcopy(meas)

    real = []

    for i in range(len(platoonlist)):
        curplatoon = platoonlist[i]
        curp = plist[i]

        leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,meas)

        obj = platoonobjfn(curp,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo,*args)

#        else:
#            obj = platoonobjfn_obj(curp,model, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)

        curreal = helper.check_realistic(sim[curplatoon[1]],platooninfo[curplatoon[1]])

        real.append(curreal)

        sim[curplatoon[1]] = meas[curplatoon[1]].copy()


    return real