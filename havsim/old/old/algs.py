
"""
calibration related functions which define algorithms (such as custom optimization algorithms or platoon formation algorithms)
"""
import numpy as np
import scipy.stats as ss
import havsim.helper as helper


def SPSA_grad(p,objfn, *args,q = 1, ck = 1e-8, **kwargs):
    #defines the SPSA gradient approximation. This can be used in place of a gradient function in any optimization algorithm; it is suggested to be used in a gradient descent algorithm with
    #a fixed step length (see spall 1992)
    #each gradient approximation uses 2 objective evaluations. There are q gradient approximations total, and it returns the average gradient.
    #therefore the functino uses 2q total objective evaluations; finite differences uses n+1 (n) evaluations, where n is the number of parameters. #(adjoint would use 2 (1) ), where
    #you can possibly save 1 evaluation if you pass in the current objective evaluation (which is not done in this code).
    #variable names follow the convention gave in spall 1992; except we call the c_k eps instead
    n  = len(p)
    grad = np.zeros((q,n)) #q rows of gradient with n columns
    eps = ck #numerical stepsize
    p = np.asarray(p) #need parameters to be an np array here
    for i in range(q): #
        delta = 2*ss.bernoulli.rvs(.5,size=n)-1 #symmetric bernoulli variables
        pp = p + eps*delta #reads as p plus
        pm = p - eps*delta # p minus
        yp = objfn(pp,*args,**kwargs)
        ym = objfn(pm,*args,**kwargs)
        grad[i,:] = (yp-ym)/(2*eps*delta) #definition of the SPSA to the gradient

    grad = np.mean(grad,0) #average across the gradients

    return grad



def pgrad_descent(fnc, fnc_der, fnc_hess, p, bounds, linesearch, args, t = 0,  eps = 1e-5, epsf = 1e-5, maxit = 1e3, der_only = False, BBlow = 1e-9, BBhi = 1,srch_type = 0,proj_type = 0,**kwargs):
    #minimize a scalar function with bounds using gradient descent
    #fnc - objective
    #fnc_der - derivative or objective and derivative
    #fnc_hess - hessian (unused)
    #p - initial guess
    #linesearch - linesearch function
    #bounds - box bounds for p
    #args - arguments that are passed to fnc, fnc_der, fnc_hess

    #t = 0 - if 1 , we will use a non-monotone line search strategy, using the linesearch function to determine sufficient decrease
    #otherwise, if t = 0 we will just use the linesearch function.

    #eps = 1e-5 - termination if relative improvement is less than eps
    #epsf =1e-5 - termination if gradient norm is less than epsf
    #maxit = 1e3 - termination if iterations are more than maxit
    #der_only indicates fnc_der only gives derivative
    #kwargs - any special arguments for linesearch need to be in kwargs

    #srch_type = 0 - scale search direction either by norm of gradient (0), using barzilai borwein scaling (1), or no scaling (any other value)
    #what scaling works best depends on the problem. for example, BB scaling works very well for the rosenbrock function.
    #for the car-following calibration, BB scaling seems to work poorly (I think because you often take small steps? )
    #in other problems, the scaling by norm of the gradient helps; for car following calibration it doesnt seem to help.

    #proj_type = 0 - either project before the linesearch (0), or project in each step of the line search (1)
    #note that the fixedstep linesearch should use proj_type = 1; nonmonotone uses projtype = 0. backtrack and weakwolfe can use either.
    #which projection type works better depends on the problem.

    #in general, you would expect that srch_type = 1, proj_type = 1 would work the best (BB scaling with projection at every linesearch step).
    if der_only:
        def fnc_objder(p, *args):
            obj = fnc(p, *args)
            grad = fnc_der(p,*args)

            return obj, grad
    else:
        fnc_objder = fnc_der

    obj, grad = fnc_objder(p, *args) #objective and gradient
    n_grad = np.linalg.norm(grad) #norm of gradient
    diff = 1 #checks reduction in objective (termination)
    iters = 1 #number of iterations (termination)
    totobjeval = 1 #keeps track of total objective and gradient evaluations
    totgradeval = 1

    if t != 0:
        watchdogls = linesearch
        linesearch = watchdog
#        ########compute the search direction in this case.....################ #actually don't do this
#        temp = grad
#        if srch_type ==0:
#            temp = temp/n_grad
#        if proj_type ==0: #each project first, then you won't have to project during line search
#            d = projection(p-temp,bounds)-p #search direction for the projected gradient
#        else:
#            d = -temp #search directino without projection
#        #########################################################################
        past = [[p, obj, grad],t+1] #initialize the past iterates for monotone
    else:
        watchdogls = None
        past = [None] #past will remain None unless we are doing the nonmonotone line search

    s = [1] #initialize BB scaling
    y = [1]

    while diff > eps and n_grad > epsf and iters < maxit:
#        print(obj)
        #do the scaling type; either scale by norm of gradient, using BB scaling, or no scaling
        temp = grad
#        if srch_type ==0 or iters ==1:  #scale by norm of gradient
        if srch_type ==0:
            temp = temp/n_grad
        elif srch_type ==1:  #BB scaling
            BBscaling = np.matmul(s,y)/np.matmul(y,y) #one possible scaling
#            BBscaling = np.matmul(s,s)/np.matmul(s,y) #this is the other possible scaling you can use
#            if np.isnan(BBscaling): #don't need this
#                BBscaling = 1/n_grad
#            print(BBscaling)
            if BBscaling < 0:
                BBscaling = BBhi
            elif BBscaling < BBlow:
                BBscaling = BBlow
            elif BBscaling > BBhi:
                BBscaling = BBhi
            temp = BBscaling*temp
        #otherwise, there will be no scaling and the search direction will simply be -grad

        if proj_type ==0: #each project first, then you won't have to project during line search
            d = projection(p-temp,bounds)-p #search direction for the projected gradient
        else:
            d = -temp #search directino without projection

        if past[-1] == 0: #in this case we need to remember the search direction of the iterate; if past[-1] == 0 it means we might have to return to that point in the non monotone search.
            past[0].append(d) #append search direction corresponding to the iterate
            if past[0][2][0] == None: #depending on what watchdogls is, we  may need to update the current gradient as well.
                past[0][2] = grad

#        dirder = np.matmul(grad,d) #directional derivative
        pn, objn, gradn, hessn, objeval, gradeval = linesearch(p,d,obj,fnc,fnc_objder,grad, args, iters, bounds, past, watchdogls, proj_type = proj_type, t = t, **kwargs)

        if gradn[0] == None: #if need to get new gradient
            objn, gradn = fnc_objder(pn,*args)
            totobjeval += 1
            totgradeval +=1

        if srch_type ==1:
            s = pn-p #definition of s and y for barzilai borwein scaling
            y = gradn-grad

        #update iterations and current values
        iters += 1
        totobjeval += objeval
        totgradeval += gradeval

        diff = abs(obj-objn)/obj #relative reduction in objective
        p = pn
        obj = objn
        grad = gradn
        n_grad = np.linalg.norm(grad)
    #report reason for termination
    if n_grad <= epsf:
        exitmessage = 'Norm of gradient reduced to '+str(n_grad)
    elif diff <= eps:
        exitmessage = 'Relative reduction in objective is '+str(diff)
    elif iters == maxit:
        exitmessage = 'Reached '+str(maxit)+' iterations'

    if past[-1] != None: #in the case we did a non monotone search we have some special termination conditions (these take place inside watchdog function) and need an extra step to report the solution
        if past[0][1] < obj:
            obj = past[0][1]
            p = past[0][0]
    out = []
    outdict = {}
    out.append(p)
    out.append(obj)
    outdict['grad'] = grad
    outdict['task'] = exitmessage
    outdict['gradeval'] = totgradeval
    outdict['objeval'] = totobjeval
    outdict['iter'] = iters
    out.append(outdict)

    return out

def pgrad_descent2(fnc, fnc_der, fnc_hess, p, bounds, linesearch, args, t = 0, eps = 1e-5, epsf = 1e-5,
                   maxit = 1e3, der_only = False, BBlow = 1e-9, BBhi = 1, srch_type = 0,proj_type = 0,**kwargs):
    #this can be used for nmbacktrack. Otherwise you can use watchdog which is a different nonmonotone strategy that is called with pgrad_descent. This combined with nmbactrack typically works the best.
    #minimize a scalar function with bounds using gradient descent
    #fnc - objective
    #fnc_der - derivative or objective and derivative
    #fnc_hess - hessian (unused)
    #p - initial guess
    #linesearch - linesearch function
    #bounds - box bounds for p
    #args - arguments that are passed to fnc, fnc_der, fnc_hess

    #t = 0 - if 1 , we will use a non-monotone line search strategy, using the linesearch function to determine sufficient decrease
    #otherwise, if t = 0 we will just use the linesearch function.

    #eps = 1e-5 - termination if relative improvement is less than eps
    #epsf =1e-5 - termination if gradient norm is less than epsf
    #maxit = 1e3 - termination if iterations are more than maxit
    #der_only indicates fnc_der only gives derivative
    #BBlow, BBhi - lower and upper bounds for the stepsize given by BB scaling.
    #kwargs - any special arguments for linesearch need to be in kwargs (adjust the parameters of the linesearch)

    #srch_type = 0 - scale search direction either by norm of gradient (0), using barzilai borwein scaling (1), or no scaling (any other value)
    #in general, BB scaling will work best, but you may need to adjust the safeguarding parameters; calibration seems to need BBlow very small, at least for the current loss function.
    #note there are two different types of BB scaling, <s,y>/<y,y> or <s,s>/<s,y>, I think the first works better in this case.

    #proj_type = 0 - either project before the linesearch (0), or project in each step of the line search (1)
    #note that the fixedstep linesearch should use proj_type = 1; nonmonotone uses projtype = 0. backtrack and weakwolfe can use either.
    #which projection type works better depends on the problem.

    #in general, you would expect that srch_type = 1, proj_type = 1 would work the best (BB scaling with projection at every linesearch step).

    ############overview of different algorithms##############

    #linesearches -
    #fixed step : choose a constant step length that decreases with some power of the iterations
    #backtrack2 : choose a step length based on backtracking armijo linesearch with safeguarded interpolation. requires only objective evaluations
    #weakwolfe2: can choose a step length that satisfies either strong or weak wolfe conditions, uses interpolation and safeguarding. Requires both gradient and objective evaluations
    #two different nonmonotone searches; each of which can be based around either of the above two line searches (explained more below)

    #algorithms -
    #pgrad_descent: can input keyword parameter t = n to use nonmonotone linesearch watchdog, which will take up to n relaxed steps before using some linesearch to enforce sufficient decrease
    #pgrad_descent2: can call both nmbacktrack, and nmweakwolfe. These are nonmonotone linesearches for the corresponding program, which relax the sufficient decrease condition
    #by only enforcing the decrease w.r.t the maximum of the past t = n iterations.
    #in general, I have found pgrad_descent2 with nmbacktrack to work the best for the calibration problem.
    #using watchdog (pgrad_descent with t \neq 0) seems to be worse than nmbacktrack, but better than nmweakwofe. So you can do watchdog for wolfe linesearch, otherwise use nmbacktrack.

    #for any of the gradient descent algorithms, you definitly want to use srch_type = 1 to use BB scaling. You may have to adjust the safeguarding parameters to achieve good results.
    #proj_type = 0 is the default. Sometimes you may find proj_type = 1 to work better, but in general proj_type = 0 (project the search direction only once) is much better.
    ################################################################

    if der_only:
        def fnc_objder(p, *args):
            obj = fnc(p, *args)
            grad = fnc_der(p,*args)

            return obj, grad
    else:
        fnc_objder = fnc_der

    obj, grad = fnc_objder(p, *args) #objective and gradient
    n_grad = np.linalg.norm(grad) #norm of gradient
    diff = 1 #checks reduction in objective (termination)
    iters = 1 #number of iterations (termination)
    totobjeval = 1 #keeps track of total objective and gradient evaluations
    totgradeval = 1

    #####################deprecated section from pgrad_descent#############
#    if t != 0:
#        watchdogls = linesearch
#        linesearch = watchdog
##        ########compute the search direction in this case.....################ #actually don't do this
##        temp = grad
##        if srch_type ==0:
##            temp = temp/n_grad
##        if proj_type ==0: #each project first, then you won't have to project during line search
##            d = projection(p-temp,bounds)-p #search direction for the projected gradient
##        else:
##            d = -temp #search directino without projection
##        #########################################################################
#        past = [[p, obj, grad],0] #initialize the past iterates for monotone
#    else:
#        watchdogls = None
#        past = [None] #past will remain None unless we are doing the nonmonotone line search
    #####################################

    past = [obj] #this is how we initialize the past. In this code, we will have special functions to handle the non-monotone,
    pastp = [p] #whereas before we were using watchdog which relied on some existing line search.


    s = [1] #initialize BB scaling
    y = [1]

    while diff > eps and n_grad > epsf and iters < maxit:
#        print(n_grad)
        #do the scaling type; either scale by norm of gradient, using BB scaling, or no scaling
        temp = grad
#        if srch_type ==0 or iters ==1:  #scale by norm of gradient
        if srch_type ==0:
            temp = temp/n_grad
        elif srch_type ==1:  #BB scaling
            BBscaling = np.matmul(s,y)/np.matmul(y,y) #one possible scaling
#            BBscaling = np.matmul(s,s)/np.matmul(s,y) #this is the other possible scaling you can use
            if BBscaling < 0:
                BBscaling = BBhi
            elif BBscaling < BBlow:
                BBscaling = BBlow
            elif BBscaling > BBhi:
                BBscaling = BBhi
            temp = BBscaling*temp
        #otherwise, there will be no scaling and the search direction will simply be -grad

        if proj_type ==0: #each project first, then you won't have to project during line search
            d = projection(p-temp,bounds)-p #search direction for the projected gradient
        else:
            d = -temp #search directino without projection

        ########deprecated
#        if past[-1] == 0: #in this case we need to remember the search direction of the iterate; if past[-1] == 0 it means we might have to return to that point in the non monotone search.
#            past[0].append(d) #append search direction corresponding to the iterate
#            if past[0][2][0] == None: #depending on what watchdogls is, we  may need to update the current gradient as well.
#                past[0][2] = grad
        #########deprecated
#        dirder = np.matmul(grad,d) #directional derivative #deprecated
        pn, objn, gradn, hessn, objeval, gradeval = linesearch(p,d,obj,fnc,fnc_objder,grad, args, iters, bounds, past, pastp, t, proj_type = proj_type, **kwargs)

        if gradn[0] == None: #if need to get new gradient
            objn, gradn = fnc_objder(pn,*args)
            totobjeval += 1
            totgradeval +=1

        if srch_type ==1:
            s = pn-p #definition of s and y for barzilai borwein scaling
            y = gradn-grad

        #update iterations and current values
        iters += 1
        totobjeval += objeval
        totgradeval += gradeval

        diff = abs(obj-objn)/obj #relative reduction in objective
        p = pn
        obj = objn
        grad = gradn
        n_grad = np.linalg.norm(grad)
    #report reason for termination
    if n_grad <= epsf:
        exitmessage = 'Norm of gradient reduced to '+str(n_grad)
    elif diff <= eps:
        exitmessage = 'Relative reduction in objective is '+str(diff)
    elif iters == maxit:
        exitmessage = 'Reached '+str(maxit)+' iterations'

    if obj > min(past):
        ind = np.argmin(past)
        obj = past[ind]
        p = pastp[ind]
    out = []
    outdict = {}
    out.append(p)
    out.append(obj)
    outdict['grad'] = grad
    outdict['task'] = exitmessage
    outdict['gradeval'] = totgradeval
    outdict['objeval'] = totobjeval
    outdict['iter'] = iters
    out.append(outdict)

    return out

def SPSA(fnc, unused1, unused2, p, bounds, unused3, args, q = 1, maxit = 1e3, maxs = 50, **kwargs):
    #minimize a scalar function with bounds using SPSA
    #fnc - objective
    #p - initial guess
    #bounds - box bounds for p
    #args - arguments that are passed to fnc
    #maxit = 1e3 - termination if iterations are more than maxit
    #q = 1 - number of times to do the SPSA gradient (q = 1 is a single realization of the stochastic perturbation \delta_k)
    #kwargs - can pass in kwargs to control constants for step length


#    obj = fnc(p, *args) #objective and gradient
    grad = SPSA_grad(p,fnc,*args)
    iters = 1 #number of iterations (termination)
    totobjeval = 2*q #keeps track of total objective and gradient evaluations
    totgradeval = 0
    diff = 0
    stuck = 0

    while iters < maxit and stuck <maxs:
#        print(obj)
        d = -grad #search direction
#        dirder = np.matmul(grad,d) #directional derivative
        pn, objn, gradn, hessn, objeval, gradeval = fixedstep(p,d,None,None,None,None,None,iters,bounds,**kwargs)

        if gradn[0] == None: #if need to get new gradient
            gradn = SPSA_grad(pn,fnc,*args)
            totobjeval += 2*q

        #update iterations and current values
        diff = np.linalg.norm(pn-p)
        if diff ==0:
            stuck += 1
        else:
            stuck = 0
        iters += 1
        p = pn
        grad = gradn

    #report reason for termination
    if iters == maxit:
        exitmessage = 'Reached '+str(maxit)+' iterations'
    if stuck ==maxs:
        exitmessage = 'Unable to make progress in '+str(maxs)+' iterations'

    obj = fnc(p,*args)

    out = []
    outdict = {}
    out.append(p)
    out.append(obj)
    outdict['grad'] = grad
    outdict['task'] = exitmessage
    outdict['gradeval'] = totgradeval
    outdict['objeval'] = totobjeval+1
    outdict['iter'] = iters
    out.append(outdict)

    return out

def SQP(fnc, fnc_der, fnc_hess1, p, bounds, linesearch, args, t = 0, hessfn = False, eps = 1e-5, epsf = 1e-5,
         maxit = 1e3, der_only = False, BBlow = 1e-9, BBhi = 1, proj_type = 0, hesslow = 1e-4, hesshi = 100, **kwargs):
    #fnc - objective
    #fnc_der - derivative or objective and derivative
    #fnc_hess - input a function only returning the gradient with keyword hessfn=False, or you can input a function that will compute the hessian directly.
    #p - initial guess
    #linesearch - linesearch function
    #bounds - box bounds for p
    #*args - arguments that are passed to fnc, fnc_der, fnc_hess
    #if hessfn is true fnc_hess1 gives explicit hessian. Otherwise the hessian will be approximated, and fnc_hess1 contains a function that will return gradient
    #eps = 1e-5 - termination if relative improvement is less than eps
    #epsf =1e-5 - termination if gradient norm is less than epsf
    #maxit = 1e3 - termination if iterations are more than maxit
    #der_only indicates fnc_der only gives derivative
    #kwargs - any special arguments for linesearch need to be in kwargs

    if der_only:
        def fnc_objder(p, *args):
            obj = fnc(p, *args)
            grad = fnc_der(p,*args)

            return obj, grad
    else:
        fnc_objder = fnc_der

    if hessfn:
        fnc_hess = fnc_hess1
    else:
        def fnc_hess(p, args, curgrad, gradfn = fnc_hess1):
            hess = helper.approx_hess(p,args,gradfn = fnc_hess1,curgrad = curgrad)
            return hess

    obj, grad = fnc_objder(p, *args) #objective and gradient
#    hess = fnc_hess(p,*args, grad) #do it at top of loop
    n_grad = np.linalg.norm(grad) #norm of gradient
    diff = 1 #checks reduction in objective (termination)
    iters = 1 #number of iterations (termination)
    totobjeval = 1 #keeps track of total objective and gradient evaluations
    totgradeval = 1
#    tothesseval = 1

    past = [obj] #this is how we initialize the past. In this code, we will have special functions to handle the non-monotone,
    pastp = [p] #whereas before we were using watchdog which relied on some existing line search.

    if t != 0:
        watchdogls = linesearch
        linesearch = watchdog
#        ########compute the search direction in this case.....################ #actually don't do this
#        temp = grad
#        if srch_type ==0:
#            temp = temp/n_grad
#        if proj_type ==0: #each project first, then you won't have to project during line search
#            d = projection(p-temp,bounds)-p #search direction for the projected gradient
#        else:
#            d = -temp #search directino without projection
#        #########################################################################
        past = [[p, obj, grad],t+1] #initialize the past iterates for monotone
    else:
        watchdogls = None
        past = [None] #past will remain None unless we are doing the nonmonotone line search


    s = [1] #initialize BB scaling
    y = [1]
    cur = 1e-2 #very small regularization prevents singular matrix
    while diff > eps and n_grad > epsf and iters < maxit:

#        cur = cur*2
#        print(obj)
        #do the scaling type; either scale by norm of gradient, using BB scaling, or no scaling
        hess = fnc_hess(p, args, grad) #get new hessian
        hess = hess + cur*np.identity(len(p)) #regularization
        safeguard = False
        d = -np.linalg.solve(hess,grad) #newton search direction
        dnorm = np.linalg.norm(d)
#        print(np.matmul(-grad,d))
        if dnorm >= hesshi*n_grad: #safeguards on hessian being poorly conditioned
            d = -grad
            safeguard = True

        elif np.matmul(-grad,d) <= hesslow*n_grad*dnorm: #safeguard on hessian not giving a descent direction
            d = -grad
            safeguard = True

#        print(safeguard)
#        if srch_type ==0 or iters ==1:  #scale by norm of gradient
        if safeguard:  #BB scaling
#            cur= cur*1.5
#            print('hi')
            BBscaling = np.matmul(s,y)/np.matmul(y,y) #one possible scaling
#            BBscaling = np.matmul(s,s)/np.matmul(s,y) #this is the other possible scaling you can use
            if BBscaling < 0:
                BBscaling = BBhi
            elif BBscaling < BBlow:
                BBscaling = BBlow
            elif BBscaling > BBhi:
                BBscaling = BBhi
            d = BBscaling*d
#        else:
#            cur = cur/1.5
        #otherwise, there will be no scaling and the search direction will simply be -grad

        if proj_type ==0: #each project first, then you won't have to project during line search
            d = projection(p+d,bounds)-p #search direction for the projected gradient

        if past[-1] == 0: #in this case we need to remember the search direction of the iterate; if past[-1] == 0 it means we might have to return to that point in the non monotone search.
            past[0].append(d) #append search direction corresponding to the iterate
            if past[0][2][0] == None: #depending on what watchdogls is, we  may need to update the current gradient as well.
                past[0][2] = grad

        pn, objn, gradn, hessn, objeval, gradeval = linesearch(p,d,obj,fnc,fnc_objder,grad, args, iters, bounds, past, watchdogls, proj_type = proj_type, t = t, **kwargs)

        if gradn[0] == None: #if need to get new gradient
            objn, gradn = fnc_objder(pn,*args)
            totobjeval += 1
            totgradeval +=1




        s = pn-p #definition of s and y for barzilai borwein scaling #for safe guarding when needed
        y = gradn-grad

        #update iterations and current values
        iters += 1
        totobjeval += objeval
        totgradeval += gradeval
#        tothesseval += 1

        diff = abs(obj-objn)/obj #relative reduction in objective
        p = pn
        obj = objn
        grad = gradn
        n_grad = np.linalg.norm(grad)

    if past[-1] != None: #in the case we did a non monotone search we have some special termination conditions (these take place inside watchdog function) and need an extra step to report the solution
        if past[0][1] < obj:
            obj = past[0][1]
            p = past[0][0]
    #report reason for termination
    if n_grad <= epsf:
        exitmessage = 'Norm of gradient reduced to '+str(n_grad)
    elif diff <= eps:
        exitmessage = 'Relative reduction in objective is '+str(diff)
    elif iters == maxit:
        exitmessage = 'Reached '+str(maxit)+' iterations'

    out = []
    outdict = {}
    out.append(p)
    out.append(obj)
    outdict['grad'] = grad
    outdict['task'] = exitmessage
    outdict['gradeval'] = totgradeval
    outdict['objeval'] = totobjeval
    outdict['iter'] = iters
    outdict['hesseval'] = iters - 1 #hessian evaluations is iters - 1
    out.append(outdict)

    return out

def SQP2(fnc, fnc_der, fnc_hess1, p, bounds, linesearch, args, t = 0, hessfn = False, eps = 1e-5, epsf = 1e-5,
         maxit = 1e3, der_only = False, BBlow = 1e-9, BBhi = 1, proj_type = 0, hesslow = 1e-4, hesshi = 100, **kwargs):
    #fnc - objective
    #fnc_der - derivative or objective and derivative
    #fnc_hess - input a function only returning the gradient with keyword hessfn=False, or you can input a function that will compute the hessian directly.
    #p - initial guess
    #linesearch - linesearch function
    #bounds - box bounds for p
    #*args - arguments that are passed to fnc, fnc_der, fnc_hess
    #if hessfn is true fnc_hess1 gives explicit hessian. Otherwise the hessian will be approximated, and fnc_hess1 contains a function that will return gradient
    #eps = 1e-5 - termination if relative improvement is less than eps
    #epsf =1e-5 - termination if gradient norm is less than epsf
    #maxit = 1e3 - termination if iterations are more than maxit
    #der_only indicates fnc_der only gives derivative
    #kwargs - any special arguments for linesearch need to be in kwargs

    if der_only:
        def fnc_objder(p, *args):
            obj = fnc(p, *args)
            grad = fnc_der(p,*args)

            return obj, grad
    else:
        fnc_objder = fnc_der

    if hessfn:
        fnc_hess = fnc_hess1
    else:
        def fnc_hess(p, args, curgrad, gradfn = fnc_hess1):
            hess = helper.approx_hess(p,args,gradfn = fnc_hess1,curgrad = curgrad)
            return hess

    obj, grad = fnc_objder(p, *args) #objective and gradient
#    hess = fnc_hess(p,*args, grad) #do it at top of loop
    n_grad = np.linalg.norm(grad) #norm of gradient
    diff = 1 #checks reduction in objective (termination)
    iters = 1 #number of iterations (termination)
    totobjeval = 1 #keeps track of total objective and gradient evaluations
    totgradeval = 1
#    tothesseval = 1

    past = [obj] #this is how we initialize the past. In this code, we will have special functions to handle the non-monotone,
    pastp = [p] #whereas before we were using watchdog which relied on some existing line search.


    s = [1] #initialize BB scaling
    y = [1]
    cur = 1e-4 #very small regularization prevents singular matrix
    while diff > eps and n_grad > epsf and iters < maxit:

#        cur = cur*2
#        print(obj)
        #do the scaling type; either scale by norm of gradient, using BB scaling, or no scaling
        hess = fnc_hess(p, args, grad) #get new hessian
        hess = hess + cur*np.identity(len(p)) #regularization
        safeguard = False
        d = -np.linalg.solve(hess,grad) #newton search direction
        dnorm = np.linalg.norm(d)
#        print(np.matmul(-grad,d))
#        print(dnorm)
        if dnorm >= hesshi*n_grad: #safeguards on hessian being poorly conditioned
            d = -grad
            safeguard = True

        elif np.matmul(-grad,d) <= hesslow*n_grad*dnorm: #safeguard on hessian not giving a descent direction
            d = -grad
            safeguard = True

#        print(safeguard)
#        if srch_type ==0 or iters ==1:  #scale by norm of gradient
        if safeguard:  #BB scaling
#            cur= cur*1.5
#            print('hi')
            BBscaling = np.matmul(s,y)/np.matmul(y,y) #one possible scaling
#            BBscaling = np.matmul(s,s)/np.matmul(s,y) #this is the other possible scaling you can use
            if BBscaling < 0:
                BBscaling = BBhi
            elif BBscaling < BBlow:
                BBscaling = BBlow
            elif BBscaling > BBhi:
                BBscaling = BBhi
            d = BBscaling*d
#        else:
#            cur = cur/1.5
        #otherwise, there will be no scaling and the search direction will simply be -grad

        if proj_type ==0: #each project first, then you won't have to project during line search
            d = projection(p+d,bounds)-p #search direction for the projected gradient


        pn, objn, gradn, hessn, objeval, gradeval = linesearch(p,d,obj,fnc,fnc_objder,grad, args, iters, bounds, past, pastp, t, proj_type = proj_type, **kwargs)

        if gradn[0] == None: #if need to get new gradient
            objn, gradn = fnc_objder(pn,*args)
            totobjeval += 1
            totgradeval +=1




        s = pn-p #definition of s and y for barzilai borwein scaling #for safe guarding when needed
        y = gradn-grad

        #update iterations and current values
        iters += 1
        totobjeval += objeval
        totgradeval += gradeval
#        tothesseval += 1

        diff = abs(obj-objn)/obj #relative reduction in objective
        p = pn
        obj = objn
        grad = gradn
        n_grad = np.linalg.norm(grad)
    #report reason for termination
    if n_grad <= epsf:
        exitmessage = 'Norm of gradient reduced to '+str(n_grad)
    elif diff <= eps:
        exitmessage = 'Relative reduction in objective is '+str(diff)
    elif iters == maxit:
        exitmessage = 'Reached '+str(maxit)+' iterations'

    if obj > min(past):
        ind = np.argmin(past)
        obj = past[ind]
        p = pastp[ind]
    out = []
    outdict = {}
    out.append(p)
    out.append(obj)
    outdict['grad'] = grad
    outdict['task'] = exitmessage
    outdict['gradeval'] = totgradeval
    outdict['objeval'] = totobjeval
    outdict['iter'] = iters
    outdict['hesseval'] = iters - 1
    out.append(outdict)

    return out

def projection(p, bounds):
    n = len(p)
    for i in range(n):
        if p[i] < bounds[i][0]:
            p[i] = bounds[i][0]
        elif p[i] > bounds[i][1]:
            p[i] = bounds[i][1]
    return p

def backtrack2(p,d,obj,fnc,fnc_objder, grad, args, iters, bounds, *fargs, c1=1e-4, alo = .1, ahi = .9, gamma = .5, proj_type = 0, maxLSiter = 40, **kwargs):
    #this is the current backtracking algorithm, it uses interpolation to define steps.
    #really the only feature still missing is the ability to terminate the search when we are at the desired accuracy. This isn't really an issue for
    #the calibration problem since we don't need machine epsilon precision, so we'll always terminate in the main algorithm and not ever in the line search.
    #alo and ahi are safeguarding parameters. default .1 and .9. gamma = .5 is the step size used when the interpolation gives a result outside of the safeguards.
    #can handle either projection in the algorithm, or projection inside this algorithm at each linesearch step.

    #initialization
    #attempts to satisfy a sufficient decrease condition by considering progressively shorter step lengths.
    #c1 controls how much of a decrease constitutes a "sufficient" decrease. typical value is c1 = 1e-4
    #gamma controls how much to shorten the step length by. typicaly value is gamma = .5
    #for calibration problem specifically I have found that using a smaller value like gamma = .2 seems to work better.
    if proj_type ==0: #projection happens in optimization algorithm
        pn = p + d
        da = d
    else: #need to project every iteration in line search
        pn = projection(p+d,bounds)
        da = pn-p

    dirder = np.matmul(grad,da) #directional derivative
#    cdirder = c1*dirder
    objn = fnc(pn,*args)
    objeval = 1
    gradeval = 0
    a = 1
    #backtracking procedure
    while objn > obj + c1*dirder: #if sufficient decrease condition is not met
        #gamma is the current modifier to the step length
        #a is the total modifier
        #dirder is the current directional derivative times the step length.
        #objb is the previous objective value, objn is the current objective value
        gamman = -dirder*a/(2*(objn-obj-dirder)) #quadratic interpolation
        #safeguards
#        if gamman < alo: #this is one way to safeguard
#            gamman = alo
#        elif gamman > ahi:
#            gamman = ahi
        if gamman < alo or gamman > ahi:  #this is how the paper recommends to safeguard
            gamman = gamma
        a = gamman*a #modify current step length
        d = gamman*d #modify current direction
        #get new step if proj_type == 0
        if proj_type ==0:
            dirder = gamman*dirder
            pn = p+d
        else:
            pn = projection(p+d,bounds) #project onto feasible set
            da = pn - p #new search direction
            dirder = np.matmul(grad,da) #directional derivative

        objn = fnc(pn,*args)
        objeval += 1

        #need a way to terminate the linesearch if needed.
        if objeval > maxLSiter:
            print('linesearch failed to find a step of sufficient decrease')
            if objn >= obj: #if current objective is worse than original
                pn = p #return original
                objn = obj
            break



    gradn = [None]
    hessn = [None]

    return pn, objn, gradn, hessn, objeval, gradeval

def nmbacktrack(p,d,obj,fnc,fnc_objder, grad, args, iters, bounds, past, pastp, t, *fargs, c1=1e-4, alo = .1, ahi = .9, gamma = .5, proj_type = 0, maxLSiter = 40, **kwargs):
    #non monotone backtracking with interpolation and safeguards; this is to be used with the pgrad_descent2
    #attempts to satisfy a sufficient decrease condition by considering progressively shorter step lengths.
    #c1 controls how much of a decrease constitutes a "sufficient" decrease. typical value is c1 = 1e-4
    #gamma controls how much to shorten the step length by. typicaly value is gamma = .5
    #for calibration problem specifically I have found that using a smaller value like gamma = .2 seems to work better.

    ##inputs
    #p - parameters for initial guess
    #d - search direction (may not be same as gradient due to projection)
    #obj - objective function value
    #fnc - function to evalate objective
    #fnc_objder - function to evaluate both objective and gradient
    #grad - gradient of objective w.r.t. current parameters
    #args - extra arguments to pass to objective/gradeint function
    #iters - numbers of iterations of optimization alg so far
    #bounds - bounds for parameters, list of tuple for each parameter with lower and upper bounds
    #past - past values of the objective (need this for nonmonotone part)
    #pastp - past values of the parameters
    #t - maximum number of iterations we can go without a sufficient decrease - if this is zero then this is just regular backtracking
    #*fargs  - any extra arguments that may be passed in (this is for other routines since they might have different inputs; in this code it is not used)
    #c1 - parameter controls sufficient decrease, if c1 is smaller, required decrease is also smaller. typically 1e-4 is used
    #alo - minimum safeguard for interpolated step size, can usually leave as .1
    #ahi - maximum safeguard for interpolated step size, can usually leave as .9
    #gamma - this is the modification to the step size in case the interpolation fails (typically left as .5)
    #proj_type = 0 - if proj type is zero we project every iteration in optimization algorithm (project search direction), otherwise everytime we get a new step size we will project (proj_type = 1)
    #maxLSiter = 40 - maximum number of iterations of linesearch algorithm before we return the best found value, typically this should find the step in ~2-3 iterations and if you get past 10 or 20
    #it might mean the search direction has a problem

    #outputs
    #pn - new parameter values
    #objn - new objective value
    #gradn - new gradient value (false if not computed)
    #hessn - new hessian value (false if not computed)
    #objeval - number of objective evaluations
    #gradeval - number of gradient evaluations

    #initialization
    if proj_type ==0: #projection happens in optimization algorithm
        pn = p + d
        da = d
    else: #need to project every iteration in line search
        pn = projection(p+d,bounds)
        da = pn-p

    dirder = np.matmul(grad,da) #directional derivative
#    cdirder = c1*dirder
    objn = fnc(pn,*args)
    objeval = 1
    gradeval = 0
    a = 1
    if iters < t:
        maxobj = obj
    else:
        maxobj = max(past)
    #backtracking procedure
    while objn > maxobj + c1*dirder: #if sufficient decrease condition is not met
        #gamma is the current modifier to the step length
        #a is the total modifier
        #dirder is the current directional derivative times the step length.
        #objb is the previous objective value, objn is the current objective value
        gamman = -dirder*a/(2*(objn-obj-dirder)) #quadratic interpolation
        #safeguards
#        if gamman < alo: #this is one way to safeguard
#            gamman = alo
#        elif gamman > ahi:
#            gamman = ahi
        if gamman < alo or gamman > ahi:  #this is how the paper recommends to safeguard
#        if True: #deubgging purposes
            gamman = gamma
        a = gamman*a #modify current step length
        d = gamman*d #modify current direction
        #get new step if proj_type == 0
        if proj_type ==0:
            dirder = gamman*dirder
            pn = p+d
        else:
            pn = projection(p+d,bounds) #project onto feasible set
            da = pn - p #new search direction
            dirder = np.matmul(grad,da) #directional derivative

        objn = fnc(pn,*args)
        objeval += 1

        #need a way to terminate the linesearch if needed.
        if objeval > maxLSiter:
            print('linesearch failed to find a step of sufficient decrease')
            if objn >= obj: #if current objective is worse than original
                pn = p #return original
                objn = obj
            break


    if iters < t:
        past.append(objn) #add the value to the past
        pastp.append(pn)
    else:
        past.pop(0) #remove the first value
        past.append(objn) #add the new value at the end
        pastp.pop(0)
        pastp.append(pn)
    gradn = [None]
    hessn = [None]

    return pn, objn, gradn, hessn, objeval, gradeval

def fixedstep(p,d,obj,fnc,fnc_objder,grad,args,iters, bounds, *fargs,c1 = 5e-4,c2 = 1, **kwargs):
    #fixed step length for line search.
    #c1 is the initial step on the first iteration.
    #the k^th step is c1 * k**c2
    #default values of c1 = 5e-4, c2 = 1.
    step = c1*(iters**-c2) #fixed step size
    pn = p +step*d #definition of new solution
    pn = projection(pn,bounds) #project onto the feasible region
    fff = [None]

    return pn, fff,fff,fff, 0, 0


def weakwolfe2(p,d,obj,fnc,fnc_objder, grad, args, iters, bounds, *fargs, c1=1e-4, c2 = .5, eps1 = 1e-1, eps2 = 1e-6, proj_type = 0, maxLSiter = 20, **kwargs):
    #fulfills either strong or weak wolfe line search. it's currently used as strong wolfe
    #compared to the original wolfe search, this one uses a (safeguarded) quadratic interpolation to give the first trial step for the zoom function; previously this was done using bisection.
    ######################
    #you just need to change one line in this program and one line in zoom and you can change this between strong wolfe
    #and weak wolfe. I think strong wolfe tends to be slightly better
    ########################

    #for trajectory calibration though it seems backtracking LS works better than using the wolfe conditions, since here the gradient is relatively expensive compared to obj. (even using adjoint)
    #in general we're going to be constrained by the budget, so even though wolfe can give us better steps, we'd rather do more iterations with slightly worse steps at each iteration.
    #note that the wolfe linesearches require the gradient to be evaluated at every trial step length, whereas backtrack/armijo only requires the objective to be evaluated

    #c1 - for sufficient decrease condition; lower = easier to accept
    #c2 - for curvature condition ; #higher = easier to accept.
    #require 0 < c1 < c2 < 1; typically values are 1e-4, .5 or 1e-4, .9. stronger curvature condition c2 = better steps, but more evaluations

    #eps1 - initial guess for the steplength ai; should choose something small, like 1e-2
    #eps2 - termination length for zoom function (accuracy for final step length); should choose something small

    #proj_type = 0 - either we project before the linesearch (0), or we project every iteration in linesearch (1)
    #maxLSiter = 40 - number of iterations we will attempt

    #aib and amax specify the range of step lengths we will consider; defined between [0,1]
    aib = 0
    amax = 1
    ai = eps1 #initial guess for step length
    objb = obj #initialize previous objective value
    ddirderb = None
#    dirderb = 0 #initialize directional derivative w.r.t. aib

    #linedir(a) will return the trial point and search direction for step length a depending on the chosen projection strategy
    #accepts step length a,
    #returns new point pn, which is in direction da/a, and has directional derivative dirdera/a
    if proj_type ==0:
        dirder = np.matmul(grad,d)
        def linedir(a, p=p,d = d, bounds=bounds, dirder = dirder):
            pn = p + a*d
            da = a*d
            dirdera = a*dirder
            return pn, da, dirdera
    else:
        def linedir(a, p=p, d=d,  bounds=bounds, grad = grad):
            pn = projection(p+a*d,bounds)
            da = pn-p
            dirdera = np.matmul(grad,da)
            return pn, da, dirdera

    objdereval = 0 #count number of objective and gradient evaluations; they are always the same for this strategy.


    for i in range(maxLSiter): #up to maxLSiter to find the bounds on a
        pn, da, dirdera = linedir(ai) #new point for the line search
        objn, gradn = fnc_objder(pn,*args) #objective and gradient for the new point
        objdereval += 1

        if objn > obj+c1*dirdera or (objn >= objb and objdereval > 1): #if sufficient decrease is not met then ai must be an upper bound on a good step length

            atrial = -dirdera*ai/(2*(objn-obj-dirdera))
            out = zoom3(aib,ai, eps2, linedir, fnc_objder,args, p,grad, obj, objb, objdereval, c1, c2,atrial) #put bounds into zoom to find good step length

            return out

        ddirder = np.matmul(gradn,da)/ai #directional derivative at new point
#        if ddirder >= c2*dirdera/ai: #if weak wolfe conditions are met
        if abs(ddirder) <= -c2*dirdera/ai: #if strong wolfe conditions are met

            return pn, objn, gradn, [None], objdereval, objdereval #we are done

        if ddirder >= 0:  #if the directional derivative is positive it means we went too far; therefore we found an upperbound

            atrial = -dirdera*ai/(2*(objn-obj-dirdera))
            out = zoom3(ai,aib, eps2, linedir, fnc_objder,args, p,grad, obj, objn, objdereval, c1 , c2, atrial) #put bounds into zoom to find good step length
            return out

        if i == maxLSiter-1:
            print('failed to find suitable range for stepsize')
            if objn >= obj:
                pn = p
                objn = obj
                gradn = grad
            break

        #interpolate to get next point
        if objdereval ==1: #quadratic interpolation first time
            aif = -dirdera*ai/(2*(objn-obj-dirdera)) #next ai to check
        else:
            d1 = ddirderb+ddirder-3*((objb-objn)/(aib-ai))
            d2 = np.sign(ai-aib)*(d1**2-ddirderb*ddirder)**.5
            aif = ai-(ai-aib)*((ddirder+d2-d1)/(ddirder-ddirderb+2*d2))

        if aif < ai or aif < 0 or np.isnan(aif): #if interpolation gives something screwy
            aif = 2*ai #increase by fixed amount
        aif = min(aif,amax) #next step length must be within range

#        #other strategy
#        aif = 2* ai
#        if aif > amax:
#            out = zoom(0,amax, eps2, linedir, fnc_objder,args, p,grad, obj, obj, objdereval, c1 , c2)
#            return out

        #progress iteration
        aib = ai
        ai = aif
        ddirderb = ddirder
        objb = objn
#        dirderb = dirdera #we potentially need this for the new trial step for zoom


    return pn, objn, gradn, [None], objdereval, objdereval #shouldn't reach here ideally; should terminate due to an if statement

def zoom3(alo, ahi, eps2, linedir, fnc_objder, args, p,grad,obj, objlo, objdereval, c1, c2, atrial ):
    #most recent zoom function corresponding to weakwolfe2.
    if abs(ahi-alo) <= eps2: #special case where bounds are already tight enough
        aj = (alo+ahi)/2 #bisection
        pn, da, dirdera = linedir(aj) #get new point, new direction, new directional derivative
        objn, gradn = fnc_objder(pn,*args) #evaluate new point
        objdereval +=1
        return pn, objn, gradn, [None], objdereval, objdereval

    #try modifying so if something satisfies sufficient decrease we will remember it
    count = 0
    ddirderb, objb, aib = None, None, None
    best = (p, obj, grad) #initialize best solution to return if can't satisfy curvature condition

    if atrial <= alo or atrial >= ahi or np.isnan(atrial):
#        print('safeguard')
        aj = (alo+ahi)/2
    else:
#        print('not safeguard')
        aj = atrial

    while abs(ahi-alo) > eps2: #iterate until convergence to good step length
        pn, da, dirdera = linedir(aj)
        objn, gradn = fnc_objder(pn,*args)
        objdereval +=1
        count += 1

        ddirder = np.matmul(gradn,da)/aj
        if objn > obj + c1*dirdera or objn >= objlo: #if sufficient decrease not met lower the upper bound
            ahi = aj
        else:
#            if ddirder >= c2*dirdera/aj: #if weak wolfe conditions are met return the solution
            if abs(ddirder) <= -c2*dirdera/aj: #if stronge wolfe conditions are met

                return pn, objn, gradn, [None], objdereval, objdereval
            if objn < best[1]: #keep track of best solution which only satisfies sufficient decrease #can get rid of this
                best = (pn, objn, gradn)
            if ddirder*(ahi-alo) >= 0: #otherwise do this
                ahi = alo
            alo = aj #pretty sure this is supposed to be here.

        if count ==1: #quadratic interpolation first time
            aif = -dirdera*aj/(2*(objn-obj-dirdera)) #next ai to check
        else:
            d1 = ddirderb+ddirder-3*((objb-objn)/(aib-aj))
            d2 = np.sign(aj-aib)*(d1**2-ddirderb*ddirder)**.5
            aif = aj-(aj-aib)*((ddirder+d2-d1)/(ddirder-ddirderb+2*d2))

        if aif < alo or aif > ahi or np.isnan(aif): #if interpolation gives something screwy (this happens occasionally so need safeguard)
            aif = (alo+ahi)/2 # use bisection
        aib = aj
        aj = aif
        ddirderb = ddirder
        objb = objn


    print('failed to find a stepsize satisfying weak wolfe conditions')
    if objn < best[1]: #if current step is better than the best #y
        best = (pn, objn, gradn)

    return best[0], best[1], best[2], [None], objdereval, objdereval

def nmweakwolfe(p,d,obj,fnc,fnc_objder, grad, args, iters, bounds, past, pastp, t, *fargs, c1=1e-4, c2 = .5, eps1 = 1e-1, eps2 = 1e-6, proj_type = 0, maxLSiter = 40, **kwargs):
    #fulfills either strong or weak wolfe line search. it's currently used as strong wolfe
    #compared to the original wolfe search, this one uses a (safeguarded) quadratic interpolation to give the first trial step for the zoom function; previously this was done using bisection.
    ######################
    #you just need to change one line in this program and one line in zoom and you can change this between strong wolfe
    #and weak wolfe. I think strong wolfe tends to be slightly better
    ########################

    #for trajectory calibration though it seems backtracking LS works better than using the wolfe conditions, since here the gradient is relatively expensive compared to obj. (even using adjoint)
    #in general we're going to be constrained by the budget, so even though wolfe can give us better steps, we'd rather do more iterations with slightly worse steps at each iteration.
    #note that the wolfe linesearches require the gradient to be evaluated at every trial step length, whereas backtrack/armijo only requires the objective to be evaluated

    #c1 - for sufficient decrease condition; lower = easier to accept
    #c2 - for curvature condition ; #higher = easier to accept.
    #require 0 < c1 < c2 < 1; typically values are 1e-4, .5 or 1e-4, .9. stronger curvature condition c2 = better steps, but more evaluations

    #eps1 - initial guess for the steplength ai; should choose something small, like 1e-2
    #eps2 - termination length for zoom function (accuracy for final step length); should choose something small

    #proj_type = 0 - either we project before the linesearch (0), or we project every iteration in linesearch (1)
    #maxLSiter = 40 - number of iterations we will attempt

    #aib and amax specify the range of step lengths we will consider; defined between [0,1]
    aib = 0
    amax = 1
    ai = eps1 #initial guess for step length
    objb = obj #initialize previous objective value
    ddirderb = None #initialize directional derivative w.r.t. aib

    #linedir(a) will return the trial point and search direction for step length a depending on the chosen projection strategy
    #accepts step length a,
    #returns new point pn, which is in direction da/a, and has directional derivative dirdera/a
    if proj_type ==0:
        dirder = np.matmul(grad,d)
        def linedir(a, p=p,d = d, bounds=bounds, dirder = dirder):
            pn = p + a*d
            da = a*d
            dirdera = a*dirder
            return pn, da, dirdera
    else:
        def linedir(a, p=p, d=d,  bounds=bounds, grad = grad):
            pn = projection(p+a*d,bounds)
            da = pn-p
            dirdera = np.matmul(grad,da)
            return pn, da, dirdera

    objdereval = 0 #count number of objective and gradient evaluations; they are always the same for this strategy.

    if iters < t:
        maxobj = obj
    else:
        maxobj = max(past)


    for i in range(maxLSiter): #up to maxLSiter to find the bounds on a
        pn, da, dirdera = linedir(ai) #new point for the line search
        objn, gradn = fnc_objder(pn,*args) #objective and gradient for the new point
        objdereval += 1

        if objn > maxobj+c1*dirdera or (objn >= objb and objdereval > 1): #if sufficient decrease is not met then ai must be an upper bound on a good step length

            atrial = -dirdera*ai/(2*(objn-obj-dirdera))
            out = zoom4(aib,ai, eps2, linedir, fnc_objder,args, p,grad, obj, objb, objdereval, c1, c2,atrial, iters, past, pastp, t, maxobj) #put bounds into zoom to find good step length

            return out

        ddirder = np.matmul(gradn,da)/ai #directional derivative at new point
#        if ddirder >= c2*dirdera/ai: #if weak wolfe conditions are met
        if abs(ddirder) <= -c2*dirdera/ai: #if strong wolfe conditions are met
            if iters < t:
                past.append(objn) #add the value to the past
                pastp.append(pn)
            else:
                past.pop(0) #remove the first value
                past.append(objn) #add the new value at the end
                pastp.pop(0)
                pastp.append(pn)

            return pn, objn, gradn, [None], objdereval, objdereval #we are done

        if ddirder >= 0:  #if the directional derivative is positive it means we went too far; therefore we found an upperbound

            atrial = -dirdera*ai/(2*(objn-obj-dirdera))
            out = zoom4(ai,aib, eps2, linedir, fnc_objder,args, p,grad, obj, objn, objdereval, c1 , c2, atrial, iters, past, pastp, t, maxobj) #put bounds into zoom to find good step length
            return out

        if i == maxLSiter-1:
            print('failed to find suitable range for stepsize')
            if objn >= obj:
                pn = p
                objn = obj
                gradn = grad
            break

        #interpolate to get next point
        if objdereval ==1: #quadratic interpolation first time
            aif = -dirdera*ai/(2*(objn-obj-dirdera)) #next ai to check
        else:
            d1 = ddirderb+ddirder-3*((objb-objn)/(aib-ai))
            d2 = np.sign(ai-aib)*(d1**2-ddirderb*ddirder)**.5
            aif = ai-(ai-aib)*((ddirder+d2-d1)/(ddirder-ddirderb+2*d2))

        if aif < ai or aif < 0 or np.isnan(aif): #if interpolation gives something screwy
            aif = 2*ai #increase by fixed amount
        aif = min(aif,amax) #next step length must be within range

#        #other strategy
#        aif = 2* ai
#        if aif > amax:
#            out = zoom(0,amax, eps2, linedir, fnc_objder,args, p,grad, obj, obj, objdereval, c1 , c2)
#            return out

        #progress iteration
        aib = ai
        ai = aif
        ddirderb = ddirder
        objb = objn
#        dirderb = dirdera #we potentially need this for the new trial step for zoom

    if iters < t: #you should never get here but in case you do.
        past.append(objn) #add the value to the past
        pastp.append(pn)
    else:
        past.pop(0) #remove the first value
        past.append(objn) #add the new value at the end
        pastp.pop(0)
        pastp.append(pn)
    return pn, objn, gradn, [None], objdereval, objdereval #shouldn't reach here ideally; should terminate due to an if statement

def zoom4(alo, ahi, eps2, linedir, fnc_objder, args, p,grad,obj, objlo, objdereval, c1, c2, atrial,iters, past, pastp, t, maxobj ):
    #zoom that works with the nmweakwolfe
    if abs(ahi-alo) <= eps2: #special case where bounds are already tight enough
        aj = (alo+ahi)/2 #bisection
        pn, da, dirdera = linedir(aj) #get new point, new direction, new directional derivative
        objn, gradn = fnc_objder(pn,*args) #evaluate new point
        objdereval +=1
        return pn, objn, gradn, [None], objdereval, objdereval

    #try modifying so if something satisfies sufficient decrease we will remember it
    ddirderb, objb, aib = None, None, None
    count = 0
    best = (p, obj, grad) #initialize best solution to return if can't satisfy curvature condition

    if atrial <= alo or atrial >= ahi or np.isnan(atrial):
#        print('safeguard')
        aj = (alo+ahi)/2
    else:
#        print('not safeguard')
        aj = atrial

    while abs(ahi-alo) > eps2: #iterate until convergence to good step length
        pn, da, dirdera = linedir(aj)
        objn, gradn = fnc_objder(pn,*args)
        objdereval +=1
        count += 1

        ddirder = np.matmul(gradn,da)/aj
        if objn > maxobj + c1*dirdera or objn >= objlo: #if sufficient decrease not met lower the upper bound
            ahi = aj
        else:
#            if ddirder >= c2*dirdera/aj: #if weak wolfe conditions are met return the solution
            if abs(ddirder) <= -c2*dirdera/aj: #if stronge wolfe conditions are met
                if iters < t: #you should never get here but in case you do.
                    past.append(objn) #add the value to the past
                    pastp.append(pn)
                else:
                    past.pop(0) #remove the first value
                    past.append(objn) #add the new value at the end
                    pastp.pop(0)
                    pastp.append(pn)

                return pn, objn, gradn, [None], objdereval, objdereval
            if objn < best[1]: #keep track of best solution which only satisfies sufficient decrease #can get rid of this
                best = (pn, objn, gradn)
            if ddirder*(ahi-alo) >= 0: #otherwise do this
                ahi = alo
            alo = aj #pretty sure this is supposed to be here.

        if count ==1: #quadratic interpolation first time
            aif = -dirdera*aj/(2*(objn-obj-dirdera)) #next ai to check
        else:
            d1 = ddirderb+ddirder-3*((objb-objn)/(aib-aj))
            d2 = np.sign(aj-aib)*(d1**2-ddirderb*ddirder)**.5
            aif = aj-(aj-aib)*((ddirder+d2-d1)/(ddirder-ddirderb+2*d2))

        if aif < alo or aif > ahi or np.isnan(aif): #if interpolation gives something screwy (this happens occasionally so need safeguard)
            aif = (alo+ahi)/2 # use bisection
        aib = aj
        aj = aif
        ddirderb = ddirder
        objb = objn


    print('failed to find a stepsize satisfying weak wolfe conditions')
    if objn < best[1]: #if current step is better than the best #y
        best = (pn, objn, gradn)

    if iters < t: #you should never get here but in case you do.
        past.append(objn) #add the value to the past
        pastp.append(pn)
    else:
        past.pop(0) #remove the first value
        past.append(objn) #add the new value at the end
        pastp.pop(0)
        pastp.append(pn)

    return best[0], best[1], best[2], [None], objdereval, objdereval


def watchdog(p,d,obj,fnc,fnc_objder, grad, args, iters, bounds, past, watchdogls, *fargs, t = 3, c0 = 1,  c1 = 1e-4, **kwargs):
    #this can be called with pgrad_descent, but not pgrad_descent2. It sometimes works better than nmbacktrack, but usually is slightly worse.

    #this assumes the search direction has already been projected; i.e. projtype = 0

    #in addition to normal calling signature, watchdog accepts:
    #past - information on the past iterates; we might have to return to those in this algorithm. since past is a list of lists, and we only operate on the inner lists,
    #past will be updated without the need to explicitly return it. (I'm pretty sure this is correct)

    #watchdogls is used to perform linesearches when needed. this can be any linesearch (weak, strong or backtracking). When accepting the steps however, we will only check that the
    #sufficient decrease conditions are met.

    #c1 = 1e-4 is the parameter for sufficient decrease

    #c0 controls the default step size. It is reasonable to just take this as being 1.
#    print(past)
    if past[1] ==t+1: #special case corresponding to last else of the below main block
        pn3, objn3, gradn3, hessn3, objeval, gradeval = watchdogls(p,d,obj,fnc,fnc_objder,grad,args,iters,bounds,kwargs,c1=c1)
        past[0] = [pn3, objn3, gradn3]
        past[-1] = 0
        return pn3, objn3, gradn3, hessn3, objeval, gradeval


    elif past[-1] < t: #past[-1] is number of steps taken not satisfying sufficient decrease. t is total number of those steps we are allowed to take.
#        print('relaxed step')
        pn = p+c0*d #new point
        objn, gradn = fnc_objder(pn,*args)

        dirder = np.matmul(past[0][2], past[0][3])

        if objn <= past[0][1] + c1*c0*dirder:  #if we meet the sufficient decrease
#            print('sufficient decrease for relaxed step')
            past[0] = [pn,objn,gradn] #update the best iteration
            past[-1] = 0 #reset the number of relaxed steps to 0
        else:
            past[-1] += 1 #otherwise we took a relaxed step; so update past to reflect that

        return pn, objn, gradn, [None], 1, 1 #return the new step
    else: #we have taken the maximum number of relaxed steps and now need to ensure sufficient decrease.
        pn2, objn2, gradn2, hessn2, objeval, gradeval = watchdogls(p,d,obj,fnc,fnc_objder,grad,args,iters,bounds,  kwargs, c1 = c1) #need to do a linesearch on the current iterate

        dirder = np.matmul(past[0][2],past[0][3]) #recall past represents the last known point that was "good" i.e. it gave a sufficient decrease
        if obj <= past[0][1] or objn2 <=  past[0][1] + c1*c0*dirder: #if the new step gives a sufficient decrease with respect to the last known good step
#            print('sufficient decrease')
            past[0] = [pn2,objn2,gradn2] #update the best iteration
            past[-1] = 0 #reset the number of relaxed steps to 0
            return pn2, objn2, gradn2, hessn2, objeval, gradeval #then we can return the new step we found

        elif objn2 > past[0][1]: #at this point, we have taken a number of relaxed steps, and then a sufficient step from the relaxed steps. We haven't yet managed to get sufficient decrease,
            #with respect to the previous step we knew gave a sufficient decrease. Therefore we must either return to the original best known point, or take another sufficient step from the point
            #we just found.
            #in this case, we will return to the original, last known step past[0][0] which gave us a sufficient decrease.
#            print('return to best step')

            pn3, objn3, gradn3, hessn3, objeval3, gradeval3 = watchdogls(past[0][0],past[0][3], past[0][1], fnc, fnc_objder, past[0][2], args, iters, bounds,  kwargs, c1=c1)
            if objn3 == past[0][1]: #it's possible we can make no progress from the linesearch. If this happens then we will get stuck in a loop, so we will need to terminate
                #we will return the same point input into watchdog. this will cause the algorithm to terminate due to the objective not decreasing.
                return p, obj, grad, [None], objeval+objeval3, gradeval + gradeval3
            past[0] = [pn3, objn3, gradn3] #assuming the linesearch was successful, we have a new point with sufficient decrease and can update the best iteration
            past[-1] = 0
            return pn3, objn3, gradn3, hessn3, objeval+objeval3, gradeval+gradeval3

        else: #the last possibiliity is that we will continue to search from the point corresponding to objn2.
#            print('continue with search')
            past[-1] = t+1 #in this case we will give the past[-1] a special argument so we will perform a special action on the next iteration of the algorithm
            #the search direction is updated inside the algorithm.
            return pn2, objn2, gradn2, hessn2, objeval, gradeval


    return #this will never be reached.

