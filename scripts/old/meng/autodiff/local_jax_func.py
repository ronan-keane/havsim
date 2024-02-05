

import jax.numpy as jnp







def r_constant_jax(currinfo, frames, T_n, rp, adj=True, h=.1):

    # given a list of times and gamma constants (rinfo for a specific vehicle = currinfo) as well as frames (t_n and T_nm1 for that specific vehicle) and the relaxation constant (rp). h is the timestep (.1 for NGSim)

    # we will make the relaxation amounts for the vehicle over the length of its trajectory

    # rinfo is precomputed in makeleadfolinfo_r. then during the objective evaluation/simulation, we compute these times.

    # note that we may need to alter the pre computed gammas inside of rinfo; that is because if you switch mutliple lanes in a short time, you may move to what looks like only a marginally shorter headway,

    # but really you are still experiencing the relaxation from the lane change you just took

    if len(currinfo) == 0:

        relax = jnp.zeros(T_n - frames[0] + 1)

        return relax, relax  # if currinfo is empty we don't have to do anything



    out = jnp.zeros((T_n - frames[0] + 1, 1))  # initialize relaxation amount for the time between t_n and T_n

    out2 = jnp.zeros((T_n - frames[0] + 1, 1))

    outlen = 1



    maxind = frames[1] - frames[

        0] + 1  # this is the maximum index we are supposed to put values into because the time between T_nm1 and T_n is not simulated. Plus 1 because of the way slices work.

    if rp < h:  # if relaxation is too small for some reason

        rp = h  # this is the smallest rp can be

    #    if rp<h: #if relaxation is smaller than the smallest it can be #deprecated

    #        return out, out2 #there will be no relaxation



    # mylen = math.ceil(

    #     rp / h) - 1  # this is how many nonzero entries will be in r each time we have the relaxation constant

    mylen = jnp.ceil(rp / h) - 1

    r = jnp.linspace(1 - h / rp, 1 - h / rp * (mylen), mylen)  # here are the relaxation constants. these are determined only by the relaxation constant. this gets multipled by the 'gamma' which is the change in headway immediately after the LC



    for i in range(len(

            currinfo)):  # frames[1]-frames[0]+1 is the length of the simulation; this makes it so it will be all zeros between T_nm1 and T_n

        entry = currinfo[i]  # the current entry for the relaxation phenomenon

        curind = entry[0] - frames[0]  # current time is entry[0]; we start at frames[0] so this is the current index

        for j in range(outlen):

            if out2[curind, j] == 0:

                if curind + mylen > maxind:  # in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)

                    out[curind:maxind, j] = r[0:maxind - curind]

                    out2[curind:maxind, j] = currinfo[i][1]

                else:  # this is the normal case

                    out[curind:curind + mylen, j] = r

                    out2[curind:curind + mylen, j] = currinfo[i][1]

                break



        else:

            newout = jnp.zeros((T_n - frames[0] + 1, 1))

            newout2 = jnp.zeros((T_n - frames[0] + 1, 1))



            if curind + mylen > maxind:  # in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)

                newout[curind:maxind, 0] = r[0:maxind - curind]

                newout2[curind:maxind, 0] = currinfo[i][1]

            else:  # this is the normal case

                newout[curind:curind + mylen, 0] = r

                newout2[curind:curind + mylen, 0] = currinfo[i][1]



            out = jnp.append(out, newout, axis=1)

            out2 = jnp.append(out2, newout2, axis=1)

            outlen += 1



    #######calculate relaxation amounts and the part we need for the adjoint calculation #different from the old way

    relax = jnp.multiply(out, out2)

    relax = jnp.sum(relax, 1)



    if adj:

        outd = -(1 / rp) * (

                    out - 1)  # derivative of out (note that this is technically not the derivative because of the piecewise nature of out/r)

        relaxadj = jnp.multiply(outd,

                                out2)  # once multiplied with out2 (called gamma in paper) it will be the derivative though.

        relaxadj = jnp.sum(relaxadj, 1)

    else:

        relaxadj = relax



    return relax, relaxadj





def platoonobjfn_obj_jax(p, model, modeladjsys, modeladj, meas, sim, platooninfo, platoons, leadinfo, folinfo, rinfo,

                     use_r=False, m=5, dim=2, h=.1, datalen=9):



    lead = {}  # dictionary where we put the relevant lead vehicle information

    obj = 0

    n = len(platoons[1:])



    for i in range(n):  # iterate over all vehicles in the platoon

        # first get the lead trajectory and length for the whole length of the vehicle

        # then we can use euler function to get the simulated trajectory

        # lastly, we want to apply the shifted trajectory strategy to finish the trajectory

        # we can then evaluate the objective function

        curp = p[m * i:m * (i + 1)]

        curvehid = platoons[i + 1]  # current vehicle ID

        t_nstar, t_n, T_nm1, T_n = platooninfo[curvehid][0:4]  # grab relevant times for current vehicle

        frames = [t_n, T_nm1]



        relax, unused = r_constant_jax(rinfo[i], frames, T_n, curp[-1], False, h)  # get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.



        lead[i] = jnp.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory

        for j in leadinfo[i]:

            curleadid = j[0]  # current leader ID

            leadt_nstar = int(sim[curleadid][0, 1])  # t_nstar for the current lead, put into int

            lead[i][j[1] - t_n:j[2] + 1 - t_n, :] = sim[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,

                                                    :]  # get the lead trajectory from simulation



        leadlen = lead[i][:, 6]  # get lead length information

        IC = platooninfo[curvehid][5:7]  # get initial conditions

        #        IC = sim[curvehid][t_n-t_nstar,dataind[0:2]]

        simtraj, reg = euler(curp, frames, IC, model, lead[i], leadlen, relax, dim,

                             h)  # get the simulated trajectory between t_n and T_nm1



        # reg keeps track of the model regime. we don't actually need it if we just want to evaluate the obj fn.



        curmeas = meas[curvehid]  # current measurements #faster if we just pass it in directly?

        if T_n > T_nm1:  # if we need to do the shifted end

            shiftedtraj = shifted_end(simtraj, curmeas, t_nstar, t_n, T_nm1, T_n)

            simtraj = jnp.append(simtraj, shiftedtraj, 0)  # add the shifted end onto the simulated trajectory

        # need to add in the simulated trajectory into sim

        #        sim[curvehid] = meas[curvehid].copy()

        sim[curvehid][t_n - t_nstar:, 2:4] = simtraj[:, 1:3]



        loss = simtraj[:T_nm1 + 2 - t_n, 1] - curmeas[t_n - t_nstar:T_nm1 + 2 - t_nstar,

                                              2]  # calculate difference between simulation and measurments up til the first entry of shifted end (not inclusive)

        loss = jnp.square(

            loss)  # squared error in distance is the loss function we are using. will need to add ability to choose loss function later

        #        obj += sum(loss)

        #        obj += sci.simps(loss,None,dx=h,axis=0,even='first')

        obj += sum(loss) * h  # weighted by h



    return obj





def euler(p, frames, IC, func, lead, leadlen, r, dim=2, h=.1, *args):

    #############

    #    this is where the main computational time is spent.

    # can we make this faster by passing things in directly? i.e. not having to do the [:,dataind[0:2]] part and np.flip part?

    # can we use cython or something to make the for loop faster?

    #############



    # this function takes in step size (h), IC, lead trajectory (lead), and function to compute derivative

    # assume that lead is the lead trajectory for the time that we are integrating over

    # input h as negative for backwards in time: input h as positive for forward euler forwards in time

    # IC is a vector containing position and speed

    # p is vector of parameters



    # only currently intended to work for ODE case

    # should be able to handle second or first order model (although currently it assumes vector input/output so could be improved)

    # only meant to do 1 trajectory at a time



    # note that the time (frames) for lead need to match the times for the trajectory we are simulating



    # this assumes data is in format of the reconstructed data. If you want to use a different format pass optional argument



    #    frames = np.zeros((1,2)) #get relevant frames

    #    frames[0,1] = lead[-1,1]

    #    frames[0,0] = lead[0,1]

    lead = lead[:, 2:4]



    t = range(int(frames[0]), int(frames[1] + 1))

    a = frames[1] - frames[0] + 1



    #    a,b = lead.shape

    simtraj = jnp.empty((a, 3))  # initialize solution

    reg = jnp.empty(a)



    simtraj[:, 0] = t  # input frames = times

    simtraj[0, [1, 2]] = IC  # initial conditions



    if dim == 2:  # dim 2 model gives speed and acceleration output from func.

        for i in range(len(t) - 1):

            out, regime = func(simtraj[i, [1, 2]], lead[i, :], p, leadlen[i],

                               r[i])  # run the model, gets as output the derivative and the current regime

            simtraj[i + 1, [1, 2]] = simtraj[

                                         i, [1, 2]] + h * out  # forward euler formula where func computes derivative

            reg[i] = regime  # regime is defined this way with i.

            # print(func(out[i,[1,2]],lead[i,:],p))



    elif dim == 1:  # dim 1 model gives speed output from func.

        for i in range(len(t) - 1):

            out, regime = func(simtraj[i, [1, 2]], lead[i, :], p, leadlen[i],

                               r[i])  # run the model, gets as output the derivative and the current regime

            simtraj[i + 1, 1] = simtraj[i, 1] + h * out[0]  # forward euler formula where func computes derivative

            simtraj[i, 2] = out[0]  # first order model gives speed output only

            reg[i] = regime  # regime is defined this way with i.

            # print(func(out[i,[1,2]],lead[i,:],p))

    reg[

        -1] = regime  # this is what the last regime value should be for the adjoint method. Anything else will make the gradient less accurate



    return simtraj, reg





def shifted_end(simtraj, meas, t_nstar, t_n, T_nm1, T_n, h=.1, dim=2):

    # model can be either dim = 1 (position output only) or dim = 2 (position + speed output)

    # t_n and T_nm1 refer to the starting and ending times of the simulation, so to use a model with delay input the delayed values.

    dim = int(dim)

    shiftedtraj = jnp.zeros((T_n - T_nm1,

                             dim + 1))  # for now we are just going to assume the output should have 3 columns (time, position, speed, respectively)

    time = range(T_nm1 + 1, T_n + 1)



    if dim == 2:  # this is for a second order ode model

        shiftedtraj[:, 0] = time

        shiftedtraj[0, 1] = simtraj[-1, 1] + h * simtraj[

            -1, 2]  # position at next timestep from the velocity the vehicle currently has

        shiftedtraj[0, 2] = meas[

            T_nm1 + 1 - t_nstar, 3]  # speed in the shifted trajectory is actually the same since we are just shifted position

        if T_n - T_nm1 == 1:  # in this case there is nothing else to do but this should be pretty rare

            return shiftedtraj

        shift = shiftedtraj[0, 1] - meas[

            T_nm1 + 1 - t_nstar, 2]  # difference between last position of simulation and measurements

        shiftedtraj[1:, [1, 2]] = meas[T_nm1 + 2 - t_nstar:,

                                  2:4]  # load in the end trajectory taken directly from the data

        shiftedtraj[1:, 1] = shiftedtraj[1:, 1] + jnp.repeat(shift,

                                                             T_n - 1 - T_nm1)  # now shift the position by the required amount



    elif dim == 1:  # dimension 1 models

        # note you might to modify shiftedtraj to fill in the last speed value but this would require shifted_end to output simtraj as well as shiftedtraj

        # so I have not done this for now and the entry simtraj[-1,2] is empty.

        # note also that this is very similar to what is being done for dim==0



        # simtraj[-1,2] = meas[T_nm1-t_nstar,3] #this is what the speed is supposed to be but will require outputting simtraj

        shiftedtraj[:, 0] = time

        shift = simtraj[-1, 2] - meas[T_nm1 - t_nstar, 2]

        shiftedtraj[:, [1, 2]] = meas[T_nm1 - t_nstar + 1:, [2, 3]]

        shiftedtraj[:, [1, 2]] = shiftedtraj[:, [1, 2]] + jnp.repeat(shift, T_n - T_nm1)



        pass

    else:  # for now this is only for newell model, dim = 0

        shiftedtraj[:, 0] = time

        shift = simtraj[-1] - meas[

            T_nm1 - t_nstar, 2]  # the shift; note that simtraj[-1] is only going to make sense for the newell model case

        shiftedtraj[:, 1] = meas[T_nm1 + 1 - t_nstar:, 2] + shift



    return shiftedtraj





def OVM(veh, lead, p, leadlen, relax, *args):

    # regime drenotes what the model is in.

    # regime = 0 is reserved for something that has no dependence on parameters: this could be the shifted end, or it could be a model regime that has no dependnce on parameters (e.g. acceleration is bounded)



    regime = 1

    out = jnp.zeros((1, 2))

    # find and replace all tanh, then brackets to paranthesis, then rename all the variables

    out[0, 0] = veh[1]

    out[0, 1] = p[3] * (

            p[0] * (jnp.tanh(p[1] * (lead[0] - leadlen - veh[0] + relax) - p[2] - p[4]) - jnp.tanh(-p[2])) - veh[1])



    # could be a good idea to make another helper function which adds this to the current value so we can constrain velocity?



    return out, regime





def OVMadj(veh, lead, p, leadlen, lam, reg, relax, relaxadj, use_r, *args):

    # args[0] controls which column of lead contains position and speed info

    # args[1] has the length of the lead vehicle

    # args[2] holds the true measurements

    # args[3] holds lambda



    # this is what we integrate to compute the gradient of objective function after solving the adjoint system



    if use_r:  # relaxation

        out = jnp.zeros((1, 6))

        if reg == 1:

            s = lead[0] - leadlen - veh[0] + relax

            out[0, 0] = -p[3] * lam[1] * (jnp.tanh(p[2]) - jnp.tanh(p[2] + p[4] - p[1] * s))

            out[0, 1] = p[0] * p[3] * lam[1] * (-s) * (1 / jnp.cosh(p[2] + p[4] - p[1] * s) ** 2)

            out[0, 2] = -p[0] * p[3] * lam[1] * (1 / jnp.cosh(p[2]) ** 2 - 1 / jnp.cosh(p[2] + p[4] - p[1] * s) ** 2)

            out[0, 3] = -lam[1] * (-veh[1] + p[0] * (jnp.tanh(p[2]) - jnp.tanh(p[2] + p[4] - p[1] * s)))

            out[0, 4] = p[0] * p[3] * lam[1] * (1 / jnp.cosh(p[2] + p[4] - p[1] * s) ** 2)

            out[0, 5] = -relaxadj * p[0] * p[1] * p[3] * lam[1] * (

                    1 / jnp.cosh(p[2] + p[4] - p[1] * (s))) ** 2  # contribution from relaxation phenomenon



    else:

        out = jnp.zeros((1, 5))

        if reg == 1:

            s = lead[0] - leadlen - veh[0]

            out[0, 0] = -p[3] * lam[1] * (jnp.tanh(p[2]) - jnp.tanh(p[2] + p[4] - p[1] * s))

            out[0, 1] = p[0] * p[3] * lam[1] * (-s) * (1 / jnp.cosh(p[2] + p[4] - p[1] * s) ** 2)

            out[0, 2] = -p[0] * p[3] * lam[1] * (1 / jnp.cosh(p[2]) ** 2 - 1 / jnp.cosh(p[2] + p[4] - p[1] * s) ** 2)

            out[0, 3] = -lam[1] * (-veh[1] + p[0] * (jnp.tanh(p[2]) - jnp.tanh(p[2] + p[4] - p[1] * s)))

            out[0, 4] = p[0] * p[3] * lam[1] * (1 / jnp.cosh(p[2] + p[4] - p[1] * s) ** 2)



    return out





def OVMadjsys(veh, lead, p, leadlen, vehstar, lam, curfol, regime, havefol, vehlen, lamp1, folp, relax, folrelax,

              shiftloss, *args):

    # add in the stuff for eqn 22 to work

    # veh = current position and speed of simulated traj

    # lead = current position and speed of leader

    # p = parameters



    # regime 1 = base model

    # regime 0 = shifted end

    # all other regimes correspond to unique pieces of the model



    # note lead and args[2] true measurements need to both be given for the specific time at which we compute the deriv

    # this intakes lambda and returns its derivative.



    out = jnp.zeros((1, 2))



    if regime == 1:  # if we are in the model then we get the first indicator function in eqn 22

        out[0, 0] = 2 * (veh[0] - vehstar[0]) + p[0] * p[1] * p[3] * lam[1] * (

                1 / jnp.cosh(p[2] + p[4] - p[1] * (lead[0] - leadlen - veh[0] + relax))) ** 2

        out[0, 1] = -lam[0] + p[3] * lam[1]



    elif regime == 0:  # otherwise we are in shifted end, we just get the contribution from loss function, which is the same on the entire interval

        out[0, 0] = shiftloss

    else:

        out[0, 0] = 2 * (veh[0] - vehstar[0])

        out[0, 1] = -lam[0]



    if havefol == 1:  # if we have a follower in the current platoon we get another term (the second indicator functino term in eqn 22)

        out[0, 0] += -folp[0] * folp[1] * folp[3] * lamp1[1] * (

                1 / jnp.cosh(folp[2] + folp[4] - folp[1] * (-vehlen + veh[0] - curfol[0] + folrelax))) ** 2



    return out





def shifted_end(simtraj, meas, t_nstar, t_n, T_nm1, T_n, h=.1, dim=2):

    # model can be either dim = 1 (position output only) or dim = 2 (position + speed output)

    # t_n and T_nm1 refer to the starting and ending times of the simulation, so to use a model with delay input the delayed values.

    dim = int(dim)

    shiftedtraj = jnp.zeros((T_n - T_nm1,

                             dim + 1))  # for now we are just going to assume the output should have 3 columns (time, position, speed, respectively)

    time = range(T_nm1 + 1, T_n + 1)



    if dim == 2:  # this is for a second order ode model

        shiftedtraj[:, 0] = time

        shiftedtraj[0, 1] = simtraj[-1, 1] + h * simtraj[

            -1, 2]  # position at next timestep from the velocity the vehicle currently has

        shiftedtraj[0, 2] = meas[

            T_nm1 + 1 - t_nstar, 3]  # speed in the shifted trajectory is actually the same since we are just shifted position

        if T_n - T_nm1 == 1:  # in this case there is nothing else to do but this should be pretty rare

            return shiftedtraj

        shift = shiftedtraj[0, 1] - meas[

            T_nm1 + 1 - t_nstar, 2]  # difference between last position of simulation and measurements

        shiftedtraj[1:, [1, 2]] = meas[T_nm1 + 2 - t_nstar:,

                                  2:4]  # load in the end trajectory taken directly from the data

        shiftedtraj[1:, 1] = shiftedtraj[1:, 1] + jnp.repeat(shift,

                                                             T_n - 1 - T_nm1)  # now shift the position by the required amount



    elif dim == 1:  # dimension 1 models

        # note you might to modify shiftedtraj to fill in the last speed value but this would require shifted_end to output simtraj as well as shiftedtraj

        # so I have not done this for now and the entry simtraj[-1,2] is empty.

        # note also that this is very similar to what is being done for dim==0



        # simtraj[-1,2] = meas[T_nm1-t_nstar,3] #this is what the speed is supposed to be but will require outputting simtraj

        shiftedtraj[:, 0] = time

        shift = simtraj[-1, 2] - meas[T_nm1 - t_nstar, 2]

        shiftedtraj[:, [1, 2]] = meas[T_nm1 - t_nstar + 1:, [2, 3]]

        shiftedtraj[:, [1, 2]] = shiftedtraj[:, [1, 2]] + jnp.repeat(shift, T_n - T_nm1)



        pass

    else:  # for now this is only for newell model, dim = 0

        shiftedtraj[:, 0] = time

        shift = simtraj[-1] - meas[

            T_nm1 - t_nstar, 2]  # the shift; note that simtraj[-1] is only going to make sense for the newell model case

        shiftedtraj[:, 1] = meas[T_nm1 + 1 - t_nstar:, 2] + shift



    return shiftedtraj





def makeleadfolinfo_r3(platoons, platooninfo, sim, use_merge_constant=False):

    # positive and negative r.



    # this will get the leader and follower info to use in the objective function and gradient calculation. this will save time

    # because of the way scipy works, we cant update the *args we pass in to our custom functions, so if we do this preprocessing here

    # it will save us from having to do this over and over again every single time we evaluate the objective or gradient.

    # however, it is still not ideal. in a totally custom implementation our optimization routines wouldn't have to do this at all

    # because we would be able to update the *args

    # additionally, in a totally custom implementation we would make use of the fact that we need to actually evaluate the objective before we can

    # evaluate the gradient. in the scipy implementation, everytime we evaluate the gradient we actually evaluate the objective again, which is totally wasted time.



    # input/output example:

    # input : platoons= [[],5]

    # output : [[[1,1,100]]], [[]] means that vehicle 5 has vehicle 1 as a leader for frame id 1 to frameid 100, and that vehicle 5 has no followers

    # which are in platoons



    # note that you can either pass in sim or meas in the position for sim.



    leadinfo = []  # initialize output

    folinfo = []

    rinfo = []



    for i in platoons[1:]:  # iterate over each vehicle in the platoon

        curleadinfo = []  # for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon

        curfolinfo = []

        currinfo = []

        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]  # get times for current vehicle

        leadlist = sim[i][t_n - t_nstar:T_nm1 - t_nstar + 1,

                   4]  # this gets the leaders for each timestep of the current vehicle\

        curlead = leadlist[0]  # initialize current leader

        curleadinfo.append([curlead, t_n])  # initialization

        for j in range(len(leadlist)):

            if leadlist[j] != curlead:  # if there is a new leader

                newlead = leadlist[j]

                oldlead = curlead

                ##############relaxation constant calculation

                newlead = int(newlead)

                oldlead = int(oldlead)

                newt_nstar = platooninfo[newlead][0]

                oldt_nstar = platooninfo[oldlead][0]

                olds = sim[oldlead][t_n + j - 1 - oldt_nstar, 2] - sim[oldlead][0, 6] - sim[i][

                    t_n + j - 1 - t_nstar, 2]  # the time is t_n+j-1; this is the headway

                news = sim[newlead][t_n + j - newt_nstar, 2] - sim[newlead][0, 6] - sim[i][

                    t_n + j - t_nstar, 2]  # the time is t_n+j

                # below if only adds if headway decreases, otherwise we will always add the relaxation constant, even if it is negative.

                #                if news < olds: #if the headway decreases, then we will add in the relaxation phenomenon

                #                    currinfo.append([t_n+j, olds-news]) #we append the time the LC happens (t_n+j), and the "gamma" which is what I'm calling the initial constant we adjust the headway by (olds-news)

                currinfo.append([t_n + j, olds - news])

                #################################################

                curlead = leadlist[j]  # update the current leader

                curleadinfo[-1].append(t_n + j - 1)  # last time (in frameID) the old leader is observed

                curleadinfo.append([curlead, t_n + j])  # new leader and the first time (in frameID) it is observed.



        curleadinfo[-1].append(t_n + len(leadlist) - 1)  # termination



        # do essentially the same things for followers now (we need the follower for adjoint system)

        # only difference is that we only need to put things in if their follower is in platoons

        follist = sim[i][t_n - t_nstar:T_n - t_nstar + 1, 5]  # list of followers

        curfol = follist[0]

        if curfol in platoons[1:]:  # if the current follower is in platoons we initialize

            curfolinfo.append([curfol, t_n])

        for j in range(len(follist)):  # check what we just made to see if we need to put stuff in folinfo

            if follist[j] != curfol:  # if there is a new follower

                curfol = follist[j]

                if curfolinfo != []:  # if there is anything in curfolinfo

                    curfolinfo[-1].append(t_n + j - 1)  # we finish the interval

                if curfol in platoons[1:]:  # if new follower is in platoons

                    curfolinfo.append([curfol, t_n + j])  # start the next interval

        if curfolinfo != []:  # if there is anything to finish

            curfolinfo[-1].append(t_n + len(follist) - 1)  # finish it

        leadinfo.append(curleadinfo)  # append what we just made to the total list

        folinfo.append(curfolinfo)

        rinfo.append(currinfo)



        if use_merge_constant:

            rinfo = merge_rconstant(platoons, platooninfo, sim, leadinfo, rinfo, 100)



    return leadinfo, folinfo, rinfo





def merge_rconstant(platoons, platooninfo, sim, leadinfo, rinfo, relax_constant=100, merge_from_lane=7, merge_lane=6,

                    datalen=9, h=.1):

    for i in range(len(platoons[1:])):

        curveh = platoons[i + 1]

        t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]

        lanelist = jnp.unique(sim[curveh][:t_n - t_nstar, 7])



        # theres a bug here because the r for merger is not being calculated correctly.

        if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][

            t_n - t_nstar, 7] == merge_lane:  # get a merge constant when a vehicle's simulation starts when they enter the highway #

            lead = jnp.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory

            for j in leadinfo[i]:

                curleadid = j[0]  # current leader ID

                leadt_nstar = int(sim[curleadid][0, 1])  # t_nstar for the current lead, put into int

                lead[j[1] - t_n:j[2] + 1 - t_n, :] = sim[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,

                                                     :]  # get the lead trajectory from simulation

            headway = lead[:, 2] - sim[curveh][t_n - t_nstar:, 2] - lead[:, 6]

            headway = headway[:T_nm1 + 1 - t_n]

            # calculate the average headway when not close to lane changing events

            headwaybool = jnp.ones(len(headway), dtype=bool)

            for j in rinfo[i]:

                headwaybool[j[0] - t_n:j[0] - t_n + relax_constant] = 0



            headway = headway[headwaybool]

            if len(headway) > 0:  # if there are any suitable headways we can use then do it

                preheadway = jnp.mean(headway)



                postlead = sim[curveh][t_n - t_nstar, 4]

                postleadt_nstar = platooninfo[postlead][0]



                posthd = sim[postlead][t_n - postleadt_nstar, 2] - sim[postlead][t_n - postleadt_nstar, 6] - sim[curveh][t_n - t_nstar, 2]



                curr = preheadway - posthd

                rinfo[i].insert(0, [t_n, curr])

            # another strategy to get the headway in the case that there aren't any places we can estimate it from



            else:

                # it never reaches this point unless in a special case

                leadlist = jnp.unique(sim[curveh][:t_n - t_nstar, 4])

                if len(

                        leadlist) > 1 and 0 in leadlist:  # otherwise if we are able to use the other strategy then use that



                    cursim = sim[curveh][:t_n - t_nstar, :].copy()

                    cursim = cursim[cursim[:, 7] == merge_from_lane]

                    cursim = cursim[cursim[:, 4] != 0]



                    curt = cursim[-1, 1]

                    curlead = cursim[-1, 4]

                    leadt_nstar = platooninfo[curlead][0]



                    prehd = sim[curlead][curt - leadt_nstar, 2] - sim[curlead][curt - leadt_nstar, 6] - cursim[

                        -1, 2]  # headway before



                    postlead = sim[curveh][t_n - t_nstar, 4]

                    postleadt_nstar = platooninfo[postlead][0]



                    posthd = sim[postlead][t_n - postleadt_nstar, 2] - sim[postlead][t_n - postleadt_nstar, 6] - \
                             sim[curveh][t_n - t_nstar, 2]

                    curr = prehd - posthd



                    rinfo[i].insert(0, [t_n, curr])

                else:  # if neither strategy can be used then we can't get a merger r constant for the current vehicle.
                    continue
    return rinfo



from jax import grad #we want to use jax's grad function

import pickle

import copy

with open('dataautodiff2.pkl','rb') as f: #meas in jax device array format 
    meas,platooninfo = pickle.load(f)

platoonobjfn_grad = grad(platoonobjfn_obj_jax)
sim_jax = copy.deepcopy(meas)
pguess_jax = jnp.array([10 * 3.3, .086 / 3.3, 1.545, 2, .175, 5.01])
args_jax   = (True,6)
curplatoon_jax = [[],581,611]
n = int(len(curplatoon_jax)-1)
leadinfo_jax,folinfo_jax,rinfo_jax = makeleadfolinfo_r3(curplatoon_jax,platooninfo,meas)
p2 = jnp.tile(pguess_jax, n)

#testobj = platoonobjfn_obj_jax(p2,OVM,OVMadjsys,OVMadj,meas,sim_jax,platooninfo,curplatoon_jax,leadinfo_jax,folinfo_jax,rinfo_jax,*args_jax)

testgrad = platoonobjfn_grad(p2,OVM,OVMadjsys,OVMadj,meas,sim_jax,platooninfo,curplatoon_jax,leadinfo_jax,folinfo_jax,rinfo_jax,*args_jax)

"""
\\ TO DO \\ 
Get the gradient of platoonobjfn_obj using jax. Compare this is the gradient from finite differences, and the gradient from the adjoint method. 

Time for running finite differences is fdertime
Time for running adjoint method is dertime
Accuracy of adjoint method is acc, Relative accuracy of adjoint method is acc2 

For jax, compute the run time to get the gradient. Get the accuracy of jax, and the relative accuracy of jax. 
Please also try to compute the memory requirements of jax and the memory requirements of the adjoint method. Links to an explanation of profiling memory in python is given in the original .pdf
"""

#%% original MWE of calling the platoonobjfn_obj and taking the gradient of it using finite differences and the adjoint method 

#from calibration import platoonobjfn_obj, platoonobjfn_der, platoonobjfn_fder, makeleadfolinfo_r3,OVM, OVMadj, OVMadjsys, r_constant, euler, euleradj, shifted_end #all functinos from main file needed
#import time 
#import copy
#import pickle 
#import numpy as np
##load data you will need 
#with open('dataautodiff.pkl','rb') as f:
#    meas,platooninfo = pickle.load(f)
##define inputs needed 
#sim = copy.deepcopy(meas)
#pguess = [10*3.3,.086/3.3, 1.545, 2, .175, 5.01]
#args   = (True,6)
#curplatoon = [[],581,611]
#n = int(len(curplatoon)-1)
#leadinfo,folinfo,rinfo = makeleadfolinfo_r3(curplatoon,platooninfo,meas)
#p2 = np.tile(pguess,n)
##run the objective and time it
#start = time.time()
#obj = platoonobjfn_obj(p2,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#end = time.time()
#objtime = end-start
##get the gradient using adjoint method and time it
#start = time.time()
#der = platoonobjfn_der(p2,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#end = time.time()
#dertime = end-start
##get the gradient using finite differences
#start = time.time()
#fder = platoonobjfn_fder(p2,OVM,OVMadjsys,OVMadj,meas,sim,platooninfo,curplatoon,leadinfo,folinfo,rinfo,*args)
#end = time.time()
#fdertime = end-start
##compare accurcy of finite difference gradient with adjoint method gradient 
#acc = np.linalg.norm(der-fder)/np.linalg.norm(fder)
#acc2 = np.divide(der-fder,fder)
#print('accuracy in norm is '+str(acc))
#print('relative error in each parameter is '+str(acc2))