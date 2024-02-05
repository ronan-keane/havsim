"""old models file

models USED TO have the following call signature:
    veh - list of state of vehicle model is being applied to
    lead - list of state for vehicle's leader
    p - parameters
    leadlen - length of lead vehicle
    *args - additional inputs should be stored in modelinfo dict, and passed through *args
    dt = .1 - timestep

    they return the derivative of the state i.e. how to update in the next timestep
"""

def IDM_b3(p, veh, lead, *args,dt=.1):
    #state is defined as [x,v,s] triples
    #IDM with bounded velocity for circular road

    outdx = veh[1]

    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(veh[2]))**2)


    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt

    return [outdx, outddx]

def IDM_b3_b(p, veh, lead, *args,dt=.1):
    #state is defined as [x,v,s] triples
    #IDM with bounded velocity and acceleration

    outdx = veh[1]

    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(veh[2]))**2)


    #bounded acceleration in m/s/s units
    if outddx > 3:
        outddx = 3
    elif outddx < -7:
        outddx = -7

    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt




    return [outdx, outddx]

def IDM_b3_sh(p, veh, lead, *args,dt=.1):
    #state is defined as [x,v,s] triples
    #IDM with bounded velocity for circular road
    #we add a speed harmonization thing that says you can't go above w/e

    #old code
#    s = lead[0]-leadlen-veh[0]
#    if s < 0: #wrap around in circular can cause negative headway values; in this case we add an extra L to headway
#        s = s + args[0]
    #check if need to modify the headway
#    if args[1]:
#        s = s + args[0]

    outdx = veh[1]
    outddx = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(veh[2]))**2)

    if veh[1]+dt*outddx < 0:
        outddx = -veh[1]/dt

    if veh[1] + dt*outddx > p[5]:
        outddx = (p[5] - veh[1])/dt


    return [outdx, outddx]

def mobil_dis(plc, p, pfold, pfnew, placeholder): #discretionary lane changing following mobil model
    pass

def IDM_b3_eql(p, s, v, find = 's', maxs = 1e4):
    #finds equilibrium solution for s or v, given the other

    #find = s - finds equilibrium headway (s) given speed v,
    #find = v - finds equilibrium velocity (v) given s

    if find == 's':
        s = ((p[2]+p[1]*v)**2/(1- (v/p[0])**4))**.5
        return s
    if find == 'v':
        eqlfun = lambda x: ((p[2]+p[1]*x)**2/(1- (x/p[0])**4))**.5 - s
        v = sc.bisect(eqlfun, 0, maxs)
        return v

def FS(p, veh, lead, *args, dt = .1):
    #follower stopper control model
    #7 parameters, we have U (desired velocity) as a parameter
    dv = (lead[1] - veh[1])
    if dv > 0:
        dv = 0
    dv = dv**2
    usev = min(lead[1],p[6])
    #determine model regime
    dx1 = p[3] + .5/p[0]*dv
    if veh[2] < dx1:
        vcmd = 0
    else:
        dx2 = p[4] + .5/p[1]*dv
        if veh[2] < dx2:
            vcmd = usev*(veh[2]-dx1)/(dx2-dx1)
        else:
            dx3 = p[5] + .5/p[2]*dv
            if veh[2] < dx3:
                vcmd = usev + (p[6] - usev)*(veh[2]-dx2)/(dx3-dx2)
            else:
                vcmd = p[6]

    outdx = veh[1]
    outddx = p[7]*(vcmd - veh[1])

    if outddx > 2:
        outddx = 2
    elif outddx < -5:
        outddx = -5

    return [outdx, outddx]

def linearCAV(p,veh,lead,*args, dt = .1):
    #p[0] = jam distance
    #p[1] = headway slope
    #p[2] = max speed
    #p[3] = sensitivity parameter for headway feedback
    #p[4] - initial beta value - sensitivity for velocity feedback
    #p[5] - short headway
    #p[6] -large headway

    Vs = p[1]*(veh[2]-p[0]) #velocity based on triangular FD
    #determine model regime
    if Vs > p[2]:
        Vs = p[2]
    if lead[1] > p[2]:
        lv = p[2]
    else:
        lv = lead[1]
    if veh[2] > p[6]:
        A = 1
        B = 0
    else:
        A = p[3]
        if veh[2] > p[5]:
            B = (1 - (veh[2]-p[5])/(p[6]-p[5]))*p[4]
        else:
            B = p[4]

    outdx = veh[1]
    outddx = A*(Vs - veh[1]) + B*(lv - veh[1])

    return [outdx, outddx]

def PIS(p, veh, lead, *args, dt = .1):
    #partial integrator with saturation control model.
    pass

def zhangkim(p, s, leadv):
    #refer to car following theory for multiphase vehicular traffic flow by zhang and kim, model c
    #delay is p[0], then in order, parameters are S0, S1, v_f, h_1
    #recall reg = 0 is reserved for shifted end
    #s is the delayed headway, i.e. lead(t - tau) - lead_len - x(t - tau)
    #leadv is the delayed velocity

    if s >= p[2]:
        out = p[3]

    elif s < p[1]:
        out = s/p[4]

    else:
        if leadv[1] >= p[3]:
            out = p[3]
        else:
            out = s/p[4]

    return out



def sv_obj(sim, auxinfo):
    #maximize squared velocity = sv
    obj = 0
    for i in sim.keys():
        for j in sim[i]: #squared velocity
            obj = obj - j[1]**2
            if j[2] < .2: #penalty for having very small or negative headway.
                obj = obj + 2**(-5*(j[2]-.2)) - 1 #-1 guarentees piecewise smooth which is condition for continuous gradient
    return obj

def l2v_obj(sim,auxinfo):
    #minimize the l2 norm of the velocity
    #there is also a term that encourages the average velocity to be large.
    obj = 0
    for i in sim.keys():
        vlist = []
        for j in sim[i]:
            vlist.append(j[1])
        avgv = np.mean(vlist)
        vlist = np.asarray(vlist)
        curobj = vlist - avgv
        curobj = np.square(curobj)
        curobj = sum(curobj)
        obj  = obj + curobj
        obj = obj - len(vlist)*avgv
    return obj

def drl_reward(nextstate, vavg, decay = .95, penalty = 1):
#    reward = 0
    for i in [8,9,10]:
#        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
        vavg[i]= nextstate[i][1]
#        cur = -(nextstate[i][1] - vavg[i])**2 + vavg[i]  #penalize oscillations, reward high average
#        if nextstate[i][2] < 1:
#            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
#            cur = cur - penalty*(nextstate[i][2]-1)**2 #here is a smaller penalty
#        reward += cur
    return 1+(vavg[9]+vavg[10])/10, vavg

def drl_reward4(nextstate, vavg, decay = .95, penalty = 1):
    reward = []
    for i in nextstate.keys():
        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
        cur = -(nextstate[i][1] - vavg[i])**2 + vavg[i]  #penalize oscillations, reward high average
        if nextstate[i][2] < .2:
#            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
            cur = cur - penalty*(nextstate[i][2]-.2)**2 #here is a smaller penalty
        reward.append(cur)
    return np.average(reward)+4, vavg

def drl_reward2(nextstate, vavg, decay = .95, lowereql = 11, eql = 15):
    vlist = [i[1] for i in nextstate.values()]
    vavg = np.average(vlist)
    return 1+ np.interp(vavg, (lowereql, eql), (0, 1)), vavg

def drl_reward3(nextstate, vavg, decay = .95, lowereql = 0, eql = 15):
    vlist = [i[1] for i in nextstate.values()]
    vavg = np.average(vlist)
    return 1+ 1/30*vavg**3, vavg

def drl_reward5(nextstate, vavg, decay = .95, lowereql = 0, eql = 15, avid = 9):
#    vlist = [i[1] for i in nextstate.values()]
    for i in [avid]:
        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1]
    return 1+1/15*nextstate[avid][1]**2 - 1/5*(nextstate[avid][1] - vavg[avid])**2 , vavg

def drl_reward8(nextstate, vavg, decay = .95, lowereql = 11, eql = 15, avid = 9):
    vlist = [i[1] for i in nextstate.values()]
    vavg = np.average(vlist)
    hdpenalty = 0
    hd = nextstate[avid][2]
    if hd < 1.75:
        hdpenalty = -1.5
    return 1+ np.interp(vavg, (lowereql, eql), (0, 5)) + vlist[avid] +hdpenalty, vavg #1 + np.interp(vavg, (lowereql, eql), (0, 1)) + vlist[avid],

def drl_reward88(nextstate, vavg, decay = .95, lowereql = 11, eql = 15, avid = 9):
    vlist = [i[1] for i in nextstate.values()]
    vavg[avid] = vavg[avid]*decay + (1 - decay)*nextstate[avid][1] #update exponential average of velocity
#    vavg = np.average(vlist)
    #constant reward + strictly positive reward when going 15 + strictly positive reward for minimizing oscillations
#    return 1 + (-(min([vlist[avid],30]) - 15)**2+15**2)/30 + (-(min([vavg[avid],30]) - 15)**2+15**2)/30, vavg
    return 1 + (-(min([vlist[avid],30]) - 15)**2+15**2)/30, vavg
#term to penalize oscillations?
#    return 1 + (-(min([vlist[avid],30]) - 15)**2+15**2)/30 + (-(vlist[avid] - vavg[avid])**2+15**2)/30, vavg

def drl_reward7(nextstate, vavg, decay = .95, lowereql = 0, eql = 15, avid = 9):
#    vlist = [i[1] for i in nextstate.values()]
#    vavg[avid] =  = np.average(vlist)
    return 1+1/5*nextstate[avid][1], vavg


def drl_reward6(nextstate, vavg, decay = .95, penalty = 1):
    reward = 0
    for i in [9]:
        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
        cur = -5*(nextstate[i][1] - vavg[i])**2 + vavg[i]**2  #penalize oscillations, reward high average
        if nextstate[i][2] < .2:
#            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
            cur = cur - penalty*(nextstate[i][2]-.2)**2 #here is a smaller penalty
        reward += cur
    return reward, vavg

def drl_reward9(nextstate, vavg, decay = .95, penalty = 1):
    reward = 0
    a = []
#    for i in [9]:
#        vavg[i] = vavg[i]*decay + (1 - decay)*nextstate[i][1] #update exponential average of velocity
#        cur = -5*(nextstate[i][1] - vavg[i])**2 + vavg[i]**2  #penalize oscillations, reward high average
#        if nextstate[i][2] < .2:
##            cur = cur - penalty*(2**(-5*(nextstate[i][2]-.2)) - 1) #if headway is small, we give a penalty
#            cur = cur - penalty*(nextstate[i][2]-.2)**2 #here is a smaller penalty
#        reward += cur
#    for i in [8,9,10]:
#        reward += (max(nextstate[i][1],0)-15)**2
#    return 1 - (reward**.5)/25.9807, vavg #25.9807 = ((15**2)*3)**.5 => max reward at all 15 speed, mapped to 0,1
    for i in [8,9,10]:
        reward += abs(nextstate[i][1]-15)/15
        a.append(nextstate[i][1])
    return 1.1 - reward/3 + np.average(a)/15, None


def avgv_obj(sim,auxinfo):
    obj = 0
    for i in sim.keys():
        for j in sim[i]:
            obj = obj - j[1]  # maximize average velocity
    return obj

# def sv_obj(sim, auxinfo, cons = 1e-4):
#     #maximize squared velocity = sv
#     obj = 0
#     for i in sim.keys():
#         for j in sim[i]: #squared velocity
#             obj = obj - j[1]**2
#     obj = obj * cons
#     for i in sim.keys():
#         for j in range(len(sim[i])): #penality for collisions
#             lead = auxinfo[i][1]
#             leadx = sim[lead][j][0]
#             leadlen = auxinfo[lead][0]
#             s = leadx - leadlen - sim[i][j][0]
#             if s < .2:
#                 obj = obj + 2**(-5*(s-.2)) - 1
#     return obj


"""
in general, I think it is better to just manually solve for the equilibrium solution when possible
instead of using root finding on the model naively.
I also think eql might be better if it uses bisection instead of newton since bisection
is more robust, and assuming we are just looking for headway or velocity
we can give bracketing bounds based on intuition
"""
def eql(model, v, p, length, tol=1e-4):
    # finds equilibrium headway for a given speed and parameters value for second order model
    # there should only be a single root
    def wrapperfun(x):
        dx, ddx = model(0, v, x, v, p, length)
        return ddx

    guess = 10
    headway = 0
    try:
        # headway = sc.newton(wrapperfun, x0=guess,maxiter = 50)
        headway = sc.bisect(wrapperfun, 0, 1e4)
    except RuntimeError:
        pass
    counter = 0

    while abs(wrapperfun(headway)) > tol and counter < 20:
        guess = guess + 10
        try:
            headway = sc.newton(wrapperfun, x0=guess, maxiter=50)
        except RuntimeError:
            pass
        counter = counter + 1

        testout = model(0, v, headway, v, p, length)
        if testout[1] == 0:
            return headway

    return headway

def noloss(*args):
    pass


def dboundary(speed, veh, dt):
    acc = (speed - veh[1])/dt
    if acc > 3:
        acc = 3
    elif acc < -7:
        acc = -7
    return [veh[1], acc]
