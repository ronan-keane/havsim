
"""
traffic models in their function form

@author: rlk268@cornell.edu
"""

from havsim import helper
import math
import numpy as np

def LLobj(p,simtimes, curtime, prev, prevspeed,count, eta, cureta, etaactive, am, leadint, curLC, LClist, h, datadelay ):
    #p[0] = time delay
    #p[1] = shift in space
    #p[2] = free flow speed
    #p[3] = etat
    #p[4] = eta1
    #p[5] = eps

    #need to update call signature so it will work
    temp = p[2]*p[0]
    temp2 = p[2]*p[0] - (1 - math.e**(-p[0]*am/p[2]))*(p[2] - prevspeed)/am*p[2]

    prevshifttime = curtime - cureta*p[0]

    temp3 = leadint(prevshifttime) - cureta*p[1]

    if temp < temp2:
        freeflow = prev + temp
        temp4 = 0
    else:
        freeflow = prev + temp2
        temp4 = 1

    if freeflow < temp3:
        newx = freeflow
        if temp4 == 0:
            reg = 1 #temp4 = 0 when we are in the temp < temp2 case
        else:
            reg = 2
    else:
        newx = temp3
        reg = 3

    if curLC is not None:
        if curtime > curLC: #if this is true we can't calculate the speed
            count = 0
            curLC = LClist.popleft()
            newspeed = prevspeed #need to do something so just repeat previous speed
        else:
            newspeed = (newx-prev)/p[0]
            count += 1
            if count > 1:
                acc = (newspeed - prevspeed)/p[0]
                if acc < 0 and not etaactive:
                    etaactive = True
                    eta = eta_helper(eta,simtimes,curtime,1,p[5],p[3],p[5],p[4],h,datadelay)
                    pass




    return newx, newspeed, reg, eta, count, etaactive, curLC, LClist

def eta_helper(eta,simtimes,start,eta0,eps0,etat,eps1,eta1,h,datadelay):
    #eta is the vector of eta which we keep around for the simulation
    #simtimes is the vector which has the times we simulate at
    #start is the starting time at which eta becomes activated
    #eta0 - starting eta value; we assume this to be 1 in the LL model
    #eps0 - rate of change for eta. it changes this amount every second when transitioning from eta0 to etat.
    #etat - eta_t, the peak value of eta, eta0 changes to etat at a rate eps0.
    #eps1 - rate of change for eta, it changes at this rate from etat to eta1
    #eta1 - the final value of eta, etat changes to eta1 at a rate of eps1
    #h - gives the distance between points in the data

    if etat - eta0 < 0:  #check sign of change
        changesign = -1
    else:
        changesign = 1

    startind = int(round((start-simtimes[0])/datadelay))
    lastind = len(eta)-1

    change = h*datadelay*eps0*changesign #eta changes this amount per discretization of eta
    lenfirst = (etat-eta0) // change  #number of points in first branch of eta amount

    temp = np.linspace(1+change,1+change*lenfirst,lenfirst)

    if eta1 - etat < 0:
        changesign2 = -1
    else:
        changesign2 = 1
    change2 = h*datadelay*eps1*changesign2

    nextpoint = etat + change2*(1 - (etat-temp[-1])/change)
    lenrest = (eta1-etat) // change2

    temp = np.append(temp,nextpoint)
    temp = np.append(temp,np.linspace(nextpoint + change2, nextpoint + change2*lenrest, lenrest))

    lentemp = len(temp)
    maxlen = lastind-startind+1
    if lentemp > maxlen: #entire temp will not fit
        eta[startind:] = temp[:maxlen]
    else:
        eta[startind:startind+lentemp] = temp #put in entire temp
        eta[startind+lentemp:] = eta1 #rest of eta is just equal to eta1.



    return eta

def OVM(veh, lead, p, leadlen, relax, *args):
    #regime drenotes what the model is in.
    #regime = 0 is reserved for something that has no dependence on parameters: this could be the shifted end, or it could be a model regime that has no dependnce on parameters (e.g. acceleration is bounded)


    regime = 1
    out = np.zeros((1,2))
    #find and replace all tanh, then brackets to paranthesis, then rename all the variables
    out[0,0] = veh[1]
    out[0,1] = p[3]*(p[0]*(np.tanh(p[1]*(lead[0]-leadlen-veh[0]+relax)-p[2]-p[4])-np.tanh(-p[2]))-veh[1])


    #could be a good idea to make another helper function which adds this to the current value so we can constrain velocity?

    return out, regime

def OVM_b(veh, lead, p, leadlen, relax, *args):
    #
    #regime drenotes what the model is in.
    #regime = 0 is for shifted end
    #bounded acceleration
    #uses adjoint function adj2


    regime = 1
    out = np.zeros((1,2))
    #find and replace all tanh, then brackets to paranthesis, then rename all the variables
    out[0,0] = veh[1]
    out[0,1] = p[3]*(p[0]*(np.tanh(p[1]*(lead[0]-leadlen-veh[0]+relax)-p[2]-p[4])-np.tanh(-p[2]))-veh[1])



    #can add bounds on acceleration

    if out[0,1] > 4*3.3: #if acceleration more than 4 m/s/s
        out[0,1] = 4*3.3
        regime = 2
    if out[0,1] <-6*3.3: #if deceleration more than 6 m/s/s
        out[0,1] = -6*3.3
        regime = 2




    return out, regime

def OVMadjsys(veh,lead, p, leadlen, vehstar, lam, curfol, regime, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss,*args):
    #add in the stuff for eqn 22 to work
    #veh = current position and speed of simulated traj
    #lead = current position and speed of leader
    #p = parameters

    #regime 1 = base model
    #regime 0 = shifted end
    #all other regimes correspond to unique pieces of the model


    #note lead and args[2] true measurements need to both be given for the specific time at which we compute the deriv
    #this intakes lambda and returns its derivative.

    out = np.zeros((1,2))

    if regime == 1: #if we are in the model then we get the first indicator function in eqn 22
        out[0,0] = 2*(veh[0]-vehstar[0])+p[0]*p[1]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*(lead[0]-leadlen-veh[0]+relax)))**2
        out[0,1] = -lam[0]+p[3]*lam[1]

    elif regime ==0: #otherwise we are in shifted end, we just get the contribution from loss function, which is the same on the entire interval
        out[0,0] = shiftloss
    else:
        out[0,0] = 2*(veh[0]-vehstar[0])
        out[0,1] = -lam[0]

    if havefol == 1: #if we have a follower in the current platoon we get another term (the second indicator functino term in eqn 22)
        out[0,0] += -folp[0]*folp[1]*folp[3]*lamp1[1]*(1/np.cosh(folp[2]+folp[4]-folp[1]*(-vehlen+veh[0]-curfol[0]+folrelax)))**2

    return out


def OVMadj(veh,lead,p,leadlen, lam, reg, relax, relaxadj, use_r,*args):


    #args[0] controls which column of lead contains position and speed info
    #args[1] has the length of the lead vehicle
    #args[2] holds the true measurements
    #args[3] holds lambda


    #this is what we integrate to compute the gradient of objective function after solving the adjoint system


    if use_r:#relaxation
        out = np.zeros((1,6))
        if reg==1:

            s = lead[0]-leadlen-veh[0]+relax
            out[0,0] = -p[3]*lam[1]*(np.tanh(p[2]) - np.tanh(p[2]+p[4]-p[1]*s))
            out[0,1] = p[0]*p[3]*lam[1]*(-s)*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,2] = -p[0]*p[3]*lam[1]*(1/np.cosh(p[2])**2 - 1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,3] = -lam[1]*(-veh[1]+p[0]*(np.tanh(p[2])-np.tanh(p[2]+p[4]-p[1]*s)))
            out[0,4] = p[0]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,5] = -relaxadj*p[0]*p[1]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*(s)))**2 #contribution from relaxation phenomenon

    else:
        out = np.zeros((1,5))
        if reg==1:


            s = lead[0]-leadlen-veh[0]
            out[0,0] = -p[3]*lam[1]*(np.tanh(p[2]) - np.tanh(p[2]+p[4]-p[1]*s))
            out[0,1] = p[0]*p[3]*lam[1]*(-s)*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,2] = -p[0]*p[3]*lam[1]*(1/np.cosh(p[2])**2 - 1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,3] = -lam[1]*(-veh[1]+p[0]*(np.tanh(p[2])-np.tanh(p[2]+p[4]-p[1]*s)))
            out[0,4] = p[0]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)



    return out

def OVMadj2(veh,lead,p,leadlen, lam, reg, relax, relaxadjpos, relaxadjneg, use_r,*args):
    #this is for 2 parameter relaxation

    #args[0] controls which column of lead contains position and speed info
    #args[1] has the length of the lead vehicle
    #args[2] holds the true measurements
    #args[3] holds lambda


    #this is what we integrate to compute the gradient of objective function after solving the adjoint system


    if use_r:#relaxation
        out = np.zeros((1,7))
        if reg==1:

            s = lead[0]-leadlen-veh[0]+relax
            out[0,0] = -p[3]*lam[1]*(np.tanh(p[2]) - np.tanh(p[2]+p[4]-p[1]*s))
            out[0,1] = p[0]*p[3]*lam[1]*(-s)*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,2] = -p[0]*p[3]*lam[1]*(1/np.cosh(p[2])**2 - 1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,3] = -lam[1]*(-veh[1]+p[0]*(np.tanh(p[2])-np.tanh(p[2]+p[4]-p[1]*s)))
            out[0,4] = p[0]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,5] = -relaxadjpos*p[0]*p[1]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*(s)))**2 #contribution from relaxation phenomenon
            out[0,6] = -relaxadjneg*p[0]*p[1]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*(s)))**2 #contribution from relaxation phenomenon


    else:
        out = np.zeros((1,5))
        if reg==1:


            s = lead[0]-leadlen-veh[0]
            out[0,0] = -p[3]*lam[1]*(np.tanh(p[2]) - np.tanh(p[2]+p[4]-p[1]*s))
            out[0,1] = p[0]*p[3]*lam[1]*(-s)*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,2] = -p[0]*p[3]*lam[1]*(1/np.cosh(p[2])**2 - 1/np.cosh(p[2]+p[4]-p[1]*s)**2)
            out[0,3] = -lam[1]*(-veh[1]+p[0]*(np.tanh(p[2])-np.tanh(p[2]+p[4]-p[1]*s)))
            out[0,4] = p[0]*p[3]*lam[1]*(1/np.cosh(p[2]+p[4]-p[1]*s)**2)



    return out

def IDM(veh,lead,p,leadlen,relax,*args):
    #vanilla IDM
    #parameters in order, according to treiber's notation are
    #v0 desired speed
    #desired time headway T
    #minimum spacing s0
    #acceleration sensitivity a
    #comfortable braking deceleration b

    regime = 1
    out = np.zeros((1,2))

    out[0,0] = veh[1]
    out[0,1] = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(lead[0]-leadlen-veh[0]+relax))**2)
#
#    s = lead[0]-leadlen-veh[0]+relax
#    print(s)
#    print(out)

    return out, regime

def IDM_b3(veh,lead,p,leadlen,relax,*args,h=.1):
    #IDM with bounded velocity
    #regime 4 = velocity is constrained by 0

    #note that IDM_b3 will use _b3 functions for adjsys and adj.
    #the _b, _b2, and vanilla use the standard adjsys and adj functions.

    regime = 1
    out = np.zeros((1,2))

    out[0,0] = veh[1]
    out[0,1] = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(lead[0]-leadlen-veh[0]+relax))**2)
#
#    s = lead[0]-leadlen-veh[0]+relax
#    print(s)
#    print(out)
    if veh[1]+h*out[0,1] < 0:
        out[0,1] = -veh[1]/h
        regime =4

    return out, regime

def IDM_b2(veh,lead,p,leadlen,relax,*args):


    regime = 1
    out = np.zeros((1,2))

    temp = veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2))

    if temp < 0:
        regime = 3
        out[0,0] = veh[1]
        out[0,1] = p[3]*(1-(veh[1]/p[0])**4-((p[2])/(lead[0]-leadlen-veh[0]+relax))**2)
    else:
        out[0,0] = veh[1]
        out[0,1] = p[3]*(1-(veh[1]/p[0])**4-((p[2]+temp)/(lead[0]-leadlen-veh[0]+relax))**2)


    return out, regime

def IDM_b(veh,lead,p,leadlen,relax,*args):

    regime = 1
    out = np.zeros((1,2))

    out[0,0] = veh[1]
    out[0,1] = p[3]*(1-(veh[1]/p[0])**4-((p[2]+veh[1]*p[1]+(veh[1]*(veh[1]-lead[1]))/(2*(p[3]*p[4])**(1/2)))/(lead[0]-leadlen-veh[0]+relax))**2)

    if out[0,1] > 4*3.3: #if acceleration more than 4 m/s/s
        out[0,1] = 4*3.3
        regime = 2
    if out[0,1] <-6*3.3: #if deceleration more than 6 m/s/s
        out[0,1] = -6*3.3
        regime = 2

    return out, regime

#at some point should optimize the adjoint systems so they don't keep doing the same calculations over and over

def IDMadjsys(veh,lead,p,leadlen, vehstar, lam, curfol, regime, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss,*args):

    out = np.zeros((1,2))

    if regime ==1:
        s = -leadlen + lead[0] + relax - veh[0]
        out[0,0] = 2*(veh[0]-vehstar[0])+ (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
           (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3
        out[0,1] = -lam[0] - p[3]*lam[1]*(-((4*veh[1]**3)/p[0]**4) - (2*(p[1] + veh[1]/(2*(p[3]*p[4])**.5) +
           (-lead[1] + veh[1])/(2*(p[3]*p[4])**.5))*(p[2] + p[1]*veh[1] +
           (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(-leadlen + lead[0] + relax - veh[0])**2)

    elif regime ==0:
        out[0,0] = shiftloss

    elif regime ==3:
        s = -leadlen + lead[0] + relax - veh[0]
        out[0,0] = (2*p[2]**2*p[3]*lam[1])/(s)**3
        out[0,1] = -lam[0] + (4*p[3]*lam[1]*veh[1]**3)/p[0]**4

    else: #regime = 2
        out[0,0] = 2*(veh[0]-vehstar[0])
        out[0,1] = -lam[0]

    if havefol==1:
        out[0,0] += -((2*folp[3]*lamp1[1]*(folp[2] + folp[1]*curfol[1] +
           (curfol[1]*(-veh[1] + curfol[1]))/(2*(folp[3]*folp[4])**.5))**2)/(-vehlen + veh[0] + folrelax - curfol[0])**3)
        out[0,1] += -((folp[3]*lamp1[1]*curfol[1]*(folp[2] + folp[1]*curfol[1] +
           (curfol[1]*(-veh[1] + curfol[1]))/(2*(folp[3]*folp[4])**.5)))/((folp[3]*folp[4])**.5*(-vehlen + veh[0] + folrelax - curfol[0])**2))

    elif havefol==3:
        out[0,0] += -((2*p[2]**2*p[3]*lamp1[1])/(-leadlen + lead[0] + relax - veh[0])**3)

    return out

def IDMadjsys_b3(veh,lead,p,leadlen, vehstar, lam, curfol, regime, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss,*args, h=.1):

    out = np.zeros((1,2))

    if regime ==1:
        s = -leadlen + lead[0] + relax - veh[0]
        out[0,0] = 2*(veh[0]-vehstar[0])+ (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
           (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3
        out[0,1] = -lam[0] - p[3]*lam[1]*(-((4*veh[1]**3)/p[0]**4) - (2*(p[1] + veh[1]/(2*(p[3]*p[4])**.5) +
           (-lead[1] + veh[1])/(2*(p[3]*p[4])**.5))*(p[2] + p[1]*veh[1] +
           (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(-leadlen + lead[0] + relax - veh[0])**2)

    elif regime ==0:
        out[0,0] = shiftloss

#    elif regime ==3:  #old version here for regimes 2 and 3.
#        s = -leadlen + lead[0] + relax - veh[0]
#        out[0,0] = (2*p[2]**2*p[3]*lam[1])/(s)**3
#        out[0,1] = -lam[0] + (4*p[3]*lam[1]*veh[1]**3)/p[0]**4
#
#    else: #regime = 2
#        out[0,0] = 2*(veh[0]-vehstar[0])
#        out[0,1] = -lam[0]

    else: #regime = 4 #velocity constrained
        out[0,0] = 2*(veh[0]-vehstar[0])
        out[0,1] = -lam[0] + lam[1]/h

    if havefol==1:
        out[0,0] += -((2*folp[3]*lamp1[1]*(folp[2] + folp[1]*curfol[1] +
           (curfol[1]*(-veh[1] + curfol[1]))/(2*(folp[3]*folp[4])**.5))**2)/(-vehlen + veh[0] + folrelax - curfol[0])**3)
        out[0,1] += -((folp[3]*lamp1[1]*curfol[1]*(folp[2] + folp[1]*curfol[1] +
           (curfol[1]*(-veh[1] + curfol[1]))/(2*(folp[3]*folp[4])**.5)))/((folp[3]*folp[4])**.5*(-vehlen + veh[0] + folrelax - curfol[0])**2))

#    elif havefol==3:
#        out[0,0] += -((2*p[2]**2*p[3]*lamp1[1])/(-leadlen + lead[0] + relax - veh[0])**3)

    return out

def IDMadj(veh,lead,p,leadlen, lam, reg, relax, relaxadj, use_r,*args):

    if use_r:#relaxation
        out = np.zeros((1,6))
        if reg==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))
            out[0,5] = -relaxadj*((2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3)

        elif reg==3:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,2] = (2*p[2]*p[3]*lam[1])/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 - p[2]**2/(s)**2)
            out[0,5] = -relaxadj*((2*p[2]**2*p[3]*lam[1])/(s)**3)

    else:
        out = np.zeros((1,5))
        if reg ==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))

        elif reg==3:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,2] = (2*p[2]*p[3]*lam[1])/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 - p[2]**2/(s)**2)

    return out

def IDMadj_b3(veh,lead,p,leadlen, lam, reg, relax, relaxadj, use_r,*args):

    if use_r:#relaxation
        out = np.zeros((1,6))
        if reg==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))
            out[0,5] = -relaxadj*((2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3)

    else:
        out = np.zeros((1,5))
        if reg ==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))


    return out


def IDMadj2(veh,lead,p,leadlen, lam, reg, relax, relaxadjpos, relaxadjneg, use_r,*args):

    if use_r:#relaxation
        out = np.zeros((1,6))
        if reg==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))
            out[0,5] = -relaxadjpos*((2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3)
            out[0,6] = -relaxadjneg*((2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3)

        elif reg==3:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,2] = (2*p[2]*p[3]*lam[1])/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 - p[2]**2/(s)**2)
            out[0,5] = -relaxadjpos*((2*p[2]**2*p[3]*lam[1])/(s)**3)
            out[0,5] = -relaxadjneg*((2*p[2]**2*p[3]*lam[1])/(s)**3)

    else:
        out = np.zeros((1,5))
        if reg ==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))

        elif reg==3:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,2] = (2*p[2]*p[3]*lam[1])/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 - p[2]**2/(s)**2)

    return out

def IDMadj2_b3(veh,lead,p,leadlen, lam, reg, relax, relaxadjpos, relaxadjneg, use_r,*args):

    if use_r:#relaxation
        out = np.zeros((1,7))
        if reg==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))
            out[0,5] = -relaxadjpos*((2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3)
            out[0,6] = -relaxadjneg*((2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2)/(s)**3)


    else:
        out = np.zeros((1,5))
        if reg ==1:
            s = -leadlen + lead[0] + relax - veh[0]
            out[0,0] = -((4*p[3]*lam[1]*veh[1]**4)/p[0]**5)
            out[0,1] = (2*p[3]*lam[1]*veh[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,2] = (2*p[3]*lam[1]*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(s)**2
            out[0,3] = (-lam[1])*(1 - veh[1]**4/p[0]**4 + (p[3]*p[4]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2) -
               (p[2] + p[1]*veh[1] + (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5))**2/(s)**2)
            out[0,4] = -((p[3]**2*lam[1]*veh[1]*(-lead[1] + veh[1])*(p[2] + p[1]*veh[1] +
               (veh[1]*(-lead[1] + veh[1]))/(2*(p[3]*p[4])**.5)))/(2*(p[3]*p[4])**(3/2)*(s)**2))

    return out

def daganzo(veh,lead,p,leadlen,relax,*args):
    #need to fully implement first order models into platoonobjfn - currently have a problem with the last point with shifted end
    regime = 1
    out = np.zeros((1,2))

    temp = (lead[0]-leadlen-veh[0]+relax-p[1])/p[0]

    if temp < p[2]:
        out[0,0] = temp
    else:
        out[0,0] = p[2]
        regime = 2


    return out, regime

def daganzoadjsys(veh,lead, p, leadlen, vehstar, lam, curfol, regime, havefol, vehlen, lamp1, folp, relax, folrelax, shiftloss,*args):
    #add in the stuff for eqn 22 to work
    #veh = current position and speed of simulated traj
    #lead = current position and speed of leader
    #p = parameters

    #the horror of my initial parameter names is given below oh jeez it is bad

    #args[0] controls which column of lead contains position and speed info
    #args[1] has the length of the lead vehicle
    #args[2] holds the true measurements
    #args[3] holds lambda

    #note lead and args[2] true measurements need to both be given for the specific time at which we compute the deriv
    #this intakes lambda and returns its derivative.

    out = np.zeros((1,2))

    if regime == 1: #if we are in the model then we get the first indicator function in eqn 22
        out[0,0] = 2*(veh[0]-vehstar[0])+lam[0]/p[0]

    elif regime ==0: #otherwise we are in shifted end, we just get the contribution from loss function, which is the same on the entire interval
        out[0,0] = shiftloss

    if havefol == 1: #if we have a follower in the current platoon we get another term (the second indicator functino term in eqn 22)
        out[0,0] += -lamp1[0]/folp[0]

    return out

def daganzoadj(veh,lead,p,leadlen, lam, reg, relax, relaxadj, use_r,*args):

    if use_r:
        if reg == 1:
            out = np.zeros((1,4))
            s = lead[0]-leadlen-veh[0]+relax
            out[0,0] = lam[0]*(s-p[1])/(p[0]**2)
            out[0,1] = lam[0]/p[0]
            out[0,3] = -relaxadj*lam[0]/p[0]
        elif reg==2:
            out = np.zeros((1,4))
            s = lead[0]-leadlen-veh[0]+relax
            out[0,2] = -lam[0]
            out[0,3] = -relaxadj*lam[0]/p[0]

    else:
        if reg == 1:
            out = np.zeros((1,3))
            s = lead[0]-leadlen-veh[0]+relax
            out[0,0] = lam[0]*(s-p[1])/(p[0]**2)
            out[0,1] = lam[0]/p[0]
        elif reg==2:
            out = np.zeros((1,3))
            s = lead[0]-leadlen-veh[0]+relax
            out[0,2] = -lam[0]

    return out

def zhangkim(p,curveh, prevveh, prevlead, prevleadlen, prevrelax):
    #refer to car following theory for multiphase vehicular traffic flow by zhang and kim, model c
    #delay is p[0], then in order, parameters are S0, S1, v_f, h_1
    #recall reg = 0 is reserved for shifted end

    out = np.zeros((1,2))
    s = prevlead[0] - prevveh[0] - prevleadlen + prevrelax

    if s >= p[2]:
        out[0,0] = p[3]
        reg = 1

    elif s < p[1]:
        out[0,0] = s/p[4]
        reg = 2

    else:
        if prevlead[1] >= p[3]:
            out[0,0] = p[3]
            reg = 1
        else:
            out[0,0] = s/p[4]
            reg = 2



    return out, reg


##these I don't even think are usable anymore
#
#def mae_dist(meas,sim, platooninfo, dim = 2, h = .1):
#    #we will sum up the distance error from t_n to T_nm1+1 for dim =2 or t_n to T_nm1 for dim = 1
#    #meas, sim, platooninfo are all for the corresponding vehicle
#    #calculates average error in distance (MAE in distance)
#    t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
#    lenloss = T_nm1-t_n+dim #len of loss
#    loss = np.zeros(lenloss)
#    sim = sim[t_n-t_nstar:,:]
#    meas = meas[t_n-t_nstar:,:]
#    for i in range(lenloss):
#        loss[i] = abs(sim[i,2]-meas[i,2])
#    ans = sum(loss)/lenloss
#    return ans
#
#def mae_speed(meas,sim, platooninfo, dim = 2, h = .1):
#    #we will sum up the speed error from t_n to T_nm1 for dim =2 or t_n to T_nm1-1 for dim = 1
#    #meas, sim, platooninfo are all for the corresponding vehicle
#    #calculates average error in distance (MAE in distance)
#    t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
#    lenloss = T_nm1-t_n+dim-1 #len of loss
#    loss = np.zeros(lenloss)
#    sim = sim[t_n-t_nstar:,:]
#    meas = meas[t_n-t_nstar:,:]
#    for i in range(lenloss):
#        loss[i] = abs(sim[i,3]-meas[i,3])
#    ans = sum(loss)/lenloss
#    return ans
#
#def rmse_dist(meas,sim, platooninfo, dim = 2, h = .1):
#    #we will sum up the distance error from t_n to T_nm1+1 for dim =2 or t_n to T_nm1 for dim = 1
#    #meas, sim, platooninfo are all for the corresponding vehicle
#    #calculates average error in distance (MAE in distance)
#    t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
#    lenloss = T_nm1-t_n+dim #len of loss
#    loss = np.zeros(lenloss)
#    sim = sim[t_n-t_nstar:,:]
#    meas = meas[t_n-t_nstar:,:]
#    for i in range(lenloss):
#        loss[i] = (sim[i,2]-meas[i,2])**2
#    ans = (sum(loss)/lenloss)**.5
#    return ans
#
#def rmse_speed(meas,sim, platooninfo, dim = 2, h = .1):
#    #we will sum up the distance error from t_n to T_nm1+1 for dim =2 or t_n to T_nm1 for dim = 1
#    #meas, sim, platooninfo are all for the corresponding vehicle
#    #calculates average error in distance (MAE in distance)
#    t_nstar, t_n, T_nm1, T_n = platooninfo[0:4]
#    lenloss = T_nm1-t_n+dim-1 #len of loss
#    loss = np.zeros(lenloss)
#    sim = sim[t_n-t_nstar:,:]
#    meas = meas[t_n-t_nstar:,:]
#    for i in range(lenloss):
#        loss[i] = (sim[i,3]-meas[i,3])**2
#    ans = (sum(loss)/lenloss)**.5
#    return ans

