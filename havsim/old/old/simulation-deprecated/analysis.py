
"""
just some random stuff, it's kind of buggy and not really useful at all. 


@author: rlk268@cornell.edu
"""



def xferIDM(eqx,eqv,p,w): 
    #returns the transfer function for the IDM model in car following regime with 
    #eqx, eqv being the equilibrium solution, p the parameter, w the frequency
    c1,c2,c3,c4,c5 = p[0],p[1],p[2],p[3],p[4]
    
    out = ((2 *c4* (c3 + c2 *eqv)**2)/eqx**3 + 
           (1j* c4 * eqv* (c3 + c2 *eqv)* w)/((c4* c5)**.5 *eqx**2))/((2* c4 *(c3 + c2 *eqv)**2)/eqx**3 - 
           1j* c4 *(-((4 *eqv**3)/c1**4) -(2 *(c3 + c2* eqv) *(c2 + eqv/(2 *(c4 *c5)**.5)))/eqx**2)* w - w**2)
    
    return out

def xferOVM(eqx,eqv,p,w):
    c1,c2,c3,c4,c5 = p[0],p[1],p[2],p[3],p[4]
    
    out = (c1 *c2 *c4*(1/math.cosh(c3 + c5 - c2 *eqx)**2))/(1j* c4* w -
          w**2 + c1 *c2* c4* (1/math.cosh(c3 + c5 - c2* eqx)**2))
    
    return out 

def xferOVM_l2(eqx,eqv,p,w):
    c1,c2,c3,c4 = p[0],p[1],p[2],p[3]
    
    out =  (c1 *c2)/(c1 *c2 + 1j*c2* w - w**2)
    
    return out 

def linearapprox(veh,eqx,eqv,p,xferfn):
    #first does a FFT transform of a lead vehicle around the equilibrium speed and headway given. 
    #then computes the transfer function and computes the linear approximation of the 
    #response of the leading vehicle, which has parameters p.
    
    #inputs - 
    #veh - lead vehicle 
    #eqx, eqv - equilibrium headway and velocity of the follower
    #note: leader and follower equilibrums can be different
    #p - parametres of the following vehicle. 
    
    #outputs - 
    #fft of the following vehicle 
    #(predicted) trajectory of the following vehicle 
    
    leadx = veh.leader[-1].x
    x0 = leadx[0]
    dt = veh.leader[-1].dt
    x1 = veh.x[0]
    
    eqtraj = np.arange(0,len(leadx)*dt*eqv,dt*eqv) #equilibrium trajectory 
    
    leadeq = np.array(leadx) - x0 - eqtraj #perturbations around the equilibrium solution
    
#    plt.plot(leadx)
#    plt.figure()
#    plt.plot(leadeq)
    
    fft = np.fft.rfft(leadeq)
    N = len(leadx)
    
    fftfol = []
    for i in range(len(fft)):
        w = 2*math.pi*i/N
        wfol = xferfn(eqx, eqv, p, w)
        wfol = (.1+.9j)
        fftfol.append(wfol*fft[i])
        
    folx = np.fft.irfft(fftfol)
    
    folx = folx + x1 + eqtraj
    
    return folx, fft, fftfol

def linearapprox2(veh,eqx,eqv,p,xferfn):
    #first does a FFT transform of a lead vehicle around the equilibrium speed and headway given. 
    #then computes the transfer function and computes the linear approximation of the 
    #response of the leading vehicle, which has parameters p.
    
    #inputs - 
    #veh - lead vehicle 
    #eqx, eqv - equilibrium headway and velocity of the follower
    #note: leader and follower equilibrums can be different
    #p - parametres of the following vehicle. 
    
    #outputs - 
    #fft of the following vehicle 
    #(predicted) trajectory of the following vehicle 
    
    leadx = veh.leader[-1].dx
    x0 = leadx[0]
    dt = veh.leader[-1].dt
    x1 = veh.x[0]
    
#    eqtraj = np.arange(0,len(leadx)*dt*eqv,dt*eqv) #equilibrium trajectory 
    
    leadeq = np.array(leadx) - eqv #perturbations around the equilibrium solution
    
#    plt.plot(leadx)
#    plt.figure()
#    plt.plot(leadeq)
    
    fft = np.fft.rfft(leadeq)
    N = len(leadx)
    
    fftfol = []
    for i in range(len(fft)):
        w = 2*math.pi*i/N
        wfol = xferfn(eqx, eqv, p, w)
#        wfolnew = wfol -.5j
#        wfolnew = wfolnew* abs(wfol)/abs(wfolnew)
        fftfol.append(wfol*fft[i])
        
    folx = np.fft.irfft(fftfol)
    
    folx = folx + eqv
    
    return folx, fft, fftfol
