
"""
@author: rlk268@cornell.edu


"""

import numpy as np 
import scipy.optimize as sc 

def IDM_b3(vehx, vehdx, leadx, leaddx, p,leadlen, *args,dt=.1):
    #IDM with bounded velocity 

    outdx = vehdx
    outddx = p[3]*(1-(vehdx/p[0])**4-((p[2]+vehdx*p[1]+(vehdx*(vehdx-leaddx))/(2*(p[3]*p[4])**(1/2)))/(leadx-leadlen-vehx))**2)
    
#    

    if vehdx+dt*outddx < 0:
        outddx = -vehdx/dt
    
    return outdx, outddx

def linearCAV(vehx, vehdx, leadx, leaddx, p, leadlen, *args, dt = .1):
    #p[0] = jam distance
    #p[1] = headway slope
    #p[2] = max speed
    #p[3] = sensitivity parameter for headway feedback
    #p[4] - initial beta value - sensitivity for velocity feedback
    #p[5] - short headway 
    #p[6] -large headway
    s = leadx - vehx - leadlen
    veh = [vehx, vehdx, s]
    lead = [leadx, leaddx]
    
    
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
    
    return outdx, outddx

def IDM_b(vehx, vehdx, leadx, leaddx, p,leadlen, *args,dt=.1):
    #IDM with bounded velocity and acceleration

    outdx = vehdx
    outddx = p[3]*(1-(vehdx/p[0])**4-((p[2]+vehdx*p[1]+(vehdx*(vehdx-leaddx))/(2*(p[3]*p[4])**(1/2)))/(leadx-leadlen-vehx))**2)
    
#    
    if outddx > 4*3.3: 
        outddx = 4*3.3
    elif outddx < -4*3.3: 
        outddx = -4*3.3

    if vehdx+dt*outddx < 0:
        outddx = -vehdx/dt
        
    
    
    return outdx, outddx

def OVM_b(vehx, vehdx, leadx, leaddx, p,leadlen, *args,dt=.1): 
    #OVm with bounded acceleration and velocity 
    

    outdx = vehdx
    outddx = p[3]*(p[0]*(np.tanh(p[1]*(leadx-leadlen-vehx)-p[2]-p[4])-np.tanh(-p[2]))-vehdx)
    
    
    
    #can add bounds on acceleration 
    
    if outddx > 4*3.3: #if acceleration more than 4 m/s/s
        outddx= 4*3.3
    elif outddx <-6*3.3: #if deceleration more than 6 m/s/s
        outddx = -6*3.3
     
    if vehdx+dt*outddx < 0:
        outddx = -vehdx/dt
        
    
    return outdx, outddx

def OVM_lb(vehx, vehdx, leadx, leaddx, p,leadlen, *args,dt=.1): 
    #OVm with a linear optimal velocity function with velocity difference term 
    #optimal velocity function looks like triangular fundamental diagram in speed-headway 
    
    #p[2] is sensitivity parameter, p[0] is jam headway, p[1] is the slope of the triangular part, p[3] is a cap on max speed, and p[4] is sensitivity for velocity difference
    outdx = vehdx
    outddx = p[2]*(p[1]*(leadx-leadlen-vehx-p[0])-vehdx) + p[4]*(leaddx-vehdx)
    
    
    
    #can add bounds on acceleration 
    
    if outddx > 4*3.3: #if acceleration more than 4 m/s/s
        outddx= 4*3.3
    elif outddx <-6*3.3: #if deceleration more than 6 m/s/s
        outddx = -6*3.3
     
    #velocity bounds
    if vehdx+dt*outddx < 0:
        outddx = -vehdx/dt
    if vehdx+dt*outddx > p[3]: #bounded velocity by p[3]
        outddx = (p[3]-vehdx)/dt
        
    
    return outdx, outddx

def OVM_lb2(vehx, vehdx, leadx, leaddx, p,leadlen, *args,dt=.1): 
    #OVm with a linear optimal velocity function
    #optimal velocity function looks like triangular fundamental diagram in speed-headway 
    
    #p[2] is sensitivity parameter, p[0] is jam headway, p[1] is the slope of the triangular part
    outdx = vehdx
    outddx = p[2]*(p[1]*(leadx-leadlen-vehx-p[0])-vehdx)
    
    
    
    #can add bounds on acceleration 
    
    if outddx > 4*3.3: #if acceleration more than 4 m/s/s
        outddx= 4*3.3
    elif outddx <-6*3.3: #if deceleration more than 6 m/s/s
        outddx = -6*3.3
     
    #velocity bounds
    if vehdx+dt*outddx < 0:
        outddx = -vehdx/dt
    if vehdx+dt*outddx > p[3]: #bounded velocity by p[3]
        outddx = (p[3]-vehdx)/dt
        
    
    return outdx, outddx

# for IDM, x = sqrt( (c3+c2*v)**2/ (1 - (v/c1)**4) )
def eql(model, v, p, length, tol=1e-4): 
    #finds equilibrium headway for a given speed and parameters value for second order model 
    #there should only be a single root 
    def wrapperfun(x):
        dx, ddx = model(0,v,x,v,p,length)
        return ddx
    
    guess = 10
    headway = 0
    try:
        headway = sc.newton(wrapperfun, x0=guess,maxiter = 50)
    except RuntimeError: 
        pass
    counter = 0
    
    while abs(wrapperfun(headway)) > tol and counter < 20: 
        guess = guess + 10
        try:
            headway = sc.newton(wrapperfun, x0=guess,maxiter = 50)
        except RuntimeError: 
            pass
        counter = counter + 1
        
        testout = model(0,v,headway,v,p,length)
        if testout[1] ==0:
            return headway 
        
    return headway



def daganzo(vehx, vehdx, leadx, leaddx, p,leadlen, *args,dt=.1):
    
    outdx = (leadx-leadlen-vehx-p[1])/p[0]
    
    if outdx > p[2]:
        outdx = p[2]
    
    return outdx