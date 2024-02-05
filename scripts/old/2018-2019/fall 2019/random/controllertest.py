
"""
@author: rlk268@cornell.edu
test a controller 
was just wondering if you can get overshoot with a first order controller. I don't think you can, I think you have to use a second order

"""
import matplotlib.pyplot as plt

def feedback(p, vcmd, v):
    return p[0]*(vcmd-v) + p[1]*(vcmd - v)**2

def feedback2(p, vcmd, v):
    return p[0]*(vcmd-v) + p[1]

def feedback3(p,vcmd, v, a):
    return p[0]*(vcmd -v) + p[1]*a

def testcontroller():
    #define parameters
    p = [2, 0]
    vcmd = 30
    v  = 15
    dt = .1
    #implement controller
    vlist = [v]
    for i in range(100):
        vcur = vlist[-1]
        a = feedback2(p,vcmd,vcur)
        vnew = vcur +dt*a
        vlist.append(vnew)
    
    #plot
    plt.figure()
    plt.plot(vlist)
    
def testcontroller2(): #example of second order controller is like harmonic oscillator with damping
    #define parameters
    p = [.4, -.5]
    vcmd = 30
    v  = 15
    a = 0
    dt = .1
    #implement controller
    vlist = [v]
    alist = [a]
    for i in range(100):
        vcur = vlist[-1]
        acur = alist[-1]
        jerk = feedback3(p,vcmd,vcur,acur)
        vnew = vcur +dt*acur
        anew = acur + dt*jerk
        vlist.append(vnew)
        alist.append(anew)
    
    #plot
    plt.figure()
    plt.plot(vlist)
    

#testcontroller2()
testcontroller()
    
        
        