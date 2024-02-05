
"""
@author: rlk268@cornell.edu
"""

import math 
import numpy as np 
from havsim.simulation.models import IDM_b3_eql
from havsim.simulation.modelsold import IDM_b3

class vehicle: 
    #class for vehicles which are described by a second order ODE 
    dim = 2
    def __init__(self, IC, t, length, leader, model, p, dt):
        self.len = length
        self.leader = [leader]
        self.model = model
        self.p = p
        self.t = t
        self.dt = dt
        
        self.x = [IC[0]]
        self.dx = [IC[1]]
        self.ddx = []
        
        self.newx = None
        self.newdx = None
        self.newddx = None
        
    def eulerstep(self):
        leadx = self.leader[-1].x[-1]
        leaddx = self.leader[-1].dx[-1]
        leadlen = self.leader[-1].len
        
        self.newdx, self.newddx = self.model(self.x[-1],self.dx[-1],leadx,leaddx,self.p,leadlen,dt = self.dt)
        self.newx = self.x[-1] + self.dt*self.newdx
        self.newdx = self.dx[-1] + self.dt*self.newddx
        
    def eulerupdate(self):
        self.x.append(self.newx)
        self.dx.append(self.newdx)
        self.ddx.append(self.newddx)
        
#    def defineleader(self, velocity):
#        self.dx = velocity
#        for i in range(len(velocity)):
#            curx = self.x[-1]
#            self.x.append(curx + self.dt*velocity[i])
            
class leadvehicle: 
    def __init__(self,ICx,t,length,velocity, dt):
        self.x = [ICx]
        self.t = t
        self.dt = dt
        self.len = length
        self.velocity = velocity
        
        self.dx = [velocity[0]]
        self.ddx = []
        self.cur = 0
        
        self.newdx = None
        
    def eulerstep(self):
        self.cur = self.cur + 1
        pass
        
    def eulerupdate(self):
        newx = self.x[-1] + self.dt*self.dx[-1]
        newdx = self.velocity[self.cur]
        self.x.append(newx)
        self.dx.append(newdx)
        
        
def simulate(universe,steps):
    for i in range(steps):
        for j in universe: 
            j.eulerstep()
        for j in universe: 
            j.eulerupdate()
    return


def egobjwrap(p,phuman,model,headway,prate,simlen,N,velocitylead,initspeed, dt = .1, reps = 5):
    obj = 0 
    for i in range(reps):
        obj += egobj(p,phuman,model,headway,prate,simlen,N,velocitylead,initspeed, dt)
    return obj / reps

def egobj(p,phuman,model,headway,prate,simlen,N,velocitylead,initspeed, dt = .1):
    #given an initial speed disturbance velocitylead, and all vehicles initially at max speed initspeed, 
    #randomally assign some vehicles as human and some as autonomous, with each population having 
    #a unique parameter set for a given model. 
    length = 5 #length doesn't really matter so just set it to 5 arbitrarily 
    safehd = 5  #safe headway is set to 5 somewaht arbitrarily 
    #initialization
    v1 = leadvehicle(0,0,length,velocitylead,dt)
    universe = [v1]
    human = np.ones((N,))
    
    if type(prate) == float: #randomally assign some vehicles 
        nav = math.floor(prate*N)
        count = 0
        #randomally assign some vehicles as autonomous 
        while count < nav: #add autonomous vehicles 
            newind = math.floor(N*np.random.rand())
            if human[newind] == 1: 
                human[newind] = 0
                count += 1
    else: 
        for i in prate: 
            human[i] = 0
        
     #initialize simulation 
    prev = 0
    for i in range(N):
        prev = prev- headway
        if human[i]==1:
            newveh = vehicle([prev,initspeed],0,length,universe[-1],model,phuman,dt)
        else: 
            newveh = vehicle([prev,initspeed],0,length,universe[-1],model,p,dt)
        universe.append(newveh)
    
    #run simulation 
    simulate(universe,simlen-1)
    
    #compute objective
    obj = 0
    for i in range(N+1):
        curdx = np.asarray(universe[i].dx) - initspeed
        obj += -sum(curdx*dt)
        
    #need to add some sort of safety constraint otherwise the solution is just going to be for AV to drive at max speed no matter what 
#    for i in range(N):
#        if human[i] ==1:
#            continue
#        else: 
#            hd = np.asarray(universe[i].x)-np.asarray(universe[i].x)-length #headway for av
#            reg = np.exp((-hd+safehd)*2) #this is a penalization term to ensure a somewhat sensible answer 
#        obj += sum(reg)
        
    return obj 
    

def eguni(p,phuman,model,headway,prate,simlen,N,velocitylead,initspeed, dt = .1):
    #given an initial speed disturbance velocitylead, and all vehicles initially at max speed initspeed, 
    #randomally assign some vehicles as human and some as autonomous, with each population having 
    #a unique parameter set for a given model. 
    length = 5 #length doesn't really matter so just set it to 5 arbitrarily 
    safehd = 5  #safe headway is set to 5 somewaht arbitrarily 
    #initialization
    v1 = leadvehicle(0,0,length,velocitylead,dt)
    universe = [v1]
    human = np.ones((N,))
    
    if type(prate) == float: #randomally assign some vehicles 
        nav = math.floor(prate*N)
        count = 0
        #randomally assign some vehicles as autonomous 
        while count < nav: #add autonomous vehicles 
            newind = math.floor(N*np.random.rand())
            if human[newind] == 1: 
                human[newind] = 0
                count += 1
    else: 
        for i in prate: 
            human[i] = 0
        
     #initialize simulation 
    prev = 0
    for i in range(N):
        prev = prev- headway
        if human[i]==1:
            newveh = vehicle([prev,initspeed],0,length,universe[-1],model,phuman,dt)
        else: 
            newveh = vehicle([prev,initspeed],0,length,universe[-1],model,p,dt)
        universe.append(newveh)
    
    #run simulation 
    simulate(universe,simlen-1)
    
    #compute objective
    obj = 0
    for i in range(N+1):
        curdx = np.asarray(universe[i].dx) - initspeed
        obj += -sum(curdx*dt)
        
    #need to add some sort of safety constraint otherwise the solution is just going to be for AV to drive at max speed no matter what 
#    for i in range(N):
#        if human[i] ==1:
#            continue
#        else: 
#            hd = np.asarray(universe[i].x)-np.asarray(universe[i].x)-length #headway for av
#            reg = np.exp((-hd+safehd)*2) #this is a penalization term to ensure a somewhat sensible answer 
#        obj += sum(reg)
        
    return universe, obj

def egaexam(p):
    simlen = 1000 #33000
    N = 50 #800
    speedoffset = -10
    length = 5
    velocity = [30 for i in range(simlen)]
    velocity[0:200] = [1/10*(.1*(i)-10)**2+20 for i in range(200)] #polynomial disturbance 
    
    velocity = np.asarray(velocity) + speedoffset
    
    t = np.arange(0,100,.1)
    
    v1 = leadvehicle(0,0,length,velocity,.1)
    universe = [v1]
    
    p2 = [33.33,1.1,2,.9,1.5]
    
    headway = IDM_b3_eql(p, None, 30+speedoffset, find = 's', maxs = 1e4) + length
    headway2 = IDM_b3_eql(p2, None, 30+speedoffset, find = 's', maxs = 1e4) + length
    prev = 0
    indlist = np.arange(0, N,10)
    for i in range(N):
        if i in indlist:
            prev = prev - headway
            newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],IDM_b3,p,.1)
            universe.append(newveh)
        else:
            prev = prev - headway2
            newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],IDM_b3,p2,.1)
            universe.append(newveh)
        
    simulate(universe,simlen-1)
    
    obj = 0
    for i in range(N+1):
        avgdx = np.mean(universe[i].dx)
        curobj = np.asarray(universe[i].dx) - avgdx
        obj += np.sum(np.square(curobj))/N - avgdx**2*simlen/N
        
    return obj 

def egaexamuni(p):
    simlen = 4000 #33000
    N = 200 #800
    speedoffset = -10
    length = 5
    velocity = [30 for i in range(simlen)]
    velocity[0:200] = [1/10*(.1*(i)-10)**2+20 for i in range(200)] #polynomial disturbance 
    
    velocity = np.asarray(velocity) + speedoffset
    
    t = np.arange(0,100,.1)
    
    v1 = leadvehicle(0,0,length,velocity,.1)
    universe = [v1]
    
    p2 = [33.33,1.1,2,.9,1.5]
    
    headway = IDM_b3_eql(p, None, 30+speedoffset, find = 's', maxs = 1e4) + length
    headway2 = IDM_b3_eql(p2, None, 30+speedoffset, find = 's', maxs = 1e4) + length
    prev = 0
    indlist = np.arange(0, N,10)
    for i in range(N):
        if i in indlist:
            prev = prev - headway
            newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],IDM_b3,p,.1)
            universe.append(newveh)
        else:
            prev = prev - headway2
            newveh = vehicle([prev,30+speedoffset],0,length,universe[-1],IDM_b3,p2,.1)
            universe.append(newveh)
        
    simulate(universe,simlen-1)
    
    obj = 0
    for i in range(N+1):
        avgdx = np.mean(universe[i].dx)
        curobj = np.asarray(universe[i].dx) - avgdx
        obj += np.sum(np.square(curobj))/N - avgdx**2*simlen/N
        
    return obj, universe 

def eulerstep(sim, t, model, N, dim, dt):
    #sim - where the output is stored 
    #t = current timestep 
    #model - derivative of the current time according to model
    #N - number of vehicles 
    #dim - order of the model (model gives acceleration = 2nd order DE -> dim = 2)
    #dt - numerical timestep
    
    cur = sim[:,:,t]
    for i in range(N):
        old = cur[i,:dim]
        new = model(old)
        sim[i,:dim,t+1] = old+dt*new
        sim[i,dim,t+1] = new[-1]
    return 

def euler(IC,steps,model,dim = 2, dt=.1):
    N = np.shape(IC)[0] #number of vehicles 
    sim = np.zeros(N,3,steps+1)
    
    sim[:,:,0] = IC
    
    for i in range(steps):
        eulerstep(sim,i,model,N,dim,dt)

    return





