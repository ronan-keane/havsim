
"""
@author: rlk268@cornell.edu
for benchmarking/debugging purposes
"""
import numpy as np 

class vehicle: 
    def __init__(self, p, lcp): 
        self.pos = 1
        self.speed = 1 
        self.p = p
        self.lcp = lcp
        
        self.posmem = [1]
        self.lcmem = []
        
    def call_cf(self, pos, speed):
        p = self.p
        out = abs(pos)**.5+abs(speed)**.5+p[0]*speed+p[1]*pos+p[2]-abs(p[4]*(pos*speed))*np.sign(speed)
        if out > p[3]: 
            out = p[3]
        elif out < p[5]:
            out = p[5]
        return out
    
    def call_lc(self):
        if np.random.rand() < .5: 
            return False
        p = self.lcp
        lout = rout = -1e10
        if self.lfol is not None: 
            pos = p[0]*self.pos + p[1]*self.lfol.pos + p[2]
            speed = p[3]*self.speed + p[4]*self.lfol.speed + p[5]
            lout = self.call_cf(pos, speed)
        if self.rfol is not None: 
            pos = p[0]*self.pos + p[6]*self.rfol.pos + p[7]
            speed = p[3]*self.speed + p[8]*self.rfol.speed + p[9]
            rout = self.call_cf(pos, speed)
        out = max(lout, rout)
        if out < p[10]: 
            return True
        elif out < p[11]:
            return False
        else:
            return True
        
    def update(self):
        self.pos += self.speed
        self.speed += self.acc
        
        self.posmem.append(self.pos)
        self.lcmem.append(self.lc)
        
        

def simulation(vehicles, timesteps):
    for j in range(timesteps):
        #the following code is able to be parralelized 
        for veh in vehicles: 
            veh.acc = veh.call_cf(veh.pos, veh.speed)
        for veh in vehicles: 
            veh.lc = veh.call_lc()
        for veh in vehicles: 
            veh.update()
        
        #this part of the code is only able to be parralelized in this toy example
        for veh in vehicles: 
            if np.random.rand() < .01: 
                for i in veh.posmem[-20:]:
                    if i>0: 
                        veh.pos += .01
    return

#code to initialize vehicles and parameters - performance of this is not important 
def generate_parameters():
    p = [-.5-np.random.rand(), -.5-np.random.rand(), -1.5-np.random.rand(),
         .25+.1*np.random.rand(), .6-.5*np.random.rand(), -.35+.1*np.random.rand()]
    lcp = [.51, .49, .1, .512, .499, .11,.48, .487, .05, .06, .2, -.2]
    
    return p, lcp
    
vehicles = []
for i in range(30):
    p, lcp = generate_parameters()
    lveh = vehicle(p, lcp)
    p, lcp = generate_parameters()
    veh = vehicle(p, lcp)
    p, lcp = generate_parameters()
    rveh = vehicle(p, lcp)
    
    lveh.lfol, lveh.rfol = None,veh
    veh.lfol, veh.rfol = lveh, rveh
    rveh.lfol, rveh.rfol = veh, None
    
    vehicles.extend([lveh, veh, rveh])

#benchmark simulation function 
#simulation(vehicles, 1000)
    
    
#try to debug tensorflow
class debugenv:
    def __init__(self):
        vehicles = []
        for i in range(30):
            p, lcp = generate_parameters()
            lveh = vehicle(p, lcp)
            p, lcp = generate_parameters()
            veh = vehicle(p, lcp)
            p, lcp = generate_parameters()
            rveh = vehicle(p, lcp)
            
            lveh.lfol, lveh.rfol = None,veh
            veh.lfol, veh.rfol = lveh, rveh
            rveh.lfol, rveh.rfol = veh, None
            
            vehicles.extend([lveh, veh, rveh])
        self.vehicles = vehicles 
        self.statememdim = 5
        
    def step(self, *args):
        simulation(self.vehicles, 10)
        return np.random.rand(1,5), np.random.rand(), False
    
    def reset(self, *args):
        self.totloss = 0
        return np.random.rand(1,5)
        