
"""
@author: rlk268@cornell.edu
houses the main code for running simulations 

TO DO //
    test and debug simulate_cir and simulate_step. 

    implementing boundary conditions
    
    getting networks working
    
    add some control models and write extended abstract
    
    adding lane changing and multi lane 
    
    how to do the adjoint calculation
    
    
    some tool for creating custom models out of modular parts, which should be part of models subpackage. 
    need to think about how to use modelinfo to implement all the parts of models that we need, and how
    to automatically get the gradient as easily as possible 
    
"""

def simulate_cir(curstate, auxinfo, modelinfo, L, timesteps, dt) :
    """
    simulates vehicles on a circular test track
    
    inputs -
    curstate - dict, gives initial state
    auxinfo - dict, initialized auxinfo, see simulate step
    modelinfo - dict, any things which may apply, see simulate step
    L - float, length of road
    timesteps - int, number of timesteps to simulate
    dt - float, length of timestep
    
    outputs - 
    sim - all simulated states 
    auxinfo - updated auxinfo
    modelinfo - updated modelinfo, first entry is L, second entry is whether we are in wrap-around state. 
    The wraparound state happens when our leader wraps around the road, but we haven't yet. 
    Thus the headway calculation needs to have L added to it. 
    """
    #initialize
    sim = {i:[curstate[i]] for i in curstate.keys()}
    
    
    for j in timesteps: 
        #update states
        nextstate, auxinfo, modelinfo = simulate_step(curstate,auxinfo,modelinfo,dt)
        
        #check for wraparound and if we need to update any special states for circular 
        #modelinfo[ID][1] is True if leader has wrapped around but follower has not 
        for i in nextstate.keys():
            if modelinfo[i][1]: #if in activated 
                if nextstate[i][0] > L: #follower wraps around so can reset 
                    nextstate[i][0] = nextstate[i][0] - L
                    modelinfo[i][1] = 0
            else: 
                if nextstate[auxinfo[i][1]][0] > L: #if leader wraps around
                    if nextstate[i][0] <= L:
                        modelinfo[i][1] = 1 #active wrap around state for i 
        
        #update iteration
        curstate = nextstate
        for i in curstate.keys(): 
            sim[i].append(curstate[i])
            
    return sim,curstate, auxinfo,modelinfo

def simulate_net():
    """
    simulate on a network
    """
    pass

def simulate_step(curstate, auxinfo, modelinfo, dt): 
    """
    does a step of the simulation
    
    inputs - 
    curstate - current state of the simulation, dictionary where each value is the current state of each vehicle 
    states are a length n list where n is the order of the model. 
    So for 2nd order model the state of a vehicle is a list of position and speed
    
    
    auxinfo - dictionary where keys are IDs, the values are length (0), curleader (1), init entry time (2), 
    parameters (3), model (4), lane (5), road (6), past lead info (7); this is information that we will always have
    
    modelinfo - dictionary where keys are IDs, the values are special extra parts which 
    are only needed for certain models. The values in here are specific to the model we are using 
    
    dt - timestep
    
    outputs - 
    nextstate - current state in the next timestep
    
    auxinfo - auxinfo and modelinfo may be changed during the timestep
    modelinfo - 
    
    """
    nextstate = {}
    for i in curstate.keys(): 
        #get necessary info for update
        leadid = auxinfo[i][1]
        lead = curstate[leadid]
        leadlen = auxinfo[leadid][0]
        
        #call model for vehicle
        out = auxinfo[i][4](curstate[i],lead,auxinfo[i][3],leadlen, modelinfo[i], dt=dt)
        
        #update position in nextstate for vehicle
        nextstate[i] = [curstate[i][0] + dt* out[0], curstate[i][1] + dt*out[1]]
        
        
    return nextstate, auxinfo, modelinfo

def eq_circular(p, length, model, eqlfun, n, L = None, v = None, perturb = 1e-2):
    #given circular road with length L with n vehicles which follow model model with parameters p, 
    #solves for the equilibrium solution and initializes vehicles in this circular road in the eql solution, 
    #with the perturbation perturb applied to one of the vehicles. 
    #you can eithe initialize L, in which case it will solve for v, or you can 
    #initialize v, and it will solve for the L. 
    #inputs - 
    #p- model parameters (scalar)
    #length - length of vehicles (scalar)
    #model - function for the model 
    #n - number of vehicles
    #l = None - length of circular test track
    #v = None - eq'l speed used 
    #perturb = 1e-2 - how much to perturb from the eql solution
    
    #outputs - 
    #curstate - state of the eql solution with perturbation
    #auxinfo - initialized auxiliary info for simulation
    #modelinfo - initialized model info for simulation 
    
    #first we need to solve for the equilibrium solution which forms the basis for the IC. 
    if L == None and v == None: 
        print('you need to specify either L or v to create the equilibrium solution')
        return
    elif L == None: 
        s = eqlfun(p,None,v,find='s')
        L = s*n
    elif v == None: 
        s = L / n 
        v = eqlfun(p,s,None,find='v')
        
    #initialize based on equilibrium
    initstate = {i: [s*i,v] for i in range(n)}
    initstate[0][0] = initstate[0][0] + perturb #apply perturbation
    initstate[0][1] = initstate[0][1] + perturb
    
    #create auxinfo
    auxinfo = {0: [length, n-1, 0, p, model, 1, 1, []]}
    for i in range(n-1):
        auxinfo[i+1] = [length, i, 0, p, model, 1, 1, []]
        
    #create modelinfo
    modelinfo = {i: [L, 0] for i in range(n)}
    modelinfo[n-1][1] = 1 #last vehicle starts in wrap-around state. 
    
    return initstate, auxinfo, modelinfo, L