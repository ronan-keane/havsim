
"""
Algorithms for forming platoons and sorting vehicles.
"""
import numpy as np
import havsim.helper as helper
import math
import networkx as nx
import copy
import matplotlib.pyplot as plt


#TODO fix code style and documentation

def makefollowerchain(leadID, dataset, n=1, picklane = 0 ):
    # you probably want to either use makeplatoonlist or makeplatoonlist_s/sortveh instead of this.
	#	given a lead vehicle, makefollowerchain forms a platoon that we can calibrate. The platoon is formed is such a way that there is no lane changing,
	#Mainly meant for testing and plotting, you probably want to use makeplatoonlist unless for some reason you don't want to have any lane changing in your calibration

	#input:
	#    leadID: vehicle ID of leader. this is the first vehicle in the platoon, so the next vehicle will be the follower of this vehicle, and then the vehicle after will be the follower of
	#    that vehicle, etc.
	#
	#    dataset: source of data. it needs to have all the entries that are in dataind for each observation (as well as the vehicle ID and frame ID entries)
	#
	#    n = 1: this is how many following vehicles will be in the followerchain.
	#
	#    picklane = 0 : laneID for leader (optional). If you give default value of 0, we just give the lane that has the most observations for the leader as the lane to use
	#
	#output:
	#    meas: all measurements with vehID as key, values as dataset[dataset==meas.val()] i.e. same format as dataset
	#    This is only for the vehicles in followerchain, and not every single vehicle in the dataset
	#
	#    followerchain: dictionary with key as vehicle ID
	#    value is array containing information about the platoon and calibration problem
	#        0 - t_nstar (first recorded time)
	#        1 - t_n (first simulated time)
	#        2 - T_nm1 (last simulated time) (read as T_{n minus 1})
	#        3 - T_n (last recorded time)
	#        4 - leader ID (in an array so that it is consistent with platooninfo)
	#        5 - position IC
	#        6 - speed IC
	#        7 - follower ID (in an array)

    #this is the newer version of makefollowerchain which always gets all vehicles in the lane, even ones which are only simulated a short time.
    #however, there is still one thing in this unclear to me which is how it is going to handle the case of a vehicle which is in the lane then merges out of the lane
    #and then back in at a later time. I think in that case, it only selects the longest continuous trajectory and ignores the other part.
    #so this edge case you need to be careful with.


    meas = {} #this is where the measurements (from data) go
    followerchain = {} #helpful information about the platoon and the followers


    lead = dataset[dataset[:,0]==leadID] #get all data for lead vehicle
    lanes, lanes_count = np.unique(lead[:,7], False, False, True) #get all lanes lead vehicle has been in. you can specify a lane manually, default to most traveled lane
    if picklane == 0: #default to most traveled lane if no lane given
        picklane = lanes[np.argmin(-lanes_count)]
    lead = lead[lead[:,7]==picklane] #get all lead trajectory data for specific lane

    lead, _= helper.checksequential(lead) #we only want sequential data for the lead trajectory i.e. don't want leader to change lanes and then change back to first lane
    leadID = lead[0,0]
    #now we have sequential measurments for the leadID in the lane picklane. we want to get followers and form a platoon of size n
    followerchain[leadID] = [int(lead[0,1]), int(lead[0,1]), int(lead[0,1]), int(lead[-1,1]), -1, -1,-1, []] #we give leader a followerchain entry as well. unused entries are set equal to -1
   #modification
    backlog = []
    leadlist = []
    i = 1

    while i < n:
#        print(i)
#        print(leadlist)
        leadlist.append(leadID)
        meas[leadID] = lead #add data of lead vehicle
        followers, followers_count = np.unique(lead[:,5], False, False, True) #get all followers for specific leader in specific lane
        #modification
#        if len(backlog) == 0:
#            pickfollower = followers[np.argmin(-followers_count)] #next follower is whichever follower has most observations #before it was just this
#        else:
#            pickfollower = backlog.pop()
        for j in followers:
            if j ==0:#don't add 0
                continue
            elif j in leadlist or j in backlog:
                continue
            backlog.append(j) #add everything to the backlog

        pickfollower = backlog[0] #take out the first thing
        backlog.remove(pickfollower)


        fullfollower = dataset[np.logical_and(dataset[:,0]==pickfollower, dataset[:,7]==picklane)] #get all measurements for follower in current lane
        #need to change this line
#        curfollower = fullfollower[fullfollower[:,4]==leadID] #get all measurements for follower with specific leader
        curfollower = []
        for j in range(len(fullfollower)):
            if fullfollower[j,4] in leadlist:
                curfollower = np.append(curfollower,fullfollower[j,:])
        curfollower = np.reshape(curfollower,(int(len(curfollower)/9),9))


        followerchain[leadID][-1].append(pickfollower) #append the new follower to the entry of its leader so we can get the platoon order if needed
        #modification

        #if curfollower is empty it means we couldn't find a follower
        testshape = curfollower.shape

        if testshape[0] ==0:
            print('warning: empty follower')
#            return meas, followerchain
            continue


        #need to check all measurements are sequential
        curfollower, _ = helper.checksequential(curfollower) #curfollower is where we have the lead measurements for follower
        extrafollower = fullfollower[fullfollower[:,1]>curfollower[-1,1]] #extrafollower is where all the "extra" measurements for follower go - no leader here
        extrafollower, _ = helper.checksequential(extrafollower, 1, True)

        testshape1 = extrafollower.shape #this prevents index error which occurs when extrafollower is empty
        if testshape1[0]==0:
            T_n = curfollower[-1,1]
        else:
            T_n = extrafollower[-1,1]

        if curfollower[-1,1]< lead[-1,1]:
            print('warning: follower trajectory not available for full lead trajectory') #you will get this print out when you use makefollowerchain on data with lane changing.
            #the print out doesn't mean anything has gone wrong, it is just to let you know that for certain vehicles in the platoon, those vehicles can't be simulated
            #for as long as a time as they should, essentially because they, or their leaders, change lanes

        followerchain[pickfollower]= [int(fullfollower[0,1]), int(curfollower[0,1]), int(curfollower[-1,1]), int(T_n), [leadID], curfollower[0,2], curfollower[0,3], []]

        #update iteration
#        lead = np.append(curfollower,extrafollower, axis = 0)
        lead = fullfollower #this is what lead needs to be to be consistent with the conventions we are using in makeplatooninfo and makeplatoon
        leadID = pickfollower
        i += 1

    meas[leadID] = lead #add data of last vehicle
    return meas, followerchain








def makeplatoon(platooninfo, leaders, simcount, curlead, totfollist, vehicles_added, meas=[],
                   cycle_num=5e5, n=10, X = math.inf, Y = 0, cirdep = False,maxn= False, previousPlatoon=[]):
#	input:
#    meas, (see function makeplatooninfo)
#
#    platooninfo, (see function makeplatooninfo) - note that makeplatoon will modify the last argument of platooninfo; which is only used for makeplatoon
#
#    leaders, (see function makeplatooninfo) - note that makeplatoon will remove vehicles from leaders after they are no longer needed
#
#    simcount (see function makeplatooninfo), - note that makeplatoon modifies this value; it keeps track of how many vehicles still have followers yet to be simulated
#
#    totfollist - output from makeplatooninfo. updated during execution. this is all followers of every vehicle in leaders

    #cycle_num = 50000 - if cirdep = True, then this controls how many cycles we resolve at a time.
    #Larger is better, lower is faster.
    #If cirdep is False, this controls how many loops we will consider adding.
    #again, larger is better, lower is faster.
    #If infinity (math.inf) is taking too long, you can try something like 1e5-1e7
#
#    n = 10: n controls how big the maximum platoon size is. n is the number of following vehicles (i.e. simulated vehicles)

    #X = math.inf - every X vehicles, we attempt to resolve a circular dependency

    #Y = 0 - when resolving circular dependencies early, this is the maximum depth of the cycle allowed.
    #the depth of a cycle is defined as the minimum depth of all vehicles involved in the cycle.
    #the depth of a vehicle is defined as 0 for vehicles who have a vehicle in leaders as a leader,
    #1 for vehicles which have vehicles with depth 0 as leaders, etc.
    #actually Y doesn't have any effect so you can just go ahead and ignore it.

    #cirdep = False - if False, no circular dependencies will be in the output.
    #this means that some platoons may potentially be very large.
    #Alternatively, you can pick cirdep = True, and there will be no very large platoons,
    #but there may be circular dependencies. cirdep= False is good.

    #maxn = False - Only gets used if cirdep is True. If maxn is True, then all platoons
    #will be of size n except for the last platoon. If maxn is False, then when we resolve
    #loops platoons may not be of size n (they can be bigger or smaller)

    #previousPlatoon - when forming platoons, we remember the previous platoon to use a tie breaking rule when adding the first vehicle
#
#
#output:
#    platooninfo, which is updated as we simulate more vehicles
#
#    leaders, which are vehicles either already simulated or vehicles that were never simulated. everything in leaders is what we build platoons off of!
#    NOTE THAT LEADERS ARE NOT ALL LEADERS. it is only a list of all vehicles we are currently building stuff off of. we remove vehicles from leaders
#    after they have no more followers we can simulate
#
#    simcount, how many more vehicles have followers that can be simulated. when simcount = 0 all vehicles in the data have been simulated.
#
#    totfollist - updated total list of followers. updated because any simulted vehicle is added as a new leader
#
#    platoons - the list of vehicles

    #################

    platoonsout = []
    platoons = [] #output
    curn = 0 #current n value
#    vehicles_added = 0

#    if previousPlatoon == None:


    #update leaders, platooninfo, totfollist, simcount so we can add all vehicles in curfix
    def addCurfix(curfix):
        nonlocal simcount, curn, platoons, totfollist, platoonsout, leaders
        for i in curfix:
            curveh = i
            chklead = platooninfo[curveh][4]

            leaders.insert(0, curveh)  # curveh will be simulated now so we can insert it into leaders
            if curveh in totfollist:
                totfollist.remove(curveh)

            for j in chklead:
                platooninfo[j][-1].remove(curveh)
                if len(platooninfo[j][-1]) < 1:  # remove a leader if all followers are gone
                    simcount += -1
                    if j in leaders:
                        leaders.remove(j)
            totfollist.extend(platooninfo[curveh][-1])
            totfollist = list(set(totfollist))
        # add all vehicles in curfix

        if curn + len(curfix) > n:
            if maxn:  # keep platoons of size n in this case if you want
                m = len(curfix)
                platoons.extend(curfix[:n - curn])  # finish off current platoon
                platoonsout.append(platoons)

                firstr = n - curn
                m = m - firstr
                count = 0
                for i in range(m // n):
                    platoonsout.append(curfix[i * n + firstr:(i + 1) * n + firstr])  # add full platoons
                    count = i + 1

                platoons = curfix[count * n + firstr:]  # rest of vehicles in curfix
                curn = len(platoons)

            else:  # here we need to keep all of curfix together
                if curn > 0:  # don't append empty platoons
                    platoonsout.append(platoons)
                platoons = curfix
                curn = len(curfix)
        else:
            platoons.extend(curfix)
            curn = curn + len(curfix)

    #extra heuristic -
    while curn < n and simcount > 0: #loop which will be exited when the platoon is of size desired n or when no more vehicles can be added
        #get scores based on chain metric - we will add vehicles using greedy algorithm
        bestVeh = None
        bestScore = None

        #modification 21
        if vehicles_added >= X:
            vehicles_added = vehicles_added % X
            curfix = addcyclesearly(totfollist, leaders, platooninfo, cycle_num, Y)
#            print(curfix)
            addCurfix(curfix)

        for i in totfollist: #apply greedy algorithm to select next vehicle
            chklead = platooninfo[i][4] #these are all the leaders needed to simulate vehicle i
            if all(j in leaders for j in chklead): #will be true if curlead contains all vehicles in chklead; in that case vehicle i can be simulated
                T = set(range(platooninfo[i][1], platooninfo[i][2] + 1))
                if not platoons: #
                    score = helper.c_metric(i, previousPlatoon, T, platooninfo, meas=meas) + \
                    helper.c_metric(i,previousPlatoon,T,platooninfo,meas=meas,metrictype='fol')
                else:
                    score = helper.c_metric(i, platoons, T, platooninfo, meas=meas) + \
                    helper.c_metric(i,platoons,T,platooninfo,meas=meas,metrictype='fol')
                if bestScore == None:
                    bestScore = score
                    bestVeh = [i]
                if score == bestScore:
                    bestVeh.append(i)
                if score > bestScore:
                    bestScore = score
                    bestVeh = [i]


        if bestVeh != None: #add the best vehicle; if = None, no vehicles can be added; move to loop resolution
            if len(bestVeh) > 1: #apply tie breaking rule, this might occur if all scores are 0.
                besttime = -1
                for i in bestVeh:
                    curtime = platooninfo[i][2] - platooninfo[i][1]
                    if curtime > besttime:
                        best = i
                bestVeh = best
            elif len(bestVeh)==1:
                bestVeh = bestVeh[0]
            curn += 1  #keep track of platoon size
            platoons.append(bestVeh)
            leaders.insert(0,bestVeh)  # append newly added follower to the leader list as well; we have simulated vehicle i, and can now treat it as a leader
            totfollist.remove(bestVeh)
            chklead = platooninfo[bestVeh][4]
            #
            totfollist.extend(platooninfo[bestVeh][-1])
            totfollist = list(set(totfollist))
            for j in chklead:
                platooninfo[j][-1].remove(bestVeh)  # remove from followers
                if len(platooninfo[j][-1]) < 1:  # if a leader has no more followers
                    simcount += -1  # adjust simcount. if simcount reaches 0 our job is finished and we can return what we have, even if curn is not equal to n
                    leaders.remove(j)  # remove it from the list of leaders
            vehicles_added += 1
        else: #resolve circular dependency

            if cirdep:
                curfix = breakcycles(totfollist, leaders, platooninfo, cycle_num)
            else:
                curfix = addcycles2(totfollist,leaders,platooninfo,cycle_num)
#                print([totfollist, leaders, curfix])
            addCurfix(curfix)

    platoonsout.append(platoons)
    return platooninfo, leaders, simcount, curlead, totfollist,vehicles_added, platoonsout


def makedepgraph(totfollist,leaders,platooninfo, Y):
    #makes dependency graph and corresponding depths for followers totfollist
    #with leaders leaders. Y is maximum depth allowed
    G = nx.DiGraph()
    depth = {j: 0 for j in totfollist}
    curdepth = set(totfollist)
    alreadyadded = set(totfollist)
    dcount = 1 #depth count
    while len(curdepth) > 0 and dcount <= Y:
        nextdepth = set()
        for j in curdepth:
            for i in platooninfo[j][4]:
                if i not in leaders:
                    G.add_edge(j,i)
                    try:
                        if dcount < depth[i]: #can update depth only if its less
                            depth[i] = dcount
                    except: #except if depth[i] doesn't exist; can initialize
                        depth[i] = dcount
                    if i not in alreadyadded and i not in curdepth:
                        nextdepth.add(i)
        # modification
        dcount+=1
        alreadyadded = alreadyadded.union(curdepth)
        curdepth = nextdepth

    return G, depth

def breakcycles(totfollist, leaders, platooninfo, cycle_num):
    #need to do research to determine whether or not the large platoons are a problem,
    #but if they are then you can use breakcycles to ensure that the platoonsize is always
    #below some threshold. In that case, you'd also want some sort of strategy for dealing
    #with the circular dependencies which will occur (e.g. calibrating both platoons until some convergeance is achieved)

    #an obvious way to improve this is as follows:
    #first, use the addcycles2 to obtain the cycle which would have been added.
    #then, find some collection of vehicles inside the cycle which minimizes some metric,
    #e.g. which  minimizes the circular dependency score. When the collection of vehicles is added,
    #then the rest of the cycle should be able to be added. Something like that.

    #its called break cycles because we take vehicle cycles and don't add them all at once
    #so it's like pretending the cycle isn't there. This will make output have circular dependency

    #iterate over all followers, try to find the graph with least amount of followers
    #we do this because we don't use the depth argument to keep track of which cycles are preferable,
    #so instead we will try to add the smaller cyclers first
    #if you don't do this you will get needlessly large cycles.
    Glen = math.inf
    Gedge= math.inf
    for i in totfollist:
        curG, depth = makedepgraph([i],leaders,platooninfo,math.inf)
        if len(curG.nodes()) < Glen or len(curG.edges())<Gedge:
            G = curG
            Glen = len(G.nodes())
            Gedge = len(G.edges())
    cyclebasis = nx.simple_cycles(G)

    universe = list(G.nodes()) #universe for set cover
    subsets = []
    count = 0
    while count < cycle_num:
        try:
            subsets.append(next(cyclebasis))
        except:
            break
        count += 1

    for i in range(len(subsets)):
        subsets[i] = set(subsets[i])
    #actually we want to solve a hitting set problem, but we do this with a set cover algorithm, so we have some extra conversion to do
    HSuni = list(range(len(subsets))) #read variable name as hitting set universe; universe for the hitting set HSuni[0] corresponds to subsets[0]
    HSsubsets = [] #this is the list of subsets for the hitting set
    for i in range(len(universe)): #each member of universe we need to replace with a set
        curveh = universe[i] #current member of universe
        cursubset = set() #initialize the set we will replace it with
        for j in range(len(subsets)): #
            if curveh in subsets[j]: # if i is in subsets[j] then we add the index to the current set for i
                cursubset.add(j)
        HSsubsets.append(cursubset)
    result = helper.greedy_set_cover(HSsubsets,HSuni) #solve the set cover problem which gives us the HSsubsets which cover HSun
    #now we take the output to the hitting set problem, and these vehicles get added.
    curfix = [universe[HSsubsets.index(i)] for i in result] #curfix will be all the vehicles in the result
    return curfix

#explanatino of addcycles2,3:
    #addcyclesearly makes dependency graph, finds cycles, and then sees what needs to be
    #added to add the cycles.
    #addcycles2 looks over each totfollist, makes its dependency graph, and then
    #just adds the smallest dependency grpah.
    #addcycles3 makes the dependency graph for all of totfollist, and then for each node of
    #that dependency graph, sees what all needs to be added to add the vehicle.

    #I am fairly certain that you can just always use 2, as 2 and 3 should output the same answer (?)
    #and 2 forms less dependency graphs.
    #for addition of cycles early, you want to check that the cycle actually exists (as opposed
    #to the normal addition of cycles, where we know it will exist), so you can use addcycles
    #note that addcycles2 is by far the fastest; in particular things can get slow if you use addcyclesearly alot
    #maybe possible to refactor addcyclesearly to be faster. That is only used for early addition of loops. Does it even help?
def addcyclesearly(totfollist, leaders, platooninfo, cycle_num, Y):
    #for adding cycles early; in this case we need to check for cycles
    G, depth = makedepgraph(totfollist,leaders,platooninfo,math.inf)
    cyclebasis = nx.simple_cycles(G)

    count = cycle_num
    bestdepth = math.inf
    bestsize = math.inf
    bestCurFix = nx.DiGraph()
    while count > 0:  # check first cycle_num cycles
        count -= 1
        try:
            cycle = next(cyclebasis)
        except:
            break
        #
        candidates = list(cycle)

        curfix, unused = makedepgraph(candidates,leaders,platooninfo,math.inf)
        curdepth = min([depth[i] for i in curfix.nodes()])
        cursize = len(curfix.nodes())
        if curdepth <= bestdepth and curdepth <=Y:
            if cursize < bestsize:
                bestCurFix = curfix
                bestdepth = curdepth
                bestsize = cursize

    return list(bestCurFix.nodes())

def addcycles2(totfollist, leaders, platooninfo, cycle_num):

    bestdepth = math.inf
    bestsize = math.inf
    for i in totfollist:
        curfix, unused = makedepgraph([i],leaders,platooninfo,math.inf)
        curdepth = 0
        cursize = len(curfix.nodes())
        if curdepth <= bestdepth:
            if cursize < bestsize:
                bestCurFix = curfix
                bestdepth = curdepth
                bestsize = cursize

    return list(bestCurFix.nodes())

def addcycles3(totfollist, leaders, platooninfo, cycle_num):
    #we add cycles all at once so there will not be circular dependencies in platoon
    G, depth = makedepgraph(totfollist,leaders,platooninfo,math.inf)
    bestdepth = math.inf
    bestsize = math.inf
    for i in G.nodes():
        curfix, unused = makedepgraph([i],leaders,platooninfo,math.inf)
        curdepth = 0
        cursize = len(curfix.nodes())
        if curdepth <= bestdepth:
            if cursize < bestsize:
                bestCurFix = curfix
                bestdepth = curdepth
                bestsize = cursize

    return list(bestCurFix.nodes())

#def resolveCycleEarly(totfollist, leaders, platooninfo, cycle_num, Y): #deprecated, use addcyclesearly
#    G, depth = makedepgraph(totfollist, leaders, platooninfo, Y)
#    if len(G.nodes()) == 0:
#        return []
#    bestdepth = math.inf
#    bestsize = math.inf
#    count = 0
#    for i in G.nodes():
#        if count>=cycle_num:
#            break
#        curfix, unused = makedepgraph([i], leaders, platooninfo, math.inf)
#        if len(curfix.nodes()) == 0:
#            continue
#        curdepth = min([depth[i] if i in depth.keys() else 0 for i in curfix.nodes()])
#        cursize = len(curfix.nodes())
#        if curdepth <= bestdepth:
#            if cursize < bestsize:
#                bestCurFix = curfix
#                bestdepth = curdepth
#                bestsize = cursize
#        count += 1
#    return list(bestCurFix.nodes())

def makeplatoonlist(data, n=1, form_platoons = True, extra_output = False,lane= None, vehs = None,cycle_num=5e5, X =  math.inf, Y = 0, cirdep = False, maxn = False):

    #this runs makeplatooninfo and makeplatoon on the data, returning the measurements (meas), information on each vehicle (platooninfo),
    #and the list of platoons to calibrate
    """
	this function is slow - need to profile and optimize
    """
	#inputs -
	# data - data in numpy format with correct indices
	# n = 1 - specified size of platoons to form
	# form_platoons = True - if False, will just return meas and platooninfo without forming platoons
	# extra_output = False - option to give extra output which is just useful for debugging purposes, making sure you are putting all vehicles into platoon

	# lane = None - If you give a float value to lane, will only look at vehicles in data which travel in lane
	# vehs = None - Can be passed as a list of vehicle IDs and the algorithm will calibrate starting from that first vehicle and stopping when it reaches the second vehicle.
	# lane and vehs are meant to be used together, i.e. lane = 2 vehs = [582,1146] you can form platoons only focusing on a specific portion of the data.
	#I'm not really sure how robust it is, or what will happen if you only give one or the other.

    #cycle_num, X, Y, cirdep, maxn - refer to makeplatoon for the options these keywords control

    #this also implements a useless vehicle heuristic, which is only designed to work if cirdep = False. if cirdep = True it may or may not work.

	#outputs -
	# meas - dictionary where keys are vehicles, values are numpy array of associated measurements, in same format as data
	# platooninfo - dictionary where keys are vehicles, value is list of useful information
	# platoonlist - list of lists where each nested list is a platoon to be calibrated


    meas, platooninfo, leaders, simcount, curlead, totfollist, followers, curleadlist = \
        helper.makeplatooninfo(data)
    num_of_leaders = len(leaders)
    num_of_vehicles = len(meas.keys()) - num_of_leaders
    platooninfocopy = copy.deepcopy(platooninfo)
    platoonoutput = []
    platoonlist = []

    if not form_platoons:
        return meas, platooninfo

    if vehs is not None: #in this special case we are giving vehicles which we want stuff to be calibrated between
        #note that the first vehicle in vehs is NOT included. So when you are selecting that first vehicle keep that in mind!
        #if you really wanted vehs[0] in the thing you could do this by calling the first makeplatoon with n-1 and putting vehs[0] in the front.

        vehlist = lanevehlist(data,lane,vehs, meas, platooninfo, needmeas = False) #special functions gets only the vehicles we want to simulate out of the whole dataset
        #after having gotten only the vehicles we want to simulate, we modify the platooninfo, leaders , totfollist, to reflect this
        #lastly we can seed curlead as the vehs[0] to start
        platooninfovehs = platooninfo
        platooninfo = {}
        for i in vehlist:
            platooninfo[i] = copy.deepcopy(platooninfovehs[i])
            templead = []
            tempfol = []
            for j in platooninfo[i][4]:
                if j in vehlist:
                    templead.append(j)
            for j in platooninfo[i][-1]:
                if j in vehlist:
                    tempfol.append(j)

            platooninfo[i][4] = templead
            platooninfo[i][-1] = tempfol

        #platooninfo is updated now we need to update the totfollist, simcount, and leaders.
        curlead = vehs[0] #curlead (first vehicle algo starts from) should be the first vehicle in vehs
        simcount = 0
        for i in vehlist:
            if len(platooninfo[i][-1]) >0:
                simcount += 1
        leaders = []
        for i in vehlist:
            if len(platooninfo[i][4]) == 0:
                leaders.append(i)
        totfollist = []
        for i in leaders:
            for j in platooninfo[i][-1]:
                totfollist.append(j)
        totfollist = list(set(totfollist))

    vehicles_added = X
    while simcount > 0:

        if platoonlist:
            previousPlatoon = platoonlist[-1]
        else:
            previousPlatoon = leaders #just give an arbitrary platoon to initialize previousPlatoon

        platooninfo, leaders, simcount, curlead, totfollist, vehicles_added, platoons = makeplatoon(
            platooninfo, leaders, simcount, curlead, totfollist, vehicles_added,
            meas=meas, cycle_num=cycle_num, n=n, cirdep = cirdep, X=X, Y=Y, previousPlatoon=previousPlatoon, maxn = maxn)
        platoonlist.extend(platoons)
        #append it to platoonoutput (output from the function)
        platoonoutput.append(platoons)

    platooninfo = platooninfocopy

    def getUseless(platoons, platooninfo, meas):
        cmetriclist = []  # True if useless, False otherwise, counts number of useless vehicles
        useless = []  # for every useless vehicle, tuple of (vehicle, platoon, platoonindex)
        platind = {}
        for platcount, i in enumerate(platoons):
            for count, j in enumerate(i):
                platind[j] = platcount
                T = set(range(platooninfo[j][1], platooninfo[j][2] + 1))
                cur = helper.c_metric(j, i, T, platooninfo, meas=meas)

                cur2 = helper.c_metric(j, i, T, platooninfo, meas=meas, metrictype='follower')

                if cur == 0 and cur2 == 0:
                    cmetriclist.append(True)
                    useless.append((j, i, platcount))
                else:
                    cmetriclist.append(False)
        return useless, platind

    useless, platind = getUseless(platoonlist, platooninfo, meas) #list of tuple (vehicle, platoon, platoonindex) for each useless vehicle
    # print("Useless vehlcles before:", len(useless))
    mustbeuseless = []
    for i in useless:
        veh = i[0]
        index = i[2]

        #check if the vehicle must be useless
        simulated = False
        leaders = platooninfo[veh][4]
        followers = platooninfo[veh][7]
        for j in leaders:
            if platooninfo[j][2] - platooninfo[j][1]!=0:
                simulated = True
                break
        if not simulated and len(platooninfo[veh][-1])==0: #if must be useless
            mustbeuseless.append(i)
            continue

        #check for a better platoon to put a vehicle in
        leadscore = -math.inf
        folscore = -math.inf
        leadersind = []
        for j in leaders:
            if platooninfo[j][1] == platooninfo[j][2]: #these vehicles don't have entries in platind
                continue
            else:
                leadersind.append(platind[j])
        followersind = [platind[j] for j in followers]
        T = set(range(platooninfo[veh][1],platooninfo[veh][2]+1))
        if len(leadersind)>0:
            leadind = max(leadersind)
            leadscore = helper.c_metric(veh, platoonlist[leadind], T, platooninfo, meas=meas) + helper.c_metric(veh, platoonlist[leadind], T, platooninfo, meas=meas, metrictype = 'follower')
        if len(followersind)>0:
            folind = min(followersind)
            folscore = helper.c_metric(veh, platoonlist[folind], T, platooninfo, meas=meas) + helper.c_metric(veh, platoonlist[folind], T, platooninfo, meas=meas,metrictype = 'follower')
        if leadscore != -math.inf or folscore != -math.inf: #if there is a viable platoon to put vehicle into
            if leadscore > folscore:  #put it into the better one
                platoonlist[index].remove(veh)
                platoonlist[leadind].append(veh)
                platind[veh] = leadind
            else:
                platoonlist[index].remove(veh)
                platoonlist[folind].append(veh)
                platind[veh] = folind

        #old way
#        done = False
#        for j in platoonlist:
#            for k in j:
#                if k in followers:
#                    j.append(veh)
#                    cirdep_list = cirdep_metric([j], platooninfo, meas, metrictype='veh')
#                    if not cirdep_list:
#                        platoonlist[index].remove(veh)
#                        done = True
#                        break
#                    else:
#                        j.remove(veh)
#            if done:
#                break

    count = 0
    for i in mustbeuseless: #add the useless vehicles by themselves
        platoonlist[i[2]+count].remove(i[0])
        platoonlist.insert(i[2]+count,[i[0]])
        count += 1

    platoonlist = [j for j in platoonlist if j != []]
    # useless2, platind = getUseless(platoonlist, platooninfo, meas)
    # print("Useless vehlcles after:", len(useless2))
    # print("Vehicles which must be useless:", len(mustbeuseless))

    if not extra_output:
        return meas, platooninfo, platoonlist
    else:
        return meas, platooninfo, platoonlist, platoonoutput, num_of_leaders, num_of_vehicles

def makeplatoonlist_s(data, n = 1, lane = 1, vehs = []):
    #this makes platoon lists by sorting all the vehicles in a lane, and then
    #simply groups the vehicles in the order they were sorted.
    #only works in a single lane. Supports either entire lane (vehs = [])
    #or for between vehs.

    if len(vehs) > 0:
        sortedvehID, meas, platooninfo = lanevehlist(data, lane, vehs, None, None, needmeas = True)
    else:
        meas, platooninfo = makeplatoonlist(data,1,False)
        vehIDs = np.unique(data[data[:,7]==lane,0])
        # sortedvehID = sortveh3(vehIDs,lane,meas,platooninfo) #algorithm for sorting vehicle IDs
        sortedvehID = sortveh(lane, meas, vehIDs)

    sortedplatoons = []
    nvehs = len(sortedvehID)
    cur, n = 0, 5
    while cur < nvehs:
        curplatoon = []
        curplatoon.extend(sortedvehID[cur:cur+n])
        sortedplatoons.append(curplatoon)
        cur = cur + n
    return meas, platooninfo, sortedplatoons

def lanevehlist(data, lane, vehs, meas, platooninfo, needmeas = False):
    #finds all vehicles between vehs[0] and vehs[1] in a specific lane, and returns the SORTED
    #list of vehicle IDs.
    #different than lanevehlist2 since it returns a sortedlist back. Also it should be more robust.
    #does not support the edge case where the vehs[0] or vehs[1] are involved in circular dependency.
    #the vehicles involved in the circular dependency may, or may not, be included in that case.

    #if needmeas = True, then it gets meas and platooninfo for you, and returns those in addition to the sortedvehlist.
    #otherwise it just returns the sortedvehlist.
    if needmeas:
        meas, platooninfo = makeplatoonlist(data, 1, False)

#    data = data[data[:,7] == lane]
#    veh0 = vehs[0]; vehm1 = vehs[-1]
#    veh0traj = data[data[:,0]==veh0]
#    vehm1traj = data[data[:,0]==vehm1]
#    firsttime = veh0traj[0,1]; lasttime = vehm1traj[-1,1]
    firsttime = platooninfo[vehs[0]][0]
    lasttime = platooninfo[vehs[-1]][3]
    data = data[np.all([data[:,1]>=firsttime, data[:,1]<=lasttime, data[:,7] == lane],axis=0)]
    vehlist = list(np.unique(data[:,0]))

    # sortedvehlist = sortveh3(vehlist, lane, meas, platooninfo)
    sortedvehlist = sortveh(lane, meas, vehlist)
    for count, i in enumerate(sortedvehlist):
        if i == vehs[0]:
            inds = [count]
        elif i == vehs[-1]:
            inds.append(count)
    sortedvehlist = sortedvehlist[inds[0]:inds[1]+1]
    if needmeas:
        return sortedvehlist, meas, platooninfo
    else:
        return sortedvehlist

def sortveh(lane, meas, vehset = None, verbose = False, method = 'leadfol'):
    #lane - lane to sort
    #meas - data in dictionary format
    #vehset - specify a set or list of vehicles, or if None, we will get every vehicle in meas in lane
    #verbose - give warning for ambiguous orders
    #method = 'leadfol' - leadfol enforces leader/follower relationships in the order, using a heuristic to break ties.
        #if method is not leadfol, only the heuristic is used.

    #returns - sorted list of vehicles

    #explanation of algorithm -
    #we find a position 'minpos' and get the times all vehicles pass this position. #then we sort vehicles according to those times
    #for vehicles which don't pass that position, they are added one at a time by finding vehicles they can fit between.
    #we do that by using overlaphelp to find positinos we can compare, then use time_at_pos to get times at the position, compare the times
    #If the order is ambiguous, we order vehicles using their leader/follower relationships ('leadfol' method). We also break ties
    #using a heuristic strategy in sortveh_heuristic.
    #if the method is not 'leadfol', only the heuristic strategy is used.
    #the leadfol method isn't 100% perfect
    #there are also some rare edge cases where the order is ambiguous
    #in particular there may be some edge cases where there are chains of vehicles which can't be compared to each other, but the beginning of the chains
    #both share a common leader. E.g. [230, 231, 240, 245, 257, 267] or [3343, 299, 3344, 311, 309]
    #we do not consistently handle circular dependencies
        #one way to handle circular dependency case is to check for it when adding vehicle (in last for loop) -
            #if found, we could order  all the vehicles in the circular dep and add all at once
            #this is a lot of work for an edge case though

    #to make this '100% perfect' you would need to give all vehicles a score representing it's 'leader score'
    #if you used this 'leader score' instead of veh2ind to initialize the negative scores, then the order would be perfect.
    #Issue now is when a leader has multiple followers which don't overlap at all,
    #and each of the followers also has its own (sub-)follower, where the (sub-)follower overlaps only with the follower
    #In that case (e.g. (230, 231, 240, 245, 257, 267)), 230/231 are the followers - their order is arbitrary and they need to have the same score.
    #240 and 245 both have 230 or 231 as a leader, and need 1 higher score. But currently because we use veh2ind for the score, the order isn't arbitrary
    #because whichever of the 230 or 231 comes first, they give their follower higher priority.
    #fairly complicated to implement these leader scores as it forces you to go through all the relationships when adding vehicles as opposed to
    #the current way where we can simply iterate through all the vehicles

    vehset = vehset.copy() #avoid making in-place modifications to input
    #get list of vehicles to sort
    if vehset == None:
        vehset = set()
        for vehid in meas.keys():
            if lane in meas[vehid][:,7]:
                vehset.add(vehid)
    if type(vehset) == list:
        vehset = set(vehset)

    #make data for algo
    all_traj = {} #trajectories in lane
    minpos = math.inf #minimum position
    guessvehs = {} #values are vehicles we guess are close to key
    leads = {}
    for veh in vehset:
        lanedata = meas[veh]
        lanedata = lanedata[lanedata[:,7]==lane]
        all_traj[veh] = lanedata

        leaders = set(np.unique(lanedata[:,4]))
        leads[veh] = leaders
        temp = leaders.union(set(np.unique(lanedata[:,5])))
        guessvehs[veh] = temp.intersection(vehset) #guessvehs has all leaders/followers
        if lanedata[0,2] < minpos:
            minpos = lanedata[0,2]
    minpos += 50 #magic number

    #sort many vehicles with some common position
    sorted_veh = [] #sorted vehicles - the output
    initial_sort = {} #sort vehicles at beginning
    for veh in vehset:
        time = time_at_pos(all_traj[veh], minpos)
        if time != None:
            initial_sort[veh] = time
            sorted_veh.append(veh)

    veh2ind = {} #maps from vehicles to index
    sorted_veh = sorted(sorted_veh, key = lambda veh: initial_sort[veh])
    for count, veh in enumerate(sorted_veh):
        veh2ind[veh] = count
        vehset.remove(veh)

    #add each remaining vehicle in vehset
    for veh in vehset:
        #find initial guess #########
        for guessveh in guessvehs[veh]:
            if guessveh in veh2ind:
                curind = veh2ind[guessveh]

                #find a position we can compare
                posoverlap = overlaphelp(all_traj[veh], all_traj[guessveh], True, True, False, True)
                if len(posoverlap) == 0:
                    #go to nextguess
                    continue
                else:
                    usepos = (posoverlap[0][0]+posoverlap[0][1])/2
                    timeveh = time_at_pos(all_traj[veh], usepos)
                    # if timeveh == None:
                    #     continue
                    timeguess = time_at_pos(all_traj[guessveh],usepos)
                    # if timeveh != None and timeguess != None:
                    if timeveh < timeguess: #search backwards
                        search = 'b'
                        firstind = 0
                        lastind = curind
                        break
                    else: #search forwards
                        search = 'f'
                        firstind = curind+1
                        lastind = len(sorted_veh)
                        break
        else: #search list for starting guess
            for curind in range(len(sorted_veh)):
                guessveh = sorted_veh[curind]
                #copy paste code from above

                #find a position we can compare
                posoverlap = overlaphelp(all_traj[veh], all_traj[guessveh], True, True, False, True)
                if len(posoverlap) == 0:
                    #go to nextguess
                    continue
                else:
                    usepos = (posoverlap[0][0]+posoverlap[0][1])/2
                    timeveh = time_at_pos(all_traj[veh], usepos)
                    # if timeveh == None:
                    #     continue
                    timeguess = time_at_pos(all_traj[guessveh],usepos)
                    # if timeveh != None and timeguess != None:
                    if timeveh < timeguess: #search backwards
                        search = 'b'
                        firstind = 0
                        lastind = curind
                        break
                    else: #search forwards
                        search = 'f'
                        firstind = curind + 1
                        lastind = len(sorted_veh)
                        break

            else:
                print('failed to find a guess for '+str(veh)+ ' so it cannot be sorted')

        vehinds = sortveh_helper_search(firstind, lastind, search, veh, sorted_veh, all_traj)
        if vehinds[-1] == 0: #at to beginning
            sorted_veh.insert(0, veh)
            veh2ind = update_inds(sorted_veh, veh2ind, 0)
        elif vehinds[0] == len(sorted_veh): #add at end
            sorted_veh.append(veh)
            veh2ind[veh] = len(sorted_veh)-1
        elif vehinds[-1]-1 == vehinds[0]: #if it fits perfectly we can add
            sorted_veh.insert(vehinds[-1],veh)
            veh2ind = update_inds(sorted_veh, veh2ind, vehinds[-1])
        else: #in this case we need to order everything in the interval we found
            probvehs = sorted_veh[vehinds[0]+1:vehinds[1]] #want to sort probvehs and veh

            if method == 'leadfol':
                #leader/follower based approach - ###
                #prepare data
                sortedleads = {} #keys are vehicles, values are leaders for key which have been sorted
                probleads = {} #values are leaders for key which are in allprobvehs
                allprobvehs = probvehs.copy()
                allprobvehs.append(veh) #we want to sort allprobvehs
                for i in allprobvehs:
                    sortedleads[i] = leads[i].intersection(sorted_veh).difference(probvehs)
                    probleads[i] = leads[i].intersection(allprobvehs)

                #assign scores to each veh based on lead relationships
                # #initialize scores old
                # leadfoltiebreak = {i:0 for i in allprobvehs} #values are a score based on leadfollower relationships - we use this to sort at the end
                # for i in allprobvehs:
                #     mymax = -math.inf
                #     for j in sortedleads[i]:
                #         temp = veh2ind[j] - vehinds[0] #or just always -1/1?
                #         if temp > mymax:
                #             mymax = temp
                #     if mymax != -math.inf:
                #         leadfoltiebreak[i] = mymax
                #new way maybe works
                leadfoltiebreak = {i:1 for i in allprobvehs} #values are a score based on leadfollower relationships - we use this to sort at the end
                for i in allprobvehs:
                    if len(sortedleads[i]) > 0:
                        leadfoltiebreak[i] = 0
                #give scores
                for i in range(len(allprobvehs)-1):
                    for j in allprobvehs:
                        for k in probleads[j]:
                            if leadfoltiebreak[k] >= leadfoltiebreak[j]:
                                leadfoltiebreak[j] = leadfoltiebreak[k] + 1
                #check for ties in score
                chkties = {}
                for i in allprobvehs:
                    score = leadfoltiebreak[i]
                    if score in chkties:
                        chkties[score].append(i)
                    else:
                        chkties[score] = [i]

                #resolve ties if necessary
                dt= 1e-5 #if we resolve ties its by adding this small amount
                for score in chkties.keys():
                    if len(chkties[score]) > 1:
                        temp = chkties[score]
                        fixedvehs, success = sortveh_heuristic(temp[0], temp[1:], all_traj, meas, guessvehs)
                        if not success:
                            if verbose:
                                print('warning - ambiguous order for '+str(temp))
                            tiebreaking = [all_traj[curveh][0,1] for curveh in probvehs]
                            mytime = all_traj[veh][0,1]
                            for count, j in enumerate(tiebreaking):
                                if mytime < j:
                                    ind = count
                            else:
                                ind = count + 1
                            fixedvehs = probvehs.copy()
                            fixedvehs.insert(ind, veh)
                        for count, i in enumerate(fixedvehs):
                            leadfoltiebreak[i] += count*dt
                #sorting from lead/fol method
                fixedvehs = sorted(allprobvehs, key = lambda veh: leadfoltiebreak[veh])

                #if vehicles have negative values they can get put before the interval
                negind = len(fixedvehs) #negind is the number of vehicles that get added before vehinds[0]
                for count, i in enumerate(fixedvehs):
                    if leadfoltiebreak[i] >= 0:
                        negind = count
                        break
                # negind = 0 #if you can give negative scores then this turns it off.########
                negvehs = fixedvehs[0:negind]
                fixedvehs = fixedvehs[negind:]

                #updat vehicles that get put before interval
                needinsert = True
                if negind == 0:
                    pass
                else:
                    for i in negvehs:
                        if i == veh: #i is new vehicle
                            tempind = math.floor(leadfoltiebreak[i]) + vehinds[0] + 1 #where it gets inserted
                            sorted_veh.insert(tempind, i)
                            needinsert = False
                            veh2ind = update_inds(sorted_veh, veh2ind, tempind)
                            vehinds = (vehinds[0]+1, vehinds[1])

                        else: #i has already been sorted
                            tempind = math.floor(leadfoltiebreak[i]) + vehinds[0] + 1 #where it needs to go
                            sorted_veh.pop(veh2ind[i])
                            sorted_veh.insert(tempind,i)
                            veh2ind = update_inds(sorted_veh, veh2ind, tempind)
                            vehinds = (vehinds[0]+1, vehinds[1])

                #update sorted list
                if len(fixedvehs) == 0:
                    pass
                else:
                    if needinsert:
                        sorted_veh.insert(vehinds[0]+1, veh) #we only insert to add an extra vehicle - this gets overwritten
                    for count,i in enumerate(range(vehinds[0]+1, vehinds[1]+1)):
                        sorted_veh[i] = fixedvehs[count]
                    veh2ind = update_inds(sorted_veh, veh2ind, vehinds[0]+1)

            else:
                #heuristic based approach - ### #old method
                fixedvehs, success = sortveh_heuristic(veh, probvehs, all_traj, meas, guessvehs)

                if success:
                    #fixedvehs has the correct order - now update sorted_veh and veh2ind
                    sorted_veh.insert(vehinds[0]+1, veh)
                    for count,i in enumerate(range(vehinds[0]+1, vehinds[1]+1)):
                        sorted_veh[i] = fixedvehs[count]
                    veh2ind = update_inds(sorted_veh, veh2ind, vehinds[0]+1)
                else:
                    #we failed to completely sort the vehicles and are left with an ambiguous order
                    #arbitrary
                    # sorted_veh.insert(vehinds[0]+1, veh)
                    # veh2ind = update_inds(sorted_veh, veh2ind, vehinds[0]+1)

                    #maybe this is a better tie breaking rule
                    tiebreaking = [all_traj[curveh][0,1] for curveh in probvehs]
                    mytime = all_traj[veh][0,1]
                    for count, j in enumerate(tiebreaking):
                        if mytime < j:
                            ind = count + vehinds[0]+1
                            break
                    else:
                        ind = vehinds[1]
                    sorted_veh.insert(ind, veh)
                    veh2ind = update_inds(sorted_veh, veh2ind, ind)
                    if verbose:
                        print('warning - ambiguous order for '+ str(sorted_veh[vehinds[0]+1:vehinds[1]+1])+' between vehicles '+str((sorted_veh[vehinds[0]], sorted_veh[vehinds[1]+1])))




    return sorted_veh

def sortveh_heuristic(veh, probvehs, all_traj, meas, guessvehs):
    #a - try to find a consistent position for all vehicles###
    for i in range(len(probvehs)):
        if i == 0:
            posoverlap = overlaphelp(all_traj[veh], all_traj[probvehs[0]],True, True, False, True)
        else:
            posoverlap = overlaphelp(posoverlap, all_traj[probvehs[i]], False, True, False, True)
        if len(posoverlap)==0:
            break
    else: #if we get through for loop then we found overlap
        usepos = (posoverlap[0][0] + posoverlap[0][1])/2
        tiebreaking = {} #all vehicles get sorted using the position we found
        for i in probvehs:
            tiebreaking[i] = time_at_pos(all_traj[i],usepos)
        tiebreaking[veh] = time_at_pos(all_traj[veh],usepos)
        fixedvehs = probvehs.copy()
        fixedvehs.append(veh)
        fixedvehs = sorted(fixedvehs, key = lambda veh: tiebreaking[veh])

        return fixedvehs, True

    #b - try to find a consistent time for all vehicles###
    for i in range(len(probvehs)):
        if i == 0:
            timeoverlap = overlaphelp(all_traj[veh], all_traj[probvehs[0]],True, True, False, False)
        else:
            timeoverlap = overlaphelp(timeoverlap, all_traj[probvehs[i]], False, True, False, False)
        if len(timeoverlap)==0:
            break
    else: #if we get through for loop then we found overlap
        usetime = timeoverlap[0][0]
        tiebreaking = {}
        for i in probvehs:
            t_nstar = meas[i][0,1]
            tiebreaking[i] = meas[i][int(usetime - t_nstar),2]
        t_nstar = meas[veh][0,1]
        tiebreaking[veh] = meas[veh][int(usetime - t_nstar),2]
        fixedvehs = probvehs.copy()
        fixedvehs.append(veh)
        fixedvehs = sorted(fixedvehs, key = lambda veh: tiebreaking[veh], reverse = True)

        return fixedvehs, True

    #c - try to find consistent leader/follower for all vehicles###
    leadfoloverlap = guessvehs[veh]
    for i in probvehs:
        leadfoloverlap = leadfoloverlap.intersection(guessvehs[i])
        if len(leadfoloverlap) == 0:
            break
    else: #if we get through for loop then we found overlap
        fixveh = leadfoloverlap.pop()
        fixvehtraj = all_traj[fixveh]
        tiebreaking = {}
        for i in probvehs:
            data1, data2 = overlaphelp(fixvehtraj, all_traj[i])
            tiebreaking[i] = np.mean(data1[:,2] - data2[:,2])
        data1, data2 = overlaphelp(fixvehtraj, all_traj[veh])
        tiebreaking[veh] = np.mean(data1[:,2] - data2[:,2])
        fixedvehs = probvehs.copy()
        fixedvehs.append(veh)
        fixedvehs = sorted(fixedvehs, key = lambda veh: tiebreaking[veh])

        return fixedvehs, True

    return None, False

def update_inds(sorted_veh, veh2ind, startind):
    #helper for sortveh
    for ind in range(startind, len(sorted_veh)):
        veh = sorted_veh[ind]
        veh2ind[veh] = ind
    return veh2ind

def sortveh_helper_search(firstind, lastind, search, veh, sorted_veh, all_traj):
    #firstind, lastind are indexes of sorted_veh we search in.
    #if search = 'b' we are looking backwards for the vehicle in front of veh
        #otherwise search = 'f' and we look forwards for the vehicle behind veh

    #return a tuple of indexes which bracket where veh belongs in sorted_veh. will always return

    inds = range(firstind, lastind)
    if search == 'b':
        inds = reversed(inds)
        prevvehind = lastind
        vehind = 0
    else:
        prevvehind = firstind - 1
        vehind = lastind

    # vehinds = prevvehind
    for guessind in inds:
        guessveh = sorted_veh[guessind]
        #find a position we can compare
        posoverlap = overlaphelp(all_traj[veh], all_traj[guessveh],True, True, False, True)
        if len(posoverlap) == 0:
            #go to nextguess
            continue
        else:
            usepos = (posoverlap[0][0]+posoverlap[0][1])/2
            timeveh = time_at_pos(all_traj[veh], usepos)
            timeguess = time_at_pos(all_traj[guessveh],usepos)

            if search == 'b':
                if timeveh > timeguess:
                    vehind = guessind
                    break
                else:
                    prevvehind = guessind
            else:
                if timeveh < timeguess:
                    vehind = guessind
                    break
                else:
                    prevvehind = guessind
    if search == 'b':
        return (vehind, prevvehind)
    else:
        return (prevvehind, vehind)


def time_at_pos(traj, pos, reverse = False):
    #given trajectory traj, returns time that the trajectory crosses pos
    #if reverse, we search traj in reversed order.
    #implementation is basically a for loop

    inds = range(len(traj))
    if not reverse:
        prev = traj[0,2]
        if prev > pos:
            return None
        for i in inds:
            cur = traj[i,2]
            if cur > pos and pos > prev:
                if traj[i,1] -1 == traj[i-1,1]:
                    return traj[i,1]
                else:
                    return None
            prev = cur
        else:
            return None
    if reverse: #copy pasted
        inds = reversed(inds)
        cur = traj[-1,2]
        if cur < pos:
            return None
        for i in inds:
            prev = traj[i,2]
            if cur > pos and pos > prev:
                if traj[i,1] -1 == traj[i-1,1]:
                    return traj[i,1]
                else:
                    return None
            cur = prev
        else:
            return None


def overlaphelp(meas1, meas2, meas1_isdata = True, meas2_isdata = True, return_data = True, get_pos = False):
    #meas1 - either a list of tuples, or a 2d array of data with rows as observations
    #meas2 - either a list of tuples, or a 2d array of data with rows as observations
    #meas1_isdata - if True, meas1 is data, otherwise meas1 is a list of tuple
    #meas2_isdata - if True, meas2 is data, otherwise meas1 is a list of tuple
    #return_data - If True, we return two 2d arrays which give all observations for meas1, meas2 with the same times
        #if False, we return a list of tuples where each tuple gives the overlap between the data
        #if get_pos = True, return_data must be False
    #get_pos - whether the function finds overlaps in positions (if True), or overlap in times

    #this function can compute the overlap between two datas, either overlap in times (times where both data have observations)
    #or overlap in pos (positions where both data have sequential observations)

    #common format for this function is a list of tuples, where each tuple is representing the boundaries for a region
    #this is meant to be used when you have two data like meas1[meas1[:,7]==2], meas2[meas2[:,7]==2] and want to compute their overlaps. Used for sorting
    #can be used to compute their overlaps in positions or times, and can be used to compute the overlap between several vehicles by using the options


    #get indices of the sequential data
    if get_pos:
        if not return_data:
            return_data = False

    if meas1_isdata:
        ind1 = helper.sequential(meas1)
        if get_pos:
            times1 = helper.indtopos(ind1, meas1)
        else:
            times1 = helper.indtotimes(ind1,meas1)
    else:
        times1 = meas1

    if meas2_isdata:
        ind2 = helper.sequential(meas2)
        if get_pos:
            times2 = helper.indtopos(ind2, meas2)
        else:
            times2 = helper.indtotimes(ind2, meas2)
    else:
        times2 = meas2

    # if return_pos:
    #     if not return_times:
    #         print('setting return_times to True')
    #         return_times = True #must be in this format for return_pos
    #         #possible to give an option to
    #     if input_pos: #True True True
    #         times1 = meas1
    #         times2 = meas2
    #     else: #True True
    #         ind1 = helper.sequential(meas1)
    #         ind2 = helper.sequential(meas2)
    #         times1 = helper.indtopos(ind1, meas1)
    #         times2 = helper.indtopos(ind2, meas2)
    # else: #____ False False
    #     #change the indices into times
    #     ind1 = helper.sequential(meas1)
    #     ind2 = helper.sequential(meas2)
    #     times1 = helper.indtotimes(ind1,meas1) #call these times but really they are the times for slices, i.e. second time has 1 extra
    #     times2 = helper.indtotimes(ind2,meas2)
    #output
    outtimes = []
    outind1 = []
    outind2 = []
    #track of where we are
    count1 = 0
    prevcount2 = 0
    while count1 < len(times1): #iterate over the first meas
        cur1 = times1[count1]
        count2 = prevcount2
        while count2 < len(times2): #iterate over the second meas
            cur2 = times2[count2]
            if cur2[0] < cur1[0] and cur2[1] < cur1[0]: #trivial case, check next 2 block
                pass
            elif cur2[0] > cur1[1] and cur2[1] > cur1[1]: #other trivial case, done checking 2 blocks and check next 1 block
                break
            elif cur2[0] <= cur1[0] and cur2[1] >= cur1[0] and cur2[1] <= cur1[1]:
                curtimes = (cur1[0], cur2[1]) #actual times of observations in slices format
                #convert times into output type
                overlaphelp_help(curtimes, outind1, outind2, outtimes, cur1, cur2, return_data)

            elif cur2[0] <= cur1[0] and cur2[1] >= cur1[1]:
                curtimes = (cur1[0],cur1[1])
                overlaphelp_help(curtimes, outind1, outind2, outtimes, cur1, cur2, return_data)
                break
            elif cur1[0] <= cur2[0] and cur1[1] >= cur2[1]:
                curtimes = (cur2[0],cur2[1])
                overlaphelp_help(curtimes, outind1, outind2, outtimes, cur1, cur2, return_data)

            else: #cur1[0] < cur2[0] and cur1[1] < cur2[1]
                curtimes = (cur2[0], cur1[1])
                overlaphelp_help(curtimes, outind1, outind2, outtimes, cur1, cur2, return_data)
                break
            #update iteration
            count2 += 1
        count1 += 1
        prevcount2 = count2

    #output
    if return_data:
        out1, out2 = [], []
        for i in outind1:
            out1.append(meas1[int(i[0]):int(i[1])])
        for i in outind2:
            out2.append(meas2[int(i[0]):int(i[1])])
        if len(out1) == 0: #handle case of empty array
            dim2 = np.shape(meas1)[1]
            out1 = np.zeros((0,dim2))
            out2 = out1
            return out1, out2
        dims = np.shape(out1)
        # try:
            # out1 = np.reshape(out1, (dims[0]*dims[1], dims[2]))
        out1 = np.concatenate(out1, axis = 0)
        # except:
        #     print('hello')
        # out2 = np.reshape(out2, (dims[0]*dims[1], dims[2]))
        out2 = np.concatenate(out2, axis =0)
        return out1, out2
    else:
        return outtimes
    # dim = np.shape(meas1)[1]
    # out1 = np.zeros((0,dim))
    # for i in outind1:
    #     out1 = np.append(out1,meas1[int(i[0]):int(i[1])],axis=0)
    # out2 = np.zeros((0,dim))
    # for i in outind2:
    #     out2 = np.append(out2,meas2[int(i[0]):int(i[1])],axis=0)


def overlaphelp_help(curtimes, outind1, outind2, outtimes, cur1, cur2, return_data):
    if not return_data:
        # curtimes[1] = curtimes[1] - 1
        outtimes.append(curtimes)
    else:
        temp1 = (curtimes[0]-cur1[0]+cur1[2], curtimes[1]-cur1[0]+cur1[2])
        outind1.append(temp1)
        temp1 = (curtimes[0]-cur2[0]+cur2[2],curtimes[1]-cur2[0]+cur2[2])
        outind2.append(temp1)

def checksort(vehlist, meas, lane):
    # very simple function plots trajectory in a line, allows you to debug the platoon order
    fig = plt.figure()
    for i in range(len(vehlist)):
        cur = meas[vehlist[i]]
        cur = cur[cur[:, 7] == lane]
        plt.plot(cur[:, 1], cur[:, 2], picker=5)

    def on_pick(event):
        ax = event.artist.axes
        curind = ax.lines.index(event.artist)  # artist index
        plt.title('selected vehicle ' + str(vehlist[curind]) + ' which has order ' + str(curind))
        plt.draw()

    fig.canvas.callbacks.connect('pick_event', on_pick)
    pass
