#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from drl import *
#%%
p = [33.33, 1.2, 2, 1.1, 1.5] #parameters for human drivers
initstate, auxinfo, roadinfo = eq_circular(p, IDM_b3, update2nd_cir, IDM_b3_eql, 41, length = 2, L = None, v = 15, perturb = 2) #create initial state on road
sim, curstate, auxinfo = simulate_cir(initstate, auxinfo,roadinfo, update_cir, timesteps = 25000, dt = .25)
vlist = {i: curstate[i][1] for i in curstate.keys()}
avid = min(vlist, key=vlist.get)

"""
Record the possible hyperparam values
"""
lr_vals = np.around(np.arange(2e-4,1.1e-3,1e-4),4) # lr arg in ACagent
entropy_vals = [1e-5, 5e-5, 1e-4, 5e-4]  # entropy_const arg in ACagent
nstep_vals = [5,10,20,math.inf] # nTDsteps arg in train()

netdepths = [2,3,4] # num_hiddenlayers
numneuron_vals = [32,64,128] # num_neurons
activations = [kl.Activation('relu'),kl.LeakyReLU(alpha = 0.3),kl.Activation('tanh')] # activationlayer

statemem_vals = [1,5,10] # statemem in circ_singleav()

"""
Setup the primary data structure for the experiment.

Returns:
    dict: The collection of initialized hyperparameters.
"""
def exp_setup():
    exp_params = {}
    exp_params["lr"] = lr_vals[0]
    exp_params["ev"] = entropy_vals[0]
    exp_params["nstep"] = nstep_vals[0]
    exp_params["netdP"] = netdepths[0]
    exp_params["nneurP"] = numneuron_vals[0]
    exp_params["actP"] = activations[0]
    exp_params["netdV"] = netdepths[0]
    exp_params["nneurV"] = numneuron_vals[0]
    exp_params["actV"] = activations[0]
    exp_params["sm"] = statemem_vals[0]
    return exp_params

"""
Train [agent_tt] in the [testenv_tt] environmnent for [nTDsteps_tt] steps, then test.

Args:
    agent_tt (ACAgent): the agent
    testenv_tt (circ_singleav): the environment
    nTDsteps_tt (int): #steps to calculate TDerrors

Returns:
    float: mean test reward achieved
    float: mean test episode length
"""
def traintest(agent_tt, testenv_tt, nTDsteps_tt):
    # agent_tt.train(testenv_tt, updates=1000, nTDsteps=nTDsteps_tt)
    # rewardslist, eplenlist = agent_tt.test(testenv_tt,1500)
    agent_tt.train(testenv_tt, updates=1, nTDsteps=nTDsteps_tt)
    rewardslist, eplenlist = agent_tt.test(testenv_tt,1)
    return np.mean(rewardslist), np.mean(eplenlist)

"""
Reset the policy and value models.

Args:
    exp_params (dict): Collection of experiment hyperparameters, in the same
    format of exp_setup().

Returns:
    PolicyModel: a new policy model set per the depth, # neurons, and activation
                 in [exp_params].
    ValueModel: a new value model set per the depth, # neurons, and activation
                in [exp_params].
"""
def resetPolVal(exp_params):
    policymodel = PolicyModel(num_actions = 3, num_hiddenlayers=exp_params["netdP"],num_neurons=exp_params["nneurP"],activationlayer=exp_params["actP"])
    valuemodel = ValueModel(num_hiddenlayers=exp_params["netdV"],num_neurons=exp_params["nneurV"],activationlayer=exp_params["actV"])
    return policymodel,valuemodel

"""
Selects best hyperparameter setting among [hyperparam_choices] based on maximal
test rewards.  E.g., selects best learning rate among lr_vals.

Args:
    exp_params (dict): Collection of experiment hyperparameters, in the same
                       format of exp_setup().
    hyperparam_choices (list): Possible hyperparameter values to evaluate.
    exp_params_key (string): exp_params key of hyperparameter for selection.
    title (string): Indication of which hyperparameter is being evaluated.
    activation (bool): Whether the hyperparameter is an activation layer.
    resetP (bool): Whether the hyperparameter concerns the policy model.
    resetV (bool): Whether the hyperparameter concerns the value model.
    resetEnv (bool): Whether the hyperparameter for selection is state memory.

Returns:
    string: Formatted intermediate results indicating test metrics for each
            element in [hyperparam_choices] and the final selected hyperparameter
            value.
"""
def grid_hyperparam_selection(exp_params, hyperparam_choices, exp_params_key, title, activation = False, resetP = False, resetV = False, resetEnv = False):
    def exp_procedure(exp_params, hyperparam):
        agent = ACagent(exp_params["polmodel"], exp_params["valmodel"], lr=exp_params["lr"], entropy_const = exp_params["ev"])
        totreward, testlen = traintest(agent, exp_params["testenv"], exp_params["nstep"])
        exp_params["polmodel"], exp_params["valmodel"] = resetPolVal(exp_params)
        res_fmt = "{}:\t{} reward over {} timesteps\n".format(hyperparam, totreward, testlen)
        return totreward, res_fmt

    div_str = '+'+'-'*42+'+\n'
    def boxed_section(title):
        text = div_str
        text += str.ljust("| {}".format(title),43) + "|\n"
        text += div_str
        return text

    rewards = []
    intermediate_res = boxed_section(title)
    for idx, hyperparam in enumerate(hyperparam_choices):
        exp_params[exp_params_key] = hyperparam

        if resetP:
            exp_params["polmodel"] = PolicyModel(num_actions = 3, \
                                                num_hiddenlayers=exp_params["netdP"],\
                                                num_neurons=exp_params["nneurP"],\
                                                activationlayer=exp_params["actP"])
        if resetV:
            exp_params["valmodel"] = ValueModel(num_hiddenlayers=exp_params["netdV"],\
                                                num_neurons=exp_params["nneurV"],\
                                                activationlayer=exp_params["actV"])

        if resetEnv:
            exp_params["testenv"] = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25,statemem=exp_params["sm"])


        totreward, res_fmt = exp_procedure(exp_params, ['relu','leaky relu','tanh'][idx] if activation else hyperparam) #"{:.4f}".format(hyperparam))

        rewards.append(totreward)
        intermediate_res += res_fmt

    exp_params[exp_params_key] = hyperparam_choices[np.argmax(rewards)]
    sel = ['relu','leaky relu','tanh'][np.argmax(rewards)] if activation else exp_params[exp_params_key] #"{:.4f}".format(exp_params[exp_params_key])
    intermediate_res += "\nSelected: {}\n".format(sel)

    print(intermediate_res)
    return intermediate_res

"""
Runs grid search over all hyperparameters recorded in exp_params.

Args:
 niters (int): # iterations to run grid search.

Returns:
 string: Formatted text results of entire experiment.
"""
def grid_search(niters = 1):
    exp_params = exp_setup()
    res = ''''''
    for i in range(niters):
        # Record which iteration
        res+= div_str + str.ljust("| Iter: {}".format(i),43) + "|\n"
        print("\n".join(res.splitlines()[-2:]))

        # Reset policy model, value model, and test environment
        exp_params["polmodel"], exp_params["valmodel"] = resetPolVal(exp_params)
        exp_params["testenv"] = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25,statemem=exp_params["sm"])

        # Learning Rate
        res += grid_hyperparam_selection(exp_params, lr_vals, "lr", "Learning rate")

        # Entropy
        res += grid_hyperparam_selection(exp_params, entropy_vals, "ev", "Entropy")

        # Nstep for TD errors
        res += grid_hyperparam_selection(exp_params, nstep_vals, "nstep", "Nstep for TD errors")

        # Policy Depth
        res += grid_hyperparam_selection(exp_params, netdepths, "netdP", "Depth (Policy)", resetP=True)

        # Policy Num Neurons
        res += grid_hyperparam_selection(exp_params, numneuron_vals, "nneurP", "Number of neurons in each layer (Policy)", resetP=True)

        # Policy Activation
        res += grid_hyperparam_selection(exp_params, activations, "actP", "Activation (Policy)", activation=True, resetP=True)

        # Value Depth
        res += grid_hyperparam_selection(exp_params, netdepths, "netdV", "Depth (Value)", resetV=True)

        # Value Num Neurons
        res += grid_hyperparam_selection(exp_params, numneuron_vals, "nneurV", "Number of neurons in each layer (Value)", resetV=True)

        # Value Activation
        res += grid_hyperparam_selection(exp_params, activations, "actV", "Activation (Value)", activation=True, resetV=True)

        # State Memory
        res += grid_hyperparam_selection(exp_params, statemem_vals, "sm", "State memory", resetEnv = True)
    return res

"""
Search over random combinations of hyperparameters recorded in exp_params.

Args:
 niters (int): # iterations to run random search.

Returns:
 dict: The entire search history, such that each key is the search iteration,
       and each associated value is the setting of hyperparameters for that
       iteration.    
 (int, dict) tuple: Which iteration achieved the highest total reward, and the
                    associated dictionary recording which hyperparameter
                    settings achieved that max reward.
"""
def rand_search(niters = 5):
    def exp_setup_rand():
        def rand_val(arr):
            return np.random.randint(len(arr))
        exp_params = {}
        exp_params["lr"] = lr_vals[rand_val(lr_vals)]
        exp_params["ev"] = entropy_vals[rand_val(entropy_vals)]
        exp_params["nstep"] = nstep_vals[rand_val(nstep_vals)]
        exp_params["netdP"] = netdepths[rand_val(netdepths)]
        exp_params["nneurP"] = numneuron_vals[rand_val(numneuron_vals)]
        exp_params["actPidx"] = rand_val(activations)
        exp_params["actP"] = activations[exp_params["actPidx"]]
        exp_params["netdV"] = netdepths[rand_val(netdepths)]
        exp_params["nneurV"] = numneuron_vals[rand_val(numneuron_vals)]
        exp_params["actVidx"] = rand_val(activations)
        exp_params["actV"] = activations[exp_params["actVidx"]]
        exp_params["sm"] = statemem_vals[rand_val(statemem_vals)]
        return exp_params

    log_results = {}

    def log_format(config):
        config.pop('polmodel',None)
        config.pop('valmodel',None)
        config.pop('testenv',None)
        actPidx = config.pop("actPidx", None)
        actVidx = config.pop("actVidx", None)
        config["actP"] = ['relu','leaky relu','tanh'][actPidx]
        config["actV"] = ['relu','leaky relu','tanh'][actVidx]

        return config

    for i in range(niters):
        exp_params = exp_setup_rand()
        exp_params["polmodel"], exp_params["valmodel"] = resetPolVal(exp_params)
        exp_params["testenv"] = circ_singleav(curstate, auxinfo, roadinfo, avid, drl_reward8,dt = .25,statemem=exp_params["sm"])

        agent = ACagent(exp_params["polmodel"], exp_params["valmodel"], lr=exp_params["lr"], entropy_const = exp_params["ev"])
        totreward, testlen = traintest(agent, exp_params["testenv"], exp_params["nstep"])

        log_results[i] = log_format(exp_params)
        log_results[i]["Total Reward"] = totreward
        log_results[i]["Test length"] = testlen

    return log_results, max(log_results.items(), key = lambda x: x[1]["Total Reward"])
