"""Code for optimization algorithms."""
import bayes_opt as bo
import tqdm
import pickle


def bayes_opt_wrapper(f, pbounds, n_iter=100, init_points=0, init_guesses=None, save_name=None,
                      prev_opt_name=None):
    """Wrapper for bayes_opt package.

    Args:
        f: function to maximize. It has a call signature f(**args) and the args are passed as a dict, such that the
            index i argument has the key 'i'. Arguments should all be scalar floats. f returns a scalar float.
        pbounds: list of tuples, where each tuple is the upper/lower bounds for the parameter with that index.
        n_iter: int number of iterations to perform (not counting guesses/initialization)
        init_points: int number of initial random points
        init_guesses: if not None, list of lists, where each list represents a set of parameters.
        save_name: if not None, filepath to save an optimization log, and the result of the optimization.
        prev_opt_name: if not None, can be the save_name from a previous run, and the log will be loaded, which
            essentially has the effect of restarting the optimizer.
    Returns:
        optimizer: the BayesianOptimization
    """
    def make_dict(out):
        return {str(count): i for count, i in enumerate(out)}

    def make_opt_str(opt):
        evals = (opt.max['target'], opt.res[-1]['target'], sum([i['target'] for i in opt.res[-10:]])/len(opt.res[-10:]))
        out = 'Best Evaluation: {:.1f}, Most Recent: {:.1f}, Moving Average: {:.1f}. '.format(*evals)
        out += 'Best parameters: ' + str(list(opt.max['params'].values())) + '. '
        out += 'Most recent: ' + str(list(opt.res[-1]['params'].values())) + '.'
        return out

    optimizer = bo.BayesianOptimization(f=f, pbounds=make_dict(pbounds), allow_duplicate_points=True, verbose=1)
    if prev_opt_name is not None:
        bo.util.load_logs(optimizer, logs=[prev_opt_name+'.json'])
    if save_name is not None:
        logger = bo.logger.JSONLogger(path=save_name)
        optimizer.subscribe(bo.Events.OPTIMIZATION_STEP, logger)
    n_init_guesses = len(init_guesses) if init_guesses is not None else 0
    total_iters = n_iter+init_points + n_init_guesses
    pbar = tqdm.tqdm(range(total_iters), total=total_iters, leave=True, position=0)
    pbar.set_description('Function Evaluations')
    pbar.set_postfix_str('idk test')

    for cur_iter in pbar:
        if cur_iter < n_init_guesses:
            optimizer.probe(params=make_dict(init_guesses[cur_iter]), lazy=False)
        elif cur_iter < n_init_guesses + init_points:
            optimizer.maximize(init_points=1, n_iter=0)
        else:
            optimizer.maximize(init_points=0, n_iter=1)
        pbar.set_postfix_str(make_opt_str(optimizer))

    if save_name is not None:
        with open(save_name+'_res.pkl', 'wb') as f:
            pickle.dump(optimizer.res, f)
    return optimizer
