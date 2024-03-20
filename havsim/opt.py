"""Code for optimization algorithms/scripts."""
import bayes_opt as bo
import tqdm
import pickle
import sys
import os
import argparse
import ast
import inspect


def parse_args(arg_names, default_args=None, description_str=None, arg_descriptions=None, n_pos_args=None):
    """Implements extra functionality when calling scripts from a python console.

    This uses the package argparse to automatically generate docstring for the script when given -help/-h option.
    This also implements the ability to use keyword arguments when calling scripts, so scripts may accept
    keyword arguments with syntax as if it were a python function. Various checks are done so that
    a suitable error message can be given in case the call fails.

    Args:
        arg_names: list of str names, the names of the variables which should be defined for the script, in order.
        default_args: list of default values for any keyword arguments. must be the same length or shorter
            than arg_names. Any variables which do not have default values are assumed to be positional arguments.
            Note that any positional arguments must come first in the order.
        description_str: None, or str to set as description in -help/-h docstring
        arg_descriptions: None, or list with same length as arg_names, giving a str description of each argument
        n_pos_args: if n_pos_args is None, the number of positional arguments is defined by comparing the length
            of default_args and arg_names. If n_pos_args is manually specified we allow
            default_args to be the same length as arg_names, which can be used to make the docstring a bit nicer.
    Returns:
        args: tuple of length arg_names, containing the ordered values to use for the script
    """
    # check inputs, determine number of positional/keyword arguments
    assert type(arg_names) == list or type(arg_names) == tuple
    max_pos_index = -1
    for count, arg in enumerate(arg_names):
        if type(arg) == str:
            continue
        elif arg is None:
            arg_names[count] = 'Undefined'
            max_pos_index = count
        else:
            raise ValueError('Expected None or str for variable name, received ' + str(arg))
    n_args = len(arg_names)
    if n_pos_args is None:
        if default_args is not None:
            assert type(default_args) == list or type(default_args) == tuple
            assert len(default_args) <= n_args, 'Expected at most '+str(n_args) + \
                                                ' default arguments, received '+str(len(default_args))
            n_pos_args = n_args - len(default_args)

            def default_value(ind):
                return default_args[ind - n_pos_args]
        else:
            n_pos_args = n_args

            def default_value(ind):
                return ind
    elif type(n_pos_args) == int:
        assert n_pos_args <= n_args
        if default_args is not None:
            assert type(default_args) == list or type(default_args) == tuple
            if len(default_args) == n_args:

                def default_value(ind):
                    return default_args[ind]
            elif len(default_args) == n_args - n_pos_args:

                def default_value(ind):
                    return default_args[ind - n_pos_args]
            else:
                raise ValueError('default_args must have ' + 'length '+str(n_args)+' or length '
                                 + str(n_args-n_pos_args)+', got length '+str(len(default_args)))
        else:

            def default_value(ind):
                return ind
    else:
        raise ValueError('Expected value with type int for n_pos_args, received '+str(n_pos_args))
    assert n_pos_args > max_pos_index, 'Argument at index ' + str(max_pos_index) + ' was not given a name' + \
        ', but a name is required since that argument was requested to be a keyword argument.'

    # create usage string for argparser help message
    file_str = str(os.path.basename(inspect.stack()[1].filename))
    usage_str = file_str
    for count, arg in enumerate(arg_names):
        if count < n_pos_args:
            usage_str += ' '+arg
        else:
            usage_str += ' ['+arg+']'
    usage_str += ' [-h]'
    usage_str += '\n  '+str(n_pos_args)+' required arguments, '+str(n_args-n_pos_args)+' optional ' + \
                 'arguments. Optional arguments can be given as keyword arguments.'
    usage_str += '\n  Examples:  \'python '+file_str
    pos_args_str, kw_args_str = '', ''
    for count, arg in enumerate(arg_names[:n_pos_args]):
        pos_args_str += ' '+str(default_value(count))
    if n_args-n_pos_args > 0:
        use_ind = n_pos_args+1 if n_args - n_pos_args > 1 else n_pos_args
        kw_args_str += arg_names[use_ind]+'='+str(default_value(use_ind))
        usage_str += pos_args_str+' '+kw_args_str+'\', or \'python '+file_str+pos_args_str+'\'.'
    else:
        usage_str += pos_args_str+'\'.'
    if description_str is not None:
        assert type(description_str) == str

    # make arg parser
    parser = argparse.ArgumentParser(usage=usage_str, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=description_str)
    parser._positionals.title = 'Args'
    if arg_descriptions is None:
        for arg in arg_names:
            parser.add_argument(arg)
    else:
        assert type(arg_descriptions) == list or type(arg_descriptions) == tuple
        assert len(arg_descriptions) == n_args
        n_no_default = n_args - len(default_args) if default_args is not None else n_args
        for count, arg in enumerate(arg_names):
            help_str = arg_descriptions[count]
            help_str = help_str if help_str[-1] == '.' else help_str+'.'
            if count < n_no_default:
                pass
            elif count < n_pos_args:
                help_str += ' Required. Example value: '+str(default_value(count))
            else:
                help_str += ' Optional. Default value: '+str(default_value(count))
            parser.add_argument(arg, help=help_str)

    # manually populate the args for parse_args, by getting all arguments from sys.argv
    args = [None]*n_args
    is_pos_arg = [False] * n_args
    arg_name_to_ind = dict(zip(arg_names, range(len(arg_names))))
    kwargs = {}
    seen_kwarg = False
    flags = []
    count = 0

    for arg in sys.argv[1:]:
        if arg[0] == '-':  # is a flag, e.g. -help
            flags.append(arg)
            continue

        assert count < n_args, 'Too many arguments given. The arguments \''+str(tuple(args))+'\' were all set ' + \
            ' and an additional argument '+arg+' was given.'

        cur_split = arg.split('=', 1)
        if len(cur_split) > 1:  # passed a keyword argument
            arg_name, arg_value = cur_split
            arg_value = arg_value[:-1] if arg_value[-1] == ',' else arg_value

            # check if kwarg is valid
            assert arg_name in arg_name_to_ind, 'Received keyword argument \''+arg+'\' but \''+arg_name+'\' is ' + \
                'not a valid argument name. Should be one of: '+str(arg_names)
            if arg_name in kwargs:
                raise ValueError('Received multiple keyword arguments for \''+arg_name+'\'')
            if is_pos_arg[arg_name_to_ind[arg_name]]:
                raise ValueError('Argument \''+arg_name+'\' was already set as \''+str(args[arg_name_to_ind[arg_name]])
                                 + '\', but received an additional keyword argument '+arg)
            if not seen_kwarg:
                assert count >= n_pos_args, 'Expected a value for positional argument \''+arg_names[count] + \
                                            '\' but received keyword argument \''+arg+'\''

            # update keyword argument
            kwargs[arg_name] = arg_value
            seen_kwarg = True
        else:  # positional argument
            arg = arg[:-1] if arg[-1] == ',' else arg
            assert not seen_kwarg, 'Received positional argument \''+arg+'\' after a keyword argument was given. Any'\
                + ' positional arguments be given before keyword arguments.\nThis error can also be caused by missing'\
                + ' " " around a dict/list.'
            # update positional argument
            args[count] = arg
            is_pos_arg[count] = True
        count += 1

    # verify that all required arguments were given, and give args to argparser
    assert sum(is_pos_arg[:n_pos_args]) == n_pos_args, 'Missing positional arguments. Expected values for \'' + \
        '\', \''.join(arg_names[:n_pos_args])+'\', but received '+', '.join(map(str, args[:n_pos_args]))
    is_default_arg = []
    for count, arg_name in enumerate(arg_names):
        if is_pos_arg[count]:  # positional argument
            is_default_arg.append(False)
        elif arg_name in kwargs:  # keyword argument
            args[count] = kwargs[arg_name]
            is_default_arg.append(False)
        else:  # optional argument taking default value
            args[count] = str('unused')
            is_default_arg.append(True)

    args.extend(flags)
    args_dict = vars(parser.parse_args(args))
    args = [None]*n_args
    for count, arg_name in enumerate(arg_names):
        if is_default_arg[count]:
            args[count] = default_value(count)
        else:
            try:
                ast.literal_eval(args_dict[arg_name])
            except ValueError:
                print('Could not evaluate input. If due to \'malformed node or string\', this error is probably '
                      'caused by missing dashes (missing \' \' and/or " ") on the input.', file=sys.stderr)
                raise
            args[count] = ast.literal_eval(args_dict[arg_name])
    return args


def bayes_opt(f, pbounds, n_iter=100, init_points=0, init_guesses=None, save_name=None, prev_opt_name=None):
    """Wrapper for bayes_opt package (pip install bayesian-optimization).

    Args:
        f: function to maximize. It has a call signature f(**args) and the 'args' are passed as a dict, such that the
            index i argument has the key 'i'. Arguments should all be scalar floats. f returns a scalar float.
        pbounds: list of tuples, where each tuple is the upper/lower bounds for the parameter with that index.
        n_iter: int number of iterations to perform (not counting guesses/initialization)
        init_points: int number of initial random points
        init_guesses: if not None, list of lists, where each list represents a set of parameters (list of floats).
        save_name: if not None, filepath to save an optimization log, and the result of the optimization.
            The 'save_name.json' file can be used internally by bayes_opt. the 'save_name_res.pkl' is a list
            of dicts, where each dict has keys 'target' (the value returned by function f) and 'params' (giving the
            parameters used to evaluate f)
        prev_opt_name: if not None, can be the save_name from a previous run, and the log will be loaded, which
            essentially has the effect of restarting the optimizer.
    Returns:
        optimizer: the BayesianOptimization object
    """
    def make_dict(out):
        return {str(count): i for count, i in enumerate(out)}

    def make_opt_str(opt):
        evals = (opt.max['target'], opt.res[-1]['target'])
        out = 'Best: {:.1f}, Last: {:.1f}. '.format(*evals)
        out += 'Best params: (' + ', '.join(['{:.1f}'.format(val) for val in opt.max['params'].values()]) + ')' + \
            '. Last: (' + ', '.join(['{:.1f}'.format(val) for val in opt.res[-1]['params'].values()]) + ')'
        return out

    optimizer = bo.BayesianOptimization(f=f, pbounds=make_dict(pbounds), allow_duplicate_points=True, verbose=0)
    if prev_opt_name is not None:
        if init_points > 0 or init_guesses is not None:
            print('Warning: requested to load previous optimizer, but also gave initial guesses')
        bo.util.load_logs(optimizer, logs=[prev_opt_name+'.json'])
    if save_name is not None:
        logger = bo.JSONLogger(path=save_name)
        optimizer.subscribe(bo.Events.OPTIMIZATION_STEP, logger)
    n_init_guesses = len(init_guesses) if init_guesses is not None else 0
    total_iters = n_iter+init_points + n_init_guesses
    pbar = tqdm.tqdm(range(total_iters), total=total_iters, leave=True, position=0, dynamic_ncols=True)
    pbar.set_description('Params tested')

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
