import havsim


def do_simulation():
    return


def evaluate_crash_rate(parameters):
    pass


def crash_confidence(crashes, n_sims, vmt_sim, z=1.96, inverse=True):
    """Calculates confidence interval of a crash rate. Assumes number of crashes per simulation is poisson distributed.

    Args:
        crashes: total number of crashes
        n_sims: total number of (identically distributed) simulations
        vmt_sim: average number of miles driven per simulation
        z: z-score corresponding to (1 - \alpha/2) percentile, where \alpha is the confidence interval.
        inverse: if True, return inverse crash rate (miles/event). Otherwise, return crash rate (event/miles)
    Returns:
        mean: average crash rate (events/miles)
        low: lower confidence interval of crash rate (events/miles)
        high: upper confidence interval of crash rate (events/miles)
    """
    mean = crashes/n_sims
    if crashes < 20:
        temp = crashes/n_sims + z**2/(2*n_sims)
        temp2 = z/2/n_sims*(4*crashes + z**2)**.5
        if inverse:
            return vmt_sim/mean, vmt_sim/(temp+temp2), vmt_sim/(temp-temp2)
        else:
            return mean/vmt_sim, (temp-temp2)/vmt_sim, (temp+temp2)/vmt_sim
    else:
        temp = crashes**.5*z/n_sims
        if inverse:
            return vmt_sim/mean, vmt_sim/(mean + temp), vmt_sim/(mean - temp)
        else:
            return mean/vmt_sim, (mean - temp)/vmt_sim, (mean + temp)/vmt_sim


if __name__ == '__main__':
    print('hi')
