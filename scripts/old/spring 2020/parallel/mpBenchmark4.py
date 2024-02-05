
"""
@author: rlk268@cornell.edu
"""
print('this is too simple use benchmark3 or benchmark1')

from multiprocessing import Pool
import time
import ray

def f(x):
    return x*x*x*x

@ray.remote
def remotef(x):
    return x*x*x*x

if __name__ == '__main__':
    p = Pool(5)
    l = [x for x in range(2000)]
    
    start = time.time()
    futures = [remotef.remote(x) for x in range(2000)]
    ray.get(futures)
    print('ray time is ' +str(time.time()-start))
    
    start = time.time()
    p.map(f, l)
    end = time.time()
    print('pool processing time {}'.format(end - start))
    
    start = time.time()
    map(f, l)
    end = time.time()
    print('sequential processing time {}'.format(end - start))