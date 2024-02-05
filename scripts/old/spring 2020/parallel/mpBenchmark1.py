
"""
@author: rlk268@cornell.edu
"""

import numpy as np
import time
# import multiprocessing
import pathos.multiprocessing as pm
import ray

class toyVehicle:
    def __init__(self):
        self.rand = np.random.rand()
        self.res = None

    def test(self, a, b):
        temp = a**(b*self.rand+a**b+(a+b)/b)
        self.res = temp**(b*self.rand+b**a)

@ray.remote
class toyVehicleRemote:
    def __init__(self):
        self.rand = np.random.rand()
        self.res = None

    def test(self, a, b):
        temp = a**(b*self.rand+a**b+(a+b)/b)
        self.res = temp**(b*self.rand+b**a)
        
    def read(self):
        return self.res
    
@ray.remote
def RemoteHelper(vehicles,a,b):
    [vehicle.test.remote(a,b) for vehicle in vehicles]
    futures = [vehicle.read.remote() for vehicle in vehicles]
    res = sum(ray.get(futures))
    return res

n = 16
cpus = 8

def ptest(n):
    vehicles = [toyVehicle() for i in range(n)]
    a = 2
    b = 1.5
    start = time.time()
    out = [vehicle.test(a,b) for vehicle in vehicles]
    res = sum([vehicle.res for vehicle in vehicles])
    end = time.time()
    print(str(end-start)+' to find result '+str(res))
    return res

out = ptest(n)
# 
# vehicles = [toyVehicleRemote.remote() for i in range(n)]
# a,b = 2, 1.5
# start = time.time()
# [vehicle.test.remote(a,b) for vehicle in vehicles]
# futures = [vehicle.read.remote() for vehicle in vehicles]
# res = sum(ray.get(futures))
# print(str(time.time()- start) + ' to find result '+str(res))


vehicles = [toyVehicleRemote.remote() for i in range(n)]
a,b = 2, 1.5
chunksize = int(n/cpus)
start = time.time()
plist = [RemoteHelper(vehicles[i*chunksize:(i+1)*chunksize], a, b) for i in range(cpus)]
ray.get(plist)
print(str(time.time()-start)+' to find result '+str(sum(plist)))





# def wrapper(args):
#     veh, a, b = args
#     veh.test(a,b)
#     return veh


# def ptest2():
#     a,b = 2, 1.5
#     vehicles = [toyVehicle() for i in range(10000)]
#     start = time.time()
#     starargs = ((veh,a,b) for veh in vehicles)

#     with pm.Pool() as pool:
#         vehicles = pool.map(wrapper, starargs)

#     res = sum(veh.res for veh in vehicles)
#     print(str(time.time()-start)+' to find result '+str(res))
#     return res


# if __name__ == '__main__':
#     out = ptest2()

