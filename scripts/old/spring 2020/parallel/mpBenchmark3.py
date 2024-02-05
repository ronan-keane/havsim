
"""
@author: rlk268@cornell.edu
"""
import time
import multiprocessing  
import ray


def fun(args):
    a,b,n = args
    a,b = a/n,b/n
    temp = a**(b*a+a**b+(a+b)/(b+1))
    return temp**(b*a+b**a)

def wrapfun(start, stop, n):
    for i in range(start, stop):
        a, b = i/n, i/n
        temp = a**(b*a+a**b+(a+b)/(b+1))
        temp = temp**(b*a+b**a)
    return

@ray.remote
def wrapfunremote(start, stop, n):
    for i in range(start, stop):
        a, b = i/n, i/n
        temp = a**(b*a+a**b+(a+b)/(b+1))
        temp = temp**(b*a+b**a)
    return

cpu = 8
n = 500000

start = time.time()
for arg in ((i,i,n) for i in range(n)):
    fun(arg)
print(time.time()-start)

# # BENCHMARK POOL
# if __name__ == "__main__":
#     start = time.time()
#     with multiprocessing.Pool() as pool:
#         pool.map(fun, ((i,i,n) for i in range(n)))
#     print(time.time()-start)

# #benchmark process

# if __name__ == '__main__':
#     start = time.time()
#     chunk = n // cpu
#     chunks = list(range(0,n,chunk))
#     chunks.append(n)
#     plist = []
#     for i in range(cpu):
#         p = multiprocessing.Process(target = wrapfun, args = (chunks[i], chunks[i+1], n))
#         plist.append(p)
#         p.start()
#     for p in plist:
#         p.join()
#     print(time.time()-start)
    
#benchmark ray
start = time.time()
chunk = n // cpu
chunks = list(range(0,n,chunk))
chunks.append(n)
plist = [wrapfunremote.remote(chunks[i], chunks[i+1], n) for i in range(cpu)]
ray.get(plist)
print(time.time()-start)
    


