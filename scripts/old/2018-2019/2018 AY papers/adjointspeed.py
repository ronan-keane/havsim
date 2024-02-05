# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:26:58 2018

@author: Pc

TEST SPEED OF ADJOINT METHOD 
ASSOCAITED PKL FILES SPEEDTEST
"""
from calibration import * 
#tests the speed of the adjoint method compared to finite differences for evaluating the gradient. 

def speed_test(data):
    entries = 15
    repeat = 10 #repeat each thing so we get a more accurate CPU time
    adjtimes = np.zeros(entries*repeat)
    fintimes = np.zeros(entries*repeat)
    objtimes = np.zeros(entries*repeat)
    spsatimes = np.zeros(entries*repeat) #to do 
    spsatimes2 = np.zeros(entries*repeat)
    
#    model = IDM_b3
#    modeladjsys = IDMadjsys_b3
#    modeladj = IDMadj_b3
    model = OVM
    modeladjsys = OVMadjsys
    modeladj = OVMadj
        
    meas, platooninfo, platoonlist = makeplatoonlist(data, entries)        
    sim = copy.deepcopy(meas) #simulation is initialized to be the same as the measurements
    pguess = [16.8*3.3,.086/3.3, 1.545, 2, .175 ] #this is what we are using for the initial guess for OVM; it comes from the bando et al paper 'phenomological study ...'
#    pguess = [40,1,1,3,10,25] #this is what we're using as initial guess for IDM 
    
    #get a platoon of suitable size 
    mylen = len(platoonlist)
    count = 0
    
    bestplatoon = platoonlist[17]
    bestlen = len(bestplatoon)-1
    curlen = bestlen
    while count < mylen and curlen != entries:
        curplatoon = platoonlist[count]
        curlen = len(curplatoon)-1 #minus 1 because first entry is an empty set 
        if curlen > bestlen:  #update if needed
            bestplatoon = curplatoon
            bestlen = curlen
        count += 1
    curplatoon = bestplatoon
    
    for n in range(entries):
        curplatoon1 = curplatoon[:n+2]
        
        #curplatoon = [[], 1014]
        myn = len(curplatoon1[1:])
        leadinfo, folinfo,rinfo = makeleadfolinfo(curplatoon1, platooninfo,meas) 
        p = np.tile(pguess, myn)
        
        for j in range(repeat):
            start = time.time()
            obj = platoonobjfn_obj(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon1, leadinfo, folinfo,rinfo)
            end = time.time()
            objtime = end-start #note that the obj is supposed to be around 500k for the initial platoon with 5 vehicles, initial guess for ovm 
            objtimes[n*10+j] = objtime
        
        for j in range(repeat):
            start = time.time()
            adjder = platoonobjfn_der(p,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon1, leadinfo, folinfo,rinfo)
            end = time.time()
            adjdertime = end-start
            adjtimes[n*10+j] = adjdertime
        
        for j in range(repeat):
            start = time.time()
            finder = sc.approx_fprime(p,platoonobjfn_obj,1e-8,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon1, leadinfo, folinfo,rinfo)
            end = time.time()
            findertime = end-start
            fintimes[n*10+j] = findertime
            
        for j in range(repeat):
            start = time.time()
            spsader = SPSA_grad(p,platoonobjfn_obj,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon1, leadinfo, folinfo,rinfo)
            end = time.time()
            spsatime = end-start
            spsatimes[n*10+j] = spsatime
            
        for j in range(repeat):
            start = time.time()
            spsader = SPSA_grad(p,platoonobjfn_obj,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon1, leadinfo, folinfo,rinfo,q=3)
            end = time.time()
            spsatime = end-start
            spsatimes2[n*10+j] = spsatime
            
        
            
    objtimesf = np.zeros(entries)
    adjtimesf = np.zeros(entries)
    fintimesf = np.zeros(entries)
    spsatimesf = np.zeros(entries)
    spsatimesf2 = np.zeros(entries)
    for i in range(entries):
        objtimesf[i] = np.mean(objtimes[repeat*i:repeat*(i+1)])
        fintimesf[i] = np.mean(fintimes[repeat*i:repeat*(i+1)])
        adjtimesf[i] = np.mean(adjtimes[repeat*i:repeat*(i+1)])
        spsatimesf[i] = np.mean(spsatimes[repeat*i:repeat*(i+1)])
        spsatimesf2[i] = np.mean(spsatimes2[repeat*i:repeat*(i+1)])
        
    return objtimes, adjtimes, fintimes, spsatimes, spsatimes2, objtimesf, adjtimesf, fintimesf, spsatimesf, spsatimesf2
    
#objtimes, adjtimes, fintimes, spsatimes, spsatimes2, objtimesf, adjtimesf, fintimesf, spsatimesf, spsatimesf2 = speed_test(data)

#with open('speedtest.pkl','wb') as f:
#    pickle.dump([objtimes, adjtimes, fintimes, spsatimes, spsatimes2, objtimesf, adjtimesf, fintimesf, spsatimesf, spsatimesf2],f)
    
with open('speedtest.pkl','rb') as f:
    objtimes, adjtimes, fintimes, spsatimes, spsatimes2, objtimesf, adjtimesf, fintimesf, spsatimesf, spsatimesf2 = pickle.load(f)
    
#%%
#plot the results 
#note that we start at 2 vehicles ( 10 parameters) because 1 vehicle (5 parameters) 
import matplotlib.pyplot as plt 
entries = 15
plt.rc('font', size=10)
plt.figure(figsize = (16,7))
plt.subplot(1,2,1)
x = range(5,5*(entries+1),5)
plt.plot(x,adjtimesf,'x',x,fintimesf,'o',x,spsatimesf,'*',x,spsatimesf2,'>')
plt.xlabel('number of parameters')
plt.ylabel('time to calculate gradient (seconds)')
plt.legend(['Adjoint Method', 'Finite Differences', 'SPSA', 'SPSA-3'])
#plt.title('Time to calculate the gradient')

plt.subplot(1,2,2)
x = range(5,5*(entries+1),5)
plt.plot(x,np.divide(adjtimesf,objtimesf),'x',x,np.divide(fintimesf,objtimesf),'o',x,np.divide(spsatimesf,objtimesf),'*',x,np.divide(spsatimesf2,objtimesf),'>')
plt.xlabel('number of parameters')
plt.ylabel('equivalent number of obj. fn. evaluations')
plt.legend(['Adjoint Method', 'Finite Differences', 'SPSA', 'SPSA-3'])
#plt.title('Time to calculate gradient relative to calculating objective')
plt.savefig('speed.png',dpi=200)

#would be nice to have a table showing speed for low values of parameters. 
