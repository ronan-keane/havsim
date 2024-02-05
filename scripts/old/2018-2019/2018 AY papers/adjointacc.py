# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:31:44 2018

@author: rlk268

TEST ACCURACY OF ADJOINT METHOD 
ASSOCIATED PKL FILES ACCTEST
"""
from calibration import * 
#test accuracy of adjoint method as the gradient becomes small
#first get a starting point and ending point for a certain vehicle

def acc_test(meas):
#sim = copy.deepcopy(meas)
    model = OVM
    modeladjsys = OVMadjsys
    modeladj = OVMadj
    pguess = [10*3.3,.086/3.3, 1.545, 2, .175 ]
    
    mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] #less conservative bounds 
    
    curplatoon = [[],1013]
    #curplatoon = [[],562]
    n = len(curplatoon[1:])
    
    leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,meas) 
    #leadinfo,folinfo,rinfo = makeleadfolinfo(curplatoon,platooninfo,meas)
    p = np.tile(pguess, n)
    bounds = np.tile(mybounds,(n,1))
    
    start = time.time()
    bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,None)
    end = time.time()
    bfgstime3 = end-start
    
    #choose how many points we will take in between start and ending
#    dim = 1000
    dim = 10
    myp = np.zeros((dim,5))
    for i in range(5):
        myp[:,i] = np.linspace(pguess[i],bfgs[0][i],dim)
    
    objlist =  np.zeros(dim)
    adjlist = np.zeros((dim,5))
    finlist = np.zeros((dim,5))
    spsalist = np.zeros((dim,5)) 
    spsalist2 = np.zeros((dim,5))
    resid = np.zeros(dim)
    relresid = np.zeros(dim)
    relresid2 = np.zeros(dim)
    relresid3 = np.zeros(dim)
    finnorm = np.zeros(dim)
    
    for i in range(dim):
        curp = myp[i,:]
        obj = platoonobjfn_obj(curp,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)
        adjder = platoonobjfn_der(curp,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)
        finder = platoonobjfn_fder(curp,OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo, rinfo)
        spsader = SPSA_grad(p,platoonobjfn_obj,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo)
        spsader2 = SPSA_grad(p,platoonobjfn_obj,model, modeladjsys, modeladj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,q=3)
        
        curresid = np.linalg.norm(adjder-finder)
        resid2 = np.linalg.norm(spsader-finder)
        resid3 = np.linalg.norm(spsader2-finder)
        curfin = np.linalg.norm(finder)
        
        objlist[i] = obj
        adjlist[i,:] = adjder #unused - just for referenece
        finlist[i,:] = finder #unused
        resid[i] = curresid #unused
        spsalist[i,:] = spsader
        spsalist2[i,:] = spsader2
        
        relresid[i] = curresid/curfin
        relresid2[i] = resid2/curfin
        relresid3[i] = resid3/curfin
        finnorm[i] = curfin
        
    return objlist, relresid, finnorm, relresid2, relresid3

objlist, relresid, finnorm, relresid2, relresid3 = acc_test(meas)

#with open('acctest.pkl','wb') as f:
#    pickle.dump([objlist, relresid, finnorm, relresid2, relresid3],f)
    
#with open('acctest.pkl','rb') as f:
#    objlist, relresid, finnorm, relresid2, relresid3 = pickle.load(f)
    
        
#%% 
relreduc = (objlist[0]-objlist)/(objlist[0]-objlist[-1])
plt.figure(figsize=(16,7))
plt.subplot(1,2,1)
plt.plot(objlist,np.log10(relresid),'.',objlist,np.log10(relresid2),'*',objlist,np.log10(relresid3),'>')
plt.xlabel('objective function value')
plt.ylabel('log of gradient relative error')
plt.legend(['Adjoint','SPSA','SPSA-3'])

#plt.figure()
#plt.plot(relreduc,resid)

#plt.figure()
plt.subplot(1,2,2)
plt.plot(np.log10(finnorm),np.log10(relresid),'.',np.log10(finnorm),np.log10(relresid2),'*',np.log10(finnorm),np.log10(relresid3),'>')
plt.xlabel('log of norm of finite difference gradient' )
plt.ylabel('log of gradient relative error')
plt.legend(['Adjoint','SPSA','SPSA-3'])
plt.savefig('acc.png',dpi=200)

#not sure if plots are good for this. I think just tables are more effective. 
    
