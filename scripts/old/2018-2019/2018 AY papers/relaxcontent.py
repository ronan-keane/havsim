# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 05:06:10 2019

@author: rlk268
"""
from calibration import * 
#note first couple functions were run with older versions of calibrate_* functions so now have incorrect syntax. those were for the OVM. 
#also, some stuff is in postercontent file. 

def noLC(platooninfo,meas):
    
    out3 = []
    rmse3 = []
    
    noLClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])==1:
            noLClist.append([[],i])
            
    
    p = [10*3.3,.086/3.3, 1.545, 2, .175 ]
    mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] 
    
    sim = copy.deepcopy(meas)
    for i in noLClist: 
        curplatoon = i
        leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,meas)
        
        bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,mybounds)
        currmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon)
        out3.append(bfgs)
        rmse3.append(currmse)
    
        sim[curplatoon[1]] = meas[curplatoon[1]].copy()
    
    return noLClist,out3,rmse3





def relax(platooninfo,meas):
    
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
            
    p = [10*3.3,.086/3.3, 1.545, 2, .175, 5 ]
    mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,45)]   
    out = [] #adjoint relaxation positive relaxation constants only; multiple relaxation amounts stack, no merging
    rmse = []
    out2 = []
    rmse2 = []
    
    sim = copy.deepcopy(meas)
    for i in LClist: 
        curplatoon = i
        leadinfo, folinfo, rinfo = makeleadfolinfo_r(curplatoon, platooninfo,meas,False)
        
        bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo,True, 6),0,mybounds)
        currmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon)
        out.append(bfgs)
        rmse.append(currmse)
        
        sim[curplatoon[1]] = meas[curplatoon[1]].copy()
        
    p = [10*3.3,.086/3.3, 1.545, 2, .175 ]
    mybounds = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)] 
    
    sim = copy.deepcopy(meas)
    for i in LClist: 
        curplatoon = i
        leadinfo, folinfo, rinfo = makeleadfolinfo(curplatoon, platooninfo,meas)
        
        bfgs = sc.fmin_l_bfgs_b(platoonobjfn_objder,p,None,(OVM, OVMadjsys, OVMadj, meas, sim, platooninfo, curplatoon, leadinfo, folinfo,rinfo),0,mybounds)
        currmse = convert_to_rmse(bfgs[1],platooninfo,curplatoon)
        out2.append(bfgs)
        rmse2.append(currmse)
    
        sim[curplatoon[1]] = meas[curplatoon[1]].copy()
        
    
    return LClist, out,out2,rmse,rmse2

#LClist, out,out2,rmse,rmse2 = relax(platooninfo,meas)

#noLClist,out3,rmse3 = noLC(platooninfo,meas)

#with open('relax_posr.pkl','wb') as f:
#    pickle.dump([LClist,noLClist,out,out2,out3,rmse,rmse2,rmse3], f) #save data 

#with open('relax_posr.pkl', 'rb') as f:    #this is the most up to date one out of these three #this is used for something somewhere
#    LClist,noLClist,out,out2,out3,rmse,rmse2,rmse3 = pickle.load(f) #load data
    
#with open('relax_pos.pkl', 'rb') as f:
#    LClist,out,out2,rmse,rmse2 = pickle.load(f) #load data 
        
        
        
#####new content attempt##########
    
def LCtest(platooninfo,meas):
    
    noLClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])==1:
            noLClist.append([[],i])
            
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_r = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds_r = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    
    noLC = calibrate_bfgs(meas,platooninfo,noLClist,makeleadfolinfo,plist_nor,bounds_nor,7.5)
    
    LC_nor = calibrate_bfgs(meas,platooninfo,LClist,makeleadfolinfo,plist_nor,bounds_nor,7.5)
    
    GA_posr = calibrate_rGA(meas,platooninfo,LClist,makeleadfolinfo_r,bounds_r)
    
    LC_posr = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r,plist_r,bounds_r,7.5)
    
    LC_negr = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r2,plist_r,bounds_r,7.5)
    
    LC_r = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r3,plist_r,bounds_r,7.5)
    
    return noLC,LC_nor,GA_posr,LC_posr,LC_negr,LC_r
    
#noLC,LC_nor,GA_posr,LC_posr,LC_negr,LC_r = LCtest(platooninfo,meas)

#with open('LCtest.pkl','wb') as f:
#    pickle.dump([noLC,LC_nor,GA_posr,LC_posr,LC_negr,LC_r], f)
    
with open('LCtest.pkl','rb') as f:
    noLC,LC_nor,GA_posr,LC_posr,LC_negr,LC_r = pickle.load(f)
    
def LCtest2(platooninfo,meas):
            
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
    
    plist_r = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds_r = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    
    LC_posr = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r,plist_r,bounds_r,0)
    
    LC_negr = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r2,plist_r,bounds_r,0)
    
    LC_r = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r3,plist_r,bounds_r,0)
    
    return LC_posr,LC_negr,LC_r

#LC_posr2,LC_negr2,LC_r2 = LCtest2(platooninfo,meas)
#
#with open('LCtest2.pkl','wb') as f:
#    pickle.dump([LC_posr2,LC_negr2,LC_r2], f)
    
with open('LCtest2.pkl','rb') as f:
    LC_posr2,LC_negr2,LC_r2 = pickle.load(f)

def LCtest3(platooninfo,meas):
            
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
    
    plist_r = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds_r = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    
    LC_posr = calibrate_rbfgs(meas,platooninfo,LClist,makeleadfolinfo_r,plist_r,bounds_r,float('inf'))
    
    return LC_posr

#LC_posr3 = LCtest3(platooninfo,meas)

#with open('LCtest3.pkl','wb') as f:
#    pickle.dump([LC_posr3],f)
    
with open('LCtest3.pkl','rb') as f:
    LC_posr3 = pickle.load(f)
    
def LCtest4(platooninfo,meas):
            
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_r = [[10*3.3,.086/3.3, 1.545, 2, .175, 5,5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60,60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60,60 ]]
    bounds_r = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75),(.1,75)]
    
    LC_nor = calibrate_bfgs(meas,platooninfo,LClist,makeleadfolinfo,plist_nor,bounds_nor,0)
    
    LC_2r = calibrate_rbfgs2(meas,platooninfo,LClist,makeleadfolinfo_r3,plist_r,bounds_r,0)
    
    return LC_nor, LC_2r

#LC_nor2, LC_2r = LCtest4(platooninfo,meas)

#with open('LCtest4.pkl','wb') as f:
#    pickle.dump([LC_nor2, LC_2r],f)
    
with open('LCtest4.pkl','rb') as f:
    LC_nor2, LC_2r = pickle.load(f)
    
def LCtest5(platooninfo,meas):
    
    
    sim = copy.deepcopy(meas)
    mergelist = []
    merge_from_lane = 7 
    merge_lane = 6
    for i in meas.keys():
        curveh = i
        t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
        lanelist = np.unique(sim[curveh][:t_n-t_nstar,7])
        if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][t_n-t_nstar,7]==merge_lane:
            mergelist.append([[],i])
    
    plist_nor = [[10*3.3,.086/3.3, 1.545, 2, .175],[10*3.3,.086/3.3, 1.545, .5, 1.5 ],[10*3.3,.086/3.3, 1, .2, .175]]
    bounds_nor = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3)]
    
    plist_r = [[10*3.3,.086/3.3, 1.545, 2, .175, 5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60 ]]
    bounds_r = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75)]
    
    plist_2r = [[10*3.3,.086/3.3, 1.545, 2, .175, 5,5 ],[20*3.3,.086/3.3/2, 1.545, .5, .175, 60,60 ],[10*3.3,.086/3.3/2, .5, .5, .175, 60,60 ]]
    bounds_2r = [(20,120),(.001,.1),(.1,2),(.1,5),(0,3), (.1,75),(.1,75)]
    
    merge_nor = calibrate_bfgs(meas,platooninfo,mergelist,makeleadfolinfo,plist_nor,bounds_nor,0)
    
    merge_r = calibrate_rbfgs(meas,platooninfo,mergelist,makeleadfolinfo_r4,plist_r,bounds_r,0)
    
    merge_2r = calibrate_rbfgs2(meas,platooninfo,mergelist,makeleadfolinfo_r4,plist_2r,bounds_2r,0)
    
    #also want to do merger tests for any vehicles that merge using the normal rule
    
    mergeLClist = [] #this is going to be all vehicles that merge according to the normal LC rule, but we don't apply the LC there. 
    #basically this will give a baseline for having no merging rule for vehicles 
    for i in meas.keys():
        unused, unused, rinfo = makeleadfolinfo_r3([[],i],platooninfo,meas)
        unused,unused,rinfo2 = makeleadfolinfo_r6([[],i],platooninfo,meas)
        if len(rinfo[0])>0:
            if len(rinfo2[0])==0:
                mergeLClist.append([[],i])
            elif rinfo[0][0] != rinfo2[0][0]:
                mergeLClist.append([[],i])
                
    mergeLC_r = calibrate_rbfgs(meas,platooninfo,mergeLClist,makeleadfolinfo_r6,plist_r,bounds_r,0)
    
    mergeLC_2r = calibrate_rbfgs2(meas,platooninfo,mergeLClist,makeleadfolinfo_r6,plist_2r,bounds_2r,0)
            
    return merge_nor, merge_r, merge_2r,mergeLC_r, mergeLC_2r

#merge_nor, merge_r, merge_2r,mergeLC_r, mergeLC_2r = LCtest5(platooninfo,meas)

#with open('LCtest5.pkl','wb') as f:
#    pickle.dump([merge_nor, merge_r, merge_2r,mergeLC_r, mergeLC_2r],f)
    
with open('LCtest5.pkl','rb') as f:
    merge_nor, merge_r, merge_2r,mergeLC_r, mergeLC_2r = pickle.load(f)
    
    
def LCtest6(platooninfo,meas):
    
    noLClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])==1:
            noLClist.append([[],i])
            
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
    plist_nor = [[40,1,1,3,10],[60,1,1,3,10],[80,1,15,1,1]]
    bounds_nor = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20)]
    
    plist_r = [[40,1,1,3,10,25],[60,1,1,3,10,5],[80,1,15,1,1,5]]
    bounds_r = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75)]
    
    plist_2r = [[40,1,1,3,10,25,25],[60,1,1,3,10,5,5],[80,1,15,1,1,5,5]]
    bounds_2r = [(20,120),(.1,5),(.1,35),(.1,20),(.1,20),(.1,75),(.1,75)]
    
    noLC = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,noLClist,makeleadfolinfo,platoonobjfn_objder,None,
                          IDM_b3,IDMadjsys_b3,IDMadj_b3,False,5,cutoff = 0,delay = False,dim=2)
    
    LC_nor = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,LClist,makeleadfolinfo,platoonobjfn_objder,None,
                          IDM_b3,IDMadjsys_b3,IDMadj_b3,False,5,cutoff = 0,delay = False,dim=2)
    
    LC_posr = calibrate_bfgs(plist_r,bounds_r,meas,platooninfo,LClist,makeleadfolinfo_r,platoonobjfn_objder,None,
                          IDM_b3,IDMadjsys_b3,IDMadj_b3,True,6,cutoff = 0,delay = False,dim=2)
    
    LC_negr = calibrate_bfgs(plist_r,bounds_r,meas,platooninfo,LClist,makeleadfolinfo_r2,platoonobjfn_objder,None,
                          IDM_b3,IDMadjsys_b3,IDMadj_b3,True,6,cutoff = 0,delay = False,dim=2)
    
    LC_r = calibrate_bfgs(plist_r,bounds_r,meas,platooninfo,LClist,makeleadfolinfo_r3,platoonobjfn_objder,None,
                          IDM_b3,IDMadjsys_b3,IDMadj_b3,True,6,cutoff = 0,delay = False,dim=2)
    
    LC_2r = calibrate_bfgs(plist_2r,bounds_2r,meas,platooninfo,LClist,makeleadfolinfo_r3,platoonobjfn_objder2,None,
                          IDM_b3,IDMadjsys_b3,IDMadj2_b3,True,7,cutoff = 0,delay = False,dim=2)
    
    return noLC,LC_nor,LC_posr,LC_negr,LC_r, LC_2r

#inoLC, iLC_nor, iLC_posr, iLC_negr, iLC_r, iLC_2r = LCtest6(platooninfo,meas)

#with open('LCtest6.pkl','wb') as f:
#    pickle.dump([inoLC, iLC_nor, iLC_posr, iLC_negr, iLC_r, iLC_2r],f)
    
with open('LCtest6.pkl','rb') as f:
    inoLC, iLC_nor, iLC_posr, iLC_negr, iLC_r, iLC_2r = pickle.load(f)
    
    
def LCtest7(platooninfo,meas):
    
    noLClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])==1:
            noLClist.append([[],i])
            
    LClist = []
    for i in meas.keys():
        if len(platooninfo[i][4])>1:
            LClist.append([[],i])
            
    plist_nor = [[1.5,60],[2.5,100],[2,150]]
    bounds_nor = [(0,5),(5,200)]
    
    plist_r = [[1.5,60,5],[2.5,100,60],[2,150,60]]
    bounds_r = [(0,5),(5,200),(.1,75)]
    
    plist_2r = [[1.5,60,5,5],[2.5,100,60,60],[2,150,60,60]]
    bounds_2r = [(0,5),(5,200),(.1,75),(.1,75)]
    
    noLC = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,noLClist,makeleadfolinfo,TTobjfn_obj,TTobjfn_fder,
                          None,None,None,False,2,cutoff = 0,delay = True,dim=1)
    
    LC_nor = calibrate_bfgs(plist_nor,bounds_nor,meas,platooninfo,LClist,makeleadfolinfo,TTobjfn_obj,TTobjfn_fder,
                          None,None,None,False,2,cutoff = 0,delay = True,dim=1)
    
    LC_posr = calibrate_bfgs(plist_r,bounds_r,meas,platooninfo,LClist,makeleadfolinfo_r,TTobjfn_obj,TTobjfn_fder,
                          None,None,None,True,3,cutoff = 0,delay = True,dim=1)
    
    LC_negr = calibrate_bfgs(plist_r,bounds_r,meas,platooninfo,LClist,makeleadfolinfo_r2,TTobjfn_obj,TTobjfn_fder,
                          None,None,None,True,3,cutoff = 0,delay = True,dim=1)
    
    LC_r = calibrate_bfgs(plist_r,bounds_r,meas,platooninfo,LClist,makeleadfolinfo_r3,TTobjfn_obj,TTobjfn_fder,
                          None,None,None,True,3,cutoff = 0,delay = True,dim=1)
    
    LC_2r = calibrate_bfgs(plist_2r,bounds_2r,meas,platooninfo,LClist,makeleadfolinfo_r3,TTobjfn_obj,TTobjfn_fder,
                          None,None,None,True,4,True,cutoff = 0,delay = True,dim=1)
    
    return noLC,LC_nor,LC_posr,LC_negr,LC_r, LC_2r

#nnoLC, nLC_nor, nLC_posr, nLC_negr, nLC_r, nLC_2r = LCtest7(platooninfo,meas)

#with open('LCtest7.pkl','wb') as f:
#    pickle.dump([nnoLC, nLC_nor, nLC_posr, nLC_negr, nLC_r, nLC_2r],f)
    
with open('LCtest7.pkl','rb') as f:
    nnoLC, nLC_nor, nLC_posr, nLC_negr, nLC_r, nLC_2r = pickle.load(f)
#

