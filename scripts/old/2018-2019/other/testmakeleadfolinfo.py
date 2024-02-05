
"""
@author: rlk268@cornell.edu
test the code after refactoring see if its broken 
"""
from havsim.calibration.helper import *

testplatoon = [381.0, 391.0, 335.0, 326.0, 334.0]

#leadinfo = makeleadinfo(testplatoon, platooninfo, meas)

#folinfo = makefolinfo(testplatoon, platooninfo, meas)

#folinfo = makefolinfo(testplatoon, platooninfo, meas,  allfollowers = False)

leadinfo, folinfo, rinfo = makeleadfolinfo(testplatoon, platooninfo,  meas, relaxtype = 'pos',mergertype = 'none')

leadinfo2, folinfo2, rinfo2 = makeleadfolinfo_r(testplatoon, platooninfo, meas)