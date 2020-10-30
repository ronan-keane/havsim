
"""
@author: rlk268@cornell.edu
"""

#tests a car following model by doing simple simulation on single lane
#turn off set_route_events function or else it will throw error
import numpy as np
import matplotlib.pyplot as plt
from havsim.simulation import Vehicle
from havsim.simulation import update_lane_routes
from havsim.simulation.models import IDM, IDM_eql
import sys

class lane:
    def __init__(self, speedfun, connect_left = [(0, None)], connect_right = [(0, None)]):
#        self.call_downstream = downstream_wrapper(speed_fun = speedfun, method = 'speed').__get__(self, lane)
        self.timeseries = speedfun
        self.road = {'name': None}
        self.events = []
        self.connect_left = connect_left
        self.connect_right = connect_right
    def get_headway(self, veh, lead):
        return lead.pos-veh.pos - lead.len

    def call_downstream(self, veh, timeind, dt):
        speed = self.timeseries[timeind]
        return (speed - veh.speed)/dt

    def get_connect_left(self, *args):
        return None
    def get_connect_right(self, *args):
        return None

# need a vehicle with no route model added.
class vehicle(Vehicle):
    def initialize(self, pos, spd, hd, starttime):
        """Sets the remaining attributes of the vehicle, making it able to be simulated.

        Args:
            pos: position at starttime
            spd: speed at starttime
            hd: headway at starttime
            starttime: first time index vehicle is simulated

        Returns:
            None.
        """
        # state
        self.pos = pos
        self.speed = spd
        self.hd = hd

        # memory
        self.starttime = starttime
        self.leadmem.append((self.lead, starttime))
        self.lanemem.append((self.lane, starttime))
        self.posmem.append(pos)
        self.speedmem.append(spd)

        # llane/rlane and l/r
        self.llane = self.lane.get_connect_left(pos)
        if self.llane is None:
            self.l_lc = None
        elif self.llane.roadname == self.road:
            self.l_lc = 'discretionary'
        else:
            self.l_lc = None
        self.rlane = self.lane.get_connect_right(pos)
        if self.rlane is None:
            self.r_lc = None
        elif self.rlane.roadname == self.road:
            self.r_lc = 'discretionary'
        else:
            self.r_lc = None

#number of timesteps and timestep length
simlen = 2000
dt = .25
nveh = 100 #number of vehicles
#seed initial disturbance/create 'network'
timeseries = [30 for i in range(simlen)]
timeseries[:200] = [1/10*(.1*i-10)**2 +20 for i in range(200)]
timeseries = np.asarray(timeseries) - 10
#speedfun = timeseries_wrapper(timeseries)
#curlane = lane(speedfun)
curlane = lane(timeseries)
#only parameters needed are for cf call
p =  [33.33, 1.1, 2, .9, 1.5]
initspeed = timeseries[0]
length = 2
# build simulation
vehicles = set()
curpos = 0
curroute = [None]
veh = vehicle(-1, curlane, p, None, length = length, route=curroute)
try:
    veh.initialize(curpos, initspeed, None, 0)
except:
    print('turn off route model in vehicle')
    sys.exit()


eql_hd = IDM_eql(p, initspeed)
vehicles.add(veh)
vehlead = veh
for i in range(nveh -1):
    curpos += -eql_hd - length
    veh = vehicle(i, curlane, p, None, length = length, lead = vehlead, route=curroute)
    veh.initialize(curpos, initspeed, eql_hd, 0)
    vehicles.add(veh)
    vehlead = veh

def test_cf(vehicles, simlen, dt):
    for i in range(simlen):
        for veh in vehicles:
            veh.set_cf( i, dt)

        for veh in vehicles:
            veh.update(i, dt)
        for veh in vehicles:
            if veh.lead is not None:
                veh.hd = veh.lane.get_headway(veh, veh.lead)
#%%

test_cf(vehicles, simlen, dt)
plt.figure()
for veh in vehicles:
    plt.plot(veh.posmem)




