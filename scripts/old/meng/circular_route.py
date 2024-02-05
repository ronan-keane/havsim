"""Experimental/old implementation of a delay model."""
from typing import List, Dict
import heapq
import random
import math
import collections
from havsim.plotting import plotformat, platoonplot
import matplotlib.pyplot as plt


# Copied from havsim/simulation/models.py with a little change
def zhangkim(parameters: List[float],
             delayed_headway: float,
             delayed_lead_speed: float):
    # refer to car following theory for multiphase vehicular traffic flow by zhang and kim, model c
    # delay is p[0], then in order, parameters are S0, S1, v_f, h_1
    # recall reg = 0 is reserved for shifted end
    # s is the delayed headway, i.e. lead(t - tau) - lead_len - x(t - tau)
    # leadv is the delayed velocity

    if delayed_headway >= parameters[2]:
        out = parameters[3]

    elif delayed_headway < parameters[1]:
        out = delayed_headway / parameters[4]

    else:
        # the original version uses delayed_lead_speed[1]
        if delayed_lead_speed >= parameters[3]:
            out = parameters[3]
        else:
            out = delayed_headway / parameters[4]

    return out


def gipps(p: List[float],
          delayed_headway: float,
          delayed_lead_speed: float,
          delayed_self_speed: float):
    # p = parameters - [\tau, \dot S_f, \ddot S_f, \Delta S^0, \hat b, b_f]
    # original formulation of gipps
    # 6 parameters version with \alpha \beta \gamma \theta taken as their nominal values
    # this assumes that \hat b and b_f are given as positive
    acc = delayed_self_speed + 2.5 * p[2] * p[0] * (1 - delayed_self_speed / p[1]) * (
            .025 + delayed_self_speed / p[1]) ** .5
    dec = -p[5] * (p[0]) + ((p[5] ** 2 * p[0] ** 2) +
                            p[5] * (2 * (delayed_headway - p[3]) - delayed_self_speed * p[0] + (
                    delayed_lead_speed ** 2) / p[4])) ** .5
    return min(acc, dec)


class VehicleStateSnapshot:
    def __init__(self,
                 pos: float,
                 speed: float,
                 headway: float):
        self.pos = pos
        self.speed = speed
        self.headway = headway


class RoadInfo:
    def __init__(self, len):
        self.len = len


class Vehicle:
    def __init__(self,
                 id: int,
                 response_time: float,
                 pos: float,
                 speed: float,
                 len: float,
                 road_info: RoadInfo,
                 lead_vehicle=None,  # reference to lead vehicle
                 ):
        self.id = id
        self.response_time = response_time
        self.lead_vehicle = lead_vehicle
        self.speed_history = [speed]
        self.pos_history = [pos % road_info.len]
        self.len = len
        self.road_info = road_info

    def get_headway(self, timestamp: float):
        pos = self.get_pos(timestamp)
        lead_pos = self.lead_vehicle.get_pos(timestamp)
        return (lead_pos - pos - self.lead_vehicle.len) % self.road_info.len  # for circular road

    def get_speed(self, timestamp: float):
        if timestamp > self.get_timestamp_max():
            raise Exception("Speed not available for timestamp specified.")
        if timestamp == self.get_timestamp_max():
            return self.speed_history[-1]
        ind = math.floor(timestamp / self.response_time)
        v1 = self.speed_history[ind]
        v2 = self.speed_history[ind + 1]
        return v1 + (v2 - v1) * (timestamp - ind * self.response_time) / self.response_time

    def get_pos(self, timestamp: float):
        if timestamp > self.get_timestamp_max():
            raise Exception("Position not available for timestamp specified.")
        if timestamp == self.get_timestamp_max():
            return self.pos_history[-1]
        ind = math.floor(timestamp / self.response_time)
        v1 = self.speed_history[ind]
        v2 = self.get_speed(timestamp)
        v = (v1 + v2) / 2
        t1 = self.response_time * ind
        return (self.pos_history[ind] + (timestamp - t1) * v) % self.road_info.len

    def respond(self):
        # Vehicle will respond every self.response_time
        last_pos = self.pos_history[-1]
        last_speed = self.speed_history[-1]
        delayed_timestamp = self.get_timestamp_max()
        new_speed = zhangkim([self.response_time, 30, 45, 30, 1.5], self.get_headway(delayed_timestamp),
                             self.lead_vehicle.get_speed(delayed_timestamp))

        self.pos_history.append((last_pos + (last_speed + new_speed) / 2 * self.response_time) % self.road_info.len)
        self.speed_history.append(new_speed)

    def get_timestamp_max(self):
        # This is also the reference time (delayed time) to respond
        return self.response_time * (len(self.speed_history) - 1)


class Task:
    def __init__(self, vehicle: Vehicle):
        self.vehicle = vehicle

    def __lt__(self, other):
        return self.vehicle.get_timestamp_max() < other.vehicle.get_timestamp_max()


def simulate_circular_road(vehicles_init: List[Vehicle],
                           T: float,
                           dt: float,
                           ) -> Dict[int, List[List[float]]]:
    tasks = [Task(vehicle) for vehicle in vehicles_init]
    finished = []
    heapq.heapify(tasks)
    while tasks:
        task = heapq.heappop(tasks)
        vehicle = task.vehicle
        if vehicle.get_timestamp_max() >= T:
            finished.append(vehicle)
        else:
            vehicle.respond()
            heapq.heappush(tasks, task)

    ts = 0.0
    res = collections.defaultdict(list)
    while ts <= T:
        for vehicle in vehicles:
            res[vehicle.id].append([vehicle.get_pos(ts),
                                    vehicle.get_speed(ts),
                                    vehicle.get_headway(ts)])
        ts += dt

    return res


# auxinfo pass as [], roadinfo pass as {0:840} #where 840 is len of road
# sim is dictionary where keys are vehicles, values are nested lists of [position, speed, headway] (same discretization for all vehicles )
def myplot(sim, auxinfo, roadinfo, platoon=[]):
    # note to self: very hacky
    meas, platooninfo = plotformat(sim, auxinfo, roadinfo, starttimeind=0, endtimeind=math.inf, density=1)
    platoonplot(meas, None, platooninfo, platoon=platoon, lane=1, colorcode=True, speed_limit=[0, 25])
    plt.ylim(0, roadinfo[0])
    plt.show()

if __name__ == '__main__':
    road = RoadInfo(len=840)
    vehicles = [Vehicle(id=i,
                        response_time=random.random() * 0.6 + 0.6,
                        pos=20 if i == 0 else 42 * i,
                        speed=30,
                        len=2,
                        road_info=road)
                for i in range(20)]
    for i in range(20):
        vehicles[i].lead_vehicle = vehicles[(i + 1) % 20]
    res = simulate_circular_road(vehicles_init=vehicles, T=100, dt=0.25)
    myplot(res, [], {0: 840})
