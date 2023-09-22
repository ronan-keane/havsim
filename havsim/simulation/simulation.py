"""Houses the main logic for updating simulations.

A simulation is defined by a collection of lanes/roads (a road network) and an initial
collection of vehicles. The road network defines both the network topology (i.e. how roads connect
with each other) as well as the inflow/outflow boundary conditions, which determine how
vehicles enter/leave the simulation. The inflow conditions additionally control what types of
vehicles enter the simulation. Vehicles are implemented in the Vehicle class and a road network
is made up of instances of the Lane class.
"""
from havsim.simulation.road_networks import get_headway
from havsim.simulation import update_lane_routes
from havsim.simulation import vehicle_orders
import copy

def update_net(vehicles, lc_actions, inflow_lanes, merge_lanes, vehid, timeind, dt):
    """Updates all quantities for a road network.

    The simulation logics are as follows. At the beginning of the timestep, all vehicles/states/events
    are assumed to be fully updated for the current timestep. Then, in order:
        -evaluate the longitudinal action (cf model) for all vehicles (done in Simulation.step)
        -evaluation the latitudinal action (lc model) for all vehicles (done in Simulation.step)
        -complete requested lane changes
            -move vehicle to the new lane and set its new route/lane events
            -updating all leader/follower relationships (including l/rfol) for all vehicles involved. This
            means updating vehicles in up to 4 lanes (opside, self lane, lcside, and new lcside).
            -apply relaxation to any vehicles as necessary
        -update all states and memory for all vehicles
        -update headway for all vehicles
        -update any merge anchors
        -updating lane/route events for all vehicles, including removing vehicles if they leave the network.
        -update all lfol/rfol relationships
        -update the inflow conditions for all lanes with inflow, including possible adding new vehicles,
            and generating the parameters for the next new vehicle
    After the updating is complete, all vehicles/states/events are updated to their values for the next
    timestep, so when the time index is incremented, the iteration can continue.

    Args:
        vehicles: set containing all vehicle objects being actively simulated
        lc_actions: dictionary with keys as vehicles which request lane changes in the current timestep,
            values are a string either 'l' or 'r' which indicates the side of the change
        inflow_lanes: iterable of lanes which have upstream (inflow) boundary conditions. These lanes
            must have get_inflow, increment_inflow, and new_vehicle methods
        merge_lanes: iterable of lanes which have merge anchors. These lanes must have a merge_anchors
            attribute
        vehid: int giving the unique vehicle ID for the next vehicle to be generated
        timeind: int giving the timestep of the simulation (0 indexed)
        dt: float of time unit that passes in each timestep

    Returns:
        vehid: updated int value of next vehicle ID to be added
        remove_vehicles: list of vehicles which were removed from simulation at current timestep

        Modifies vehicles in place (potentially all their non-parameter/bounds attributes, e.g. pos/spd,
        lead/fol relationships, their lane/roads, memory, etc.).
    """
    # update followers/leaders for all lane changes
    relaxvehs = []  # keeps track of vehicles which have relaxation applied
    for veh in lc_actions.keys():
        relaxvehs.append(veh.fol)

        # update leader follower relationships, lane/road
        update_lane_routes.update_veh_after_lc(lc_actions, veh, timeind)  # this is the only thing
        # which can't be done in parralel

        relaxvehs.append(veh)
        relaxvehs.append(veh.fol)

        # update a vehicle's lane events and route events for the new lane
        update_lane_routes.set_lane_events(veh)
        update_lane_routes.set_route_events(veh, timeind)

    for veh in set(relaxvehs):  # apply relaxation
        veh.set_relax(timeind, dt)

    # update all states, memory and headway
    for veh in vehicles:
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None:
            veh.hd = get_headway(veh, veh.lead)

    # update left and right followers
    # vehicle_orders.update_all_lrfol(vehicles)
    vehicle_orders.update_all_lrfol_multiple(vehicles)  # handles edge cases explicitly, slower

    # update merge_anchors
    for curlane in merge_lanes:
        update_lane_routes.update_merge_anchors(curlane, lc_actions)

    # update roads (lane events) and routes
    remove_vehicles = []
    for veh in vehicles:
        # check vehicle's lane events and route events, acting if necessary
        update_lane_routes.update_lane_events(veh, timeind, remove_vehicles)
        update_lane_routes.update_route_events(veh, timeind)
    # remove vehicles which leave
    for veh in remove_vehicles:
        vehicles.remove(veh)

    # update inflow, adding vehicles if necessary
    for curlane in inflow_lanes:
        vehid = curlane.increment_inflow(vehicles, vehid, timeind, dt)

    # for veh in vehicles:  # debugging
    #     if not veh._chk_leadfol(verbose = False):
    #         # print('-------- Report for Vehicle '+str(veh.vehid)+' at time '+str(timeind)+'--------')
    #         # veh._leadfol()
    #         veh._chk_leadfol()
    #         # raise ValueError('incorrect vehicle order')

    return vehid, remove_vehicles


class Simulation:
    """Implements a traffic microsimulation.

    Basically just a wrapper for update_net. For more information on traffic
    microsimulation refer to the full documentation which has extra details and explanation.

    Attributes:
        inflow lanes: list of all lanes which have inflow to them (i.e. all lanes which have upstream
            boundary conditions, meaning they can add vehicles to the simulation)
        merge_lanes: list of all lanes which have merge anchors
        vehicles: set of all vehicles which are in the simulation at the first time index. This is kept
            updated so that vehicles is always the set of all vehicles currently being simulated.
        prev_vehicles: set of all vehicles which have been removed from simulation. So prev_vehicles and
            vehicles are disjoint sets, and their union contains all vehicles which have been simulated.
        vehid: starting vehicle ID for the next vehicle to be added. Used for hashing vehicles.
        timeind: the current time index of the simulation (int). Updated as simulation progresses.
        dt: constant float. timestep for the simulation.
    """

    def __init__(self, vehicles=None, prev_vehicles=None, vehid=1, timeind=0, dt=.25, roads=None):
        """Inits simulation.

        Args:
            inflow_lanes: list of all Lanes which have inflow to them
            merge_lanes: list of all Lanes which have merge anchors
            vehicles: set of all Vehicles in simulation in first timestep
            prev_vehicles: list of all Vehicles which were previously removed from simulation.
            vehid: vehicle ID used for the next vehicle to be created.
            timeind: starting time index (int) for the simulation.
            dt: float for how many time units pass for each timestep. Defaults to .25.
            roads: optional, list of all the roads in the simulation, if this field is specified,
              inflow_lanes and merge_lanes will be deduced using roads.

        Returns:
            None. Note that we keep references to all vehicles through vehicles and prev_vehicles,
            a Vehicle stores its own memory.
        """
        # automatically get inflow/merge lanes
        assert isinstance(roads, list)
        self.roads = roads
        self.inflow_lanes = []
        self.merge_lanes = []
        for road in roads:
            for lane in road.lanes:
                if hasattr(lane, "get_inflow"):
                    self.inflow_lanes.append(lane)
                if hasattr(lane, "merge_anchors") and lane.merge_anchors:
                    self.merge_lanes.append(lane)
        self.init_merge_anchors = {}
        for lane in self.merge_lanes:
            self.init_merge_anchors[lane] = [anchor.copy() for anchor in lane.merge_anchors]

        self.init_vehicles = vehicles
        self.init_prev_vehicles = prev_vehicles
        self.init_vehid = vehid
        self.init_timeind = timeind
        self.dt = dt

        self.vehicles = None
        self.prev_vehicles = None
        self.vehid = None
        self.timeind = None
        self.reset()

    def step(self):
        """Logic for doing a single step of simulation."""
        lc_actions = {}

        for veh in self.vehicles:
            veh.set_cf(self.timeind, self.dt)

        for veh in self.vehicles:
            veh.set_lc(lc_actions, self.timeind, self.dt)

        self.vehid, remove_vehicles = update_net(self.vehicles, lc_actions, self.inflow_lanes,
                                                 self.merge_lanes, self.vehid, self.timeind, self.dt)

        self.timeind += 1
        self.prev_vehicles.extend(remove_vehicles)

    def simulate(self, timesteps):
        """Call step method timesteps number of times."""
        for i in range(timesteps):
            self.step()

    def reset(self):
        """Reset simulation to initial state."""
        self.vehicles = set() if self.init_vehicles is None else copy.deepcopy(self.init_vehicles)
        self.prev_vehicles = [] if self.init_prev_vehicles is None else copy.deepcopy(self.init_prev_vehicles)
        self.vehid = self.init_vehid
        self.timeind = self.init_timeind
        # reset state of boundary conditions
        for curlane in self.inflow_lanes:
            self.vehid = curlane.initialize_inflow(self.vehid)
        # reset state of all AnchorVehicles
        for road in self.roads:
            for lane in road.lanes:
                lane.anchor.reset()
        # reset merge anchors
        for lane in self.merge_lanes:
            lane.merge_anchors = [anchor.copy() for anchor in self.init_merge_anchors[lane]]

class CrashesSimulation(Simulation):
    """Keeps track of crashes in a simulation. Vehicles must have update_after_crash method."""
    def __init__(self, **kwargs):
        self.near_miss_veh = set()
        self.crashed_veh = set()
        self.crashes = []
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.near_miss_veh = set()
        self.crashed_veh = set()
        self.crashes = []

    def step(self):
        super().step()

        for veh in self.vehicles:
            lead, hd = veh.lead, veh.hd
            if lead is not None:
                if 0 < hd/(veh.speed - lead.speed + 1e-6) < 0.4 or hd < 0:  # check for possible near misses
                    self.near_miss_veh.add(veh)
                if hd < 0:  # check for crashes
                    if veh.crashed and lead.crashed:
                        continue
                    most_recent_lc_time = max(veh.lanemem[-1][1], lead.lanemem[-1][1])
                    if self.timeind - most_recent_lc_time < 6:  # LC is not 'completed', so no crash has occured yet
                        continue
                    if not veh.crashed and not lead.crashed:  # normal case of new crash
                        veh.update_after_crash(self.timeind)
                        lead.update_after_crash(self.timeind)
                        self.crashed_veh.add(veh)
                        self.crashed_veh.add(lead)
                        self.crashes.append([veh, lead])
                    else:  # another vehicle is added to the previous crash
                        (crashed_veh, new_veh) = (veh, lead) if veh.crashed else (lead, veh)
                        new_veh.update_after_crash(self.timeind)
                        self.crashed_veh.add(new_veh)
                        for crash in self.crashes[-1::-1]:
                            if crashed_veh in crash:
                                crash.append(new_veh)
                                break
