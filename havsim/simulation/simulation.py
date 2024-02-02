"""Houses the main logic for updating simulations.

The Simulation class is the main high-level API for setting up and running a simulation.
Simulations consist of multiple Roads (which in turn house Lanes), and does the simulation using Vehicles.
The Vehicle class defines the vehicle update model and stores the memory of the simulation.
The Simulation is mainly set up by defining the Roads/Lanes, and setting up the upstream
and downstream boundary conditions (see havsim.simulation.road.Road)
"""
import havsim.simulation as hs
import copy
import time
import tqdm
import sys


def update_net(vehicles, lc_actions, lc_followers, inflow_lanes, merge_lanes, vehid, timeind, dt):
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
        lc_followers: For any Vehicle which request to change lanes, the new follower must be a key in lc_followers,
                value is a list of all vehicles which requested change.
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
    """
    # update followers/leaders for all lane changes
    for vehlist in lc_followers.values():  # check for multiple vehicles changing at the same time into the same gap
        if len(vehlist) > 1:
            for count, veh in enumerate(vehlist):
                if not veh.in_disc:
                    tiebreak_ind = count
                    break
            else:
                tiebreak_ind = 0
            for count, veh in enumerate(vehlist):
                if count == tiebreak_ind:
                    continue
                del lc_actions[veh]

    relaxvehs = []  # keeps track of vehicles which need relaxation applied
    for veh in lc_actions:
        relaxvehs.append(veh.fol)

        # update leader follower relationships, lane/road
        hs.update_lane_routes.update_veh_after_lc(lc_actions, veh, timeind)

        relaxvehs.append(veh)
        relaxvehs.append(veh.fol)

        # update a vehicle's lane events and route events for the new lane
        hs.update_lane_routes.set_lane_events(veh)
        hs.update_lane_routes.set_route_events(veh, timeind)

    for veh in set(relaxvehs):  # apply relaxation
        veh.set_relax(timeind, dt)

    # update all states, memory and headway
    for veh in vehicles:
        veh.update(timeind, dt)
    for veh in vehicles:
        if veh.lead is not None:
            veh.hd = hs.get_headway(veh, veh.lead)

    # update left and right followers
    hs.vehicle_orders.update_all_lrfol_multiple(vehicles)

    # update merge_anchors
    for curlane in merge_lanes:
        hs.update_lane_routes.update_merge_anchors(curlane, lc_actions)

    # update roads (lane events) and routes
    remove_vehicles = []
    for veh in vehicles:
        # check vehicle's lane events and route events, acting if necessary
        hs.update_lane_routes.update_lane_events(veh, timeind, remove_vehicles)
        hs.update_lane_routes.update_route_events(veh, timeind)
    # remove vehicles which leave
    for veh in remove_vehicles:
        vehicles.remove(veh)

    # update inflow, adding vehicles if necessary
    for curlane in inflow_lanes:
        vehid = curlane.increment_inflow(vehicles, vehid, timeind)

    # for veh in vehicles:  # debugging
    #     if not veh._chk_leadfol(verbose = False):
    #         veh._chk_leadfol()

    return vehid, remove_vehicles


class Simulation:
    """Implements a traffic microsimulation.

    Basically just a wrapper for update_net.

    Attributes:
        roads: list of all Roads
        inflow lanes: list of all Lanes which have inflow to them (i.e. all lanes which have upstream
            boundary conditions, meaning they can add vehicles to the simulation)
        merge_lanes: list of all Lanes which have merge anchors
        vehicles: set of all vehicles currently being simulated.
        prev_vehicles: list of all vehicles which have been removed from simulation.
        vehid: starting vehicle ID for the next vehicle to be added. Used for hashing vehicles.
        timeind: the current time index of the simulation (int). Updated as simulation progresses.
        dt: constant float. timestep for the simulation.
    """

    def __init__(self, vehicles=None, prev_vehicles=None, vehid=1, timeind=0, dt=.2, roads=None, timesteps=1):
        """Inits simulation.

        Args:
            vehicles: set of all Vehicles in simulation in first timestep
            prev_vehicles: list of all Vehicles which were previously removed from simulation.
            vehid: vehicle ID used for the next vehicle to be created.
            timeind: starting time index (int) for the simulation.
            dt: float for how many time units pass for each timestep. Defaults to .2.
            roads: list of all the roads in the simulation
            timesteps: int number of default timesteps

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
        self.timesteps = timesteps

        self.vehicles = None
        self.prev_vehicles = None
        self.vehid = None
        self.timeind = None
        self.reset()

    def step(self):
        """Logic for doing a single step of simulation."""
        lc_actions = {}
        lc_followers = {}
        timeind = self.timeind

        for veh in self.vehicles:
            veh.set_cf(timeind)

        for veh in self.vehicles:
            lc_actions, lc_followers = veh.set_lc(lc_actions, lc_followers, timeind)

        self.vehid, remove_vehicles = update_net(self.vehicles, lc_actions, lc_followers, self.inflow_lanes,
                                                 self.merge_lanes, self.vehid, timeind, self.dt)

        self.timeind += 1
        self.prev_vehicles.extend(remove_vehicles)

    def simulate(self, timesteps=None, pbar=True, verbose=True, return_times=False):
        """Do simulation for requested number of timesteps and return all vehicles.

        Args:
            timesteps: int number of timesteps to run simulation. If None, use default value.
            pbar: bool, if True then do simulation with progress bar
            verbose: bool, if True then do simulation with progress bar and print out when finished
            return_times: bool, if True then additionally return the total simulation time and number of timesteps
        Returns:
            all_vehicles: list of Vehicles in the simulation
            elapsed_time: float clock time taken to run simulation
            total_timesteps: total number of timesteps, added over all vehicles
        """
        timesteps = self.timesteps if timesteps is None else timesteps
        elapsed_time = time.time()
        if pbar:
            my_pbar = tqdm.tqdm(range(timesteps), file=sys.stdout)
            my_pbar.set_description('Simulation timesteps')
            for i in my_pbar:
                self.step()
            my_pbar.close()
        else:
            for i in range(timesteps):
                self.step()
        elapsed_time = time.time() - elapsed_time

        all_vehicles = self.prev_vehicles.copy()
        all_vehicles.extend(self.vehicles)
        if verbose or return_times:
            total_timesteps = sum([self.timeind-max(veh.start, self.timeind-timesteps)+1 if veh.end is None
                                   else veh.end-max(veh.start, self.timeind-timesteps)+1 for veh in all_vehicles])
        if verbose:
            print('simulation time is {:.1f} seconds over {:.2e} timesteps ({:n} vehicles)'.format(
                elapsed_time, total_timesteps, len(all_vehicles)))
        if return_times:
            return all_vehicles, elapsed_time, total_timesteps
        return all_vehicles

    def reset(self):
        """Reset simulation to initial state."""
        del self.vehicles
        del self.prev_vehicles

        self.vehicles = set() if self.init_vehicles is None else copy.deepcopy(self.init_vehicles)
        self.prev_vehicles = [] if self.init_prev_vehicles is None else copy.deepcopy(self.init_prev_vehicles)
        self.vehid = self.init_vehid
        self.timeind = self.init_timeind
        # reset state of boundary conditions
        for curlane in self.inflow_lanes:
            self.vehid = curlane.initialize_inflow(self.vehid, self.timeind)
        # reset state of all AnchorVehicles
        for road in self.roads:
            for lane in road.lanes:
                lane.anchor.reset()
                lane.dt = self.dt  # also make sure all lanes have access to dt
        # reset merge anchors
        for lane in self.merge_lanes:
            lane.merge_anchors = [anchor.copy() for anchor in self.init_merge_anchors[lane]]


class CrashesSimulation(Simulation):
    """Keeps track of crashes in a simulation. Vehicles must have update_after_crash method.

    Attributes:
        near_misses: set of all vehicles which experience a near miss, but not a crash
        crashes: list of crashes, each crash is a list of vehicles involved in the crash
        maybe_sideswipes: dict with keys as vehicles, values are (time, lead, changer, previous lane), for keeping
            track of a lane change which may cause a sideswipe if fully completed
        near_miss_times: dict with keys as vehicles, values as list of times of near miss status
    """
    def __init__(self, **kwargs):
        self.near_misses = None
        self.crashes = []
        self.maybe_sideswipes = {}
        self.near_miss_times = {}
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.near_misses = None
        del self.crashes
        del self.maybe_sideswipes
        del self.near_miss_times
        self.crashes = []
        self.maybe_sideswipes = {}
        self.near_miss_times = {}

    def step(self):
        """Modified step logic additionally checks for near misses, sideswipes and rear ends."""
        super().step()
        timeind = self.timeind

        # check for sideswipes
        remove_veh = []
        for veh in self.maybe_sideswipes:
            lc_timeind, lead, changer, prev_lane = self.maybe_sideswipes[veh]
            if veh.lead is None:
                if lead.end:
                    remove_veh.append(veh)
                    continue
                check_crash = True
            elif veh.lead != lead:
                check_crash = True
            else:
                check_crash = False

            if check_crash:
                new_changer = veh if veh.lanemem[-1][1] > lc_timeind else lead
                if new_changer == changer:
                    if new_changer.lane.anchor == prev_lane.anchor:
                        pass  # sideswipe is avoided by aborting the lane change
                    else:
                        crashed = ('sideswipe', timeind)
                        self._add_new_crash(veh, lead, crashed, timeind)
                elif new_changer.lane.anchor != prev_lane.anchor:
                    pass  # sideswipe is avoided by vehicle in target lane making lane change
                else:
                    crashed = ('sideswipe', timeind)
                    self._add_new_crash(veh, lead, crashed, timeind)
                remove_veh.append(veh)
                continue

            if veh.hd > 0:  # sideswipe was avoided before lane change completes
                remove_veh.append(veh)
            elif timeind > lc_timeind + int(1.8/self.dt):  # lane change completes
                crashed = ('sideswipe', timeind)
                self._add_new_crash(veh, lead, crashed, timeind)
                remove_veh.append(veh)

        for veh in remove_veh:
            del self.maybe_sideswipes[veh]

        # check for near misses and rear ends
        for veh in self.vehicles:
            lead, hd = veh.lead, veh.hd
            if lead is not None:
                if hd < 0:  # check for rear ends
                    if veh.crashed and lead.crashed:
                        continue
                    if veh in self.maybe_sideswipes:
                        continue
                    lc_time, changer = self._find_recent_lc_time(veh, lead)
                    if timeind < lc_time + int(1.8/self.dt) + 1:
                        self.maybe_sideswipes[veh] = (timeind, lead, changer, changer.lanemem[-2][0])
                    else:
                        crashed = ('rear end', timeind)
                        self._add_new_crash(veh, lead, crashed, timeind)

                if 0 < hd/(veh.speed - lead.speed + 1e-6) < 0.5 or hd < 0:  # check for possible near misses
                    if veh in self.near_miss_times:
                        self.near_miss_times[veh].append(timeind)
                    elif not veh.crashed:
                        self.near_miss_times[veh] = [timeind]


    def simulate(self, timesteps=None, pbar=True, verbose=True, return_times=False):
        out = super().simulate(timesteps=timesteps, pbar=pbar, verbose=verbose, return_times=return_times)
        self._process_near_miss_times()

        if verbose:
            n_misses = sum([len(veh.near_misses) for veh in self.near_misses])
            print('number of near misses: {:n} ({:n} vehicles)'.format(n_misses, len(self.near_misses)))
            n_crashed_veh = sum([len(crash) for crash in self.crashes])
            print('number of crashes: {:n} ({:n} vehicles)'.format(len(self.crashes), n_crashed_veh))
        return out

    def _add_new_crash(self, veh, lead, crashed, timeind):
        if not veh.crashed and not lead.crashed:
            veh.update_after_crash(timeind, crashed)
            lead.update_after_crash(timeind, crashed)
            self.crashes.append([veh, lead])
            return
        elif veh.crashed:
            crashed_veh, new_veh = veh, lead
        else:
            crashed_veh, new_veh = lead, veh
        for crash in self.crashes[-1::-1]:
            if crashed_veh in crash:
                crash.append(new_veh)
                crashed = (crashed[0], crash[0].crash_time)
                new_veh.update_after_crash(timeind, crashed)
                break

    def _process_near_miss_times(self):
        """Convert near miss times into near miss intervals (CrashesVehicles have reference to their near misses)."""
        # convert all near miss times into intervals
        near_misses = {}
        for veh, times in self.near_miss_times.items():
            cur_near_miss = []
            if times[-1] == times[0] + len(times) - 1:
                cur_near_miss.append((times[0], times[-1]))
            else:
                start_ind, prev_time, cur_len = 0, times[0], len(times)
                while start_ind < cur_len:
                    for ind in range(start_ind+1, cur_len):
                        if times[ind] > prev_time + 20:
                            cur_near_miss.append((times[start_ind], prev_time))
                            start_ind, prev_time = ind, times[ind]
                            break
                        else:
                            prev_time = times[ind]
                    else:
                        cur_near_miss.append((times[start_ind], times[-1]))
                        break
                    if times[-1] == times[start_ind] + cur_len - 1 - start_ind:
                        cur_near_miss.append((times[start_ind], times[-1]))
                        break
            near_misses[veh] = cur_near_miss

        # remove near miss intervals that are due to crashes
        for crash in self.crashes:
            if crash[0] in near_misses:
                self._maybe_fix_near_miss(crash[0], near_misses[crash[0]])
            for veh in crash[2:]:
                if veh in near_misses:
                    self._maybe_fix_near_miss(veh, near_misses[veh])

        # save results to near_misses attributes in simulation/vehicles
        self.near_misses = set()
        for veh, times in near_misses.items():
            if len(times) > 0:
                veh.near_misses = times
                self.near_misses.add(veh)

    @staticmethod
    def _find_recent_lc_time(veh, lead):
        veh_lc = veh.lanemem[-1][1] if len(veh.lanemem) > 1 else -100000000
        lead_lc = lead.lanemem[-1][1] if len(lead.lanemem) > 1 else -100000000
        if lead_lc > veh_lc:
            return lead_lc, lead
        else:
            return veh_lc, veh

    @staticmethod
    def _maybe_fix_near_miss(veh, cur_near_miss):
        crash_time = veh.crash_time
        remove_inds = []
        for count, interval in enumerate(cur_near_miss):
            if interval[1] >= crash_time:
                remove_inds.append(count)
        for count in remove_inds:
            cur_near_miss.pop(count)
