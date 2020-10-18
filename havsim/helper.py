"""Helper functions and utilities for loading/manipulating data."""
import numpy as np
import heapq
import math
from collections import defaultdict
# TODO - fix code style and documentation


def load_dataset(dataset, column_dict={'veh_id': 0, 'frame_id': 1, 'lane': 13, 'pos': 5, 'veh_len': 8,
                                       'lead': 14, 'fol': 15}, dt=0.1, alpha=0.1, None_val=0):
    """Returns dictionary with keys veh_id, and values as data object.

    Args:
        dataset: Consists of rows of observation, assumed to be sorted by vehicle IDs, times.
            each observation has the values of
            veh id - vehicle ID (unique float)
            frame id - time index (integer)
            lane - lane the vehicle is on
            pos - position in direction of travel
            veh len - vehicle length
            lead - lead vehicle ID
            fol - following vehicle ID
        column_dict: Gives the column indices for each value
        dt: float representing timestep in simulation
        alpha: float representing alpha in exponential moving average (which we
            utilize to smooth positions). lower = less smoothing.

    Returns:
        dictionary with keys of vehicle ids, values of VehicleData.
    """

    def _get_vehid(lane_np, pos):
        """Gets vehicle id of the follower/leader in the given lane (utilizing y_val of the curr veh).

        Args:
            lane_np: np.ndarray, a subset of dataset (only includes rows of a certain frame id,
                and lane). This must be sorted via the pos column or it produces incorrect
                solutions
            pos: float, the pos value of the vehicle we're searching up (the vehicle
                whose lfol/rfol/llead/rlead we are currently trying to triangulate)

        Returns:
            fol_id: float, the vehicle id of the follower in the given lane
            lead_id: float, the vehicle id of the leader in the given lane
        """
        idx_of_fol = np.searchsorted(lane_np[:, col_dict['pos']], pos) - 1
        if idx_of_fol != -1:
            return lane_np[idx_of_fol, col_dict['veh_id']], lane_np[idx_of_fol, col_dict['lead']]
        else:
            return None, lane_np[0, col_dict['veh_id']]

    def _get_fol_lead(row_np):
        """Returns lfol/llead/rfol/llead of a given row in the dataframe.

        Note that we assume that the lanes have integer indices (e.g. 1,2,3, etc.) and that
        1 connects to 2, etc. So this is only designed to work for a road segment.

        Args:
            row_np: np.ndarray, must be a vector. The row of the dataset, which represents
                a vehicle at a given frame_id.

        Returns:
            lfol: float, vehicle id of left follower of vehicle at given frame id
            llead: float, vehicle id of left leader of vehicle at given frame id
            rfol: float, vehicle id of right follower of vehicle at given frame id
            rlead: float, vehicle id of right leader of vehicle at given frame id
        """
        # lane_id_to_indices must accept lane index, return all observations in that lane, at the same
        # time index as row_np
        lane_id, pos = row_np[col_dict['lane']], row_np[col_dict['pos']]
        lfol, llead, rfol, rlead = None, None, None, None

        # lfol/llead
        if (lane_id >= 2 and lane_id <= 6) or \
                (lane_id == 7 and pos >= 400 and pos <= 750):
            left_lane_id = lane_id - 1

            if left_lane_id in lane_id_to_indices:
                min_idx, max_idx = lane_id_to_indices[left_lane_id]
                left_lane_np = curr_frame[min_idx:max_idx]
                lfol, llead = _get_vehid(left_lane_np, pos)

        # rfol/rlead
        if (lane_id >= 1 and lane_id <= 5) or \
                (lane_id == 6 and pos >= 400 and pos <= 750):
            right_lane_id = lane_id + 1

            if right_lane_id in lane_id_to_indices:
                min_idx, max_idx = lane_id_to_indices[right_lane_id]
                right_lane_np = curr_frame[min_idx:max_idx]
                rfol, rlead = _get_vehid(right_lane_np, pos)

        return lfol, llead, rfol, rlead

    columns = list(column_dict.keys())
    col_idx = list(column_dict.values())

    dataset = dataset[:, col_idx]
    col_dict = {col: idx for idx, col in enumerate(columns)}

    # generate lfol/llead/rfol/rlead data
    # sort dataset in times, lanes, positions
    dataset = np.hstack((dataset, np.expand_dims(np.array(range(len(dataset))), axis=1)))
    sort_tuple = (dataset[:, col_dict['pos']], dataset[:, col_dict['lane']],
                  dataset[:, col_dict['frame_id']])
    dataset_sortframes = dataset[np.lexsort(sort_tuple)]

    # get lfol/llead/rfol/rlead for each vehicle in each time
    lc_data = [[], [], [], []]  # shape of (4, number of observations), where the 4 is lfol,llead,rfol,rlead
    frame_ids, indices, counts = \
        np.unique(dataset_sortframes[:, col_dict['frame_id']], return_index=True, return_counts=True)
    for count, frame_id in enumerate(frame_ids):
        curr_frame = dataset_sortframes[indices[count]:indices[count]+counts[count], :]

        # generate min/max index for each lane within this frame
        lane_id_to_indices = {}
        lane_ids, lane_indices, lane_counts = np.unique(curr_frame[:, col_dict['lane']], return_index=True,
                                                        return_counts=True)
        for countlane, lane_id in enumerate(lane_ids):
            lane_id_to_indices[lane_id] = (lane_indices[countlane],
                                           lane_indices[countlane]+lane_counts[countlane])

        # calculate lc data
        for row_idx in range(curr_frame.shape[0]):
            for idx, lc_datum in enumerate(_get_fol_lead(curr_frame[row_idx, :])):
                lc_data[idx].append(lc_datum)
    lc_data = np.array(lc_data).T
    dataset = np.hstack((dataset[:, :-1], lc_data[np.argsort(dataset_sortframes[:, -1])]))

    # record the new columns for lfol,llead,rfol,rlead
    lc_cols = ['lfol', 'llead', 'rfol', 'rlead']
    for col in lc_cols:
        col_dict[col] = len(columns)
        columns.append(col)

    def convert_to_mem(veh_np, veh_dict):
        """Generates compressed representation of lane/lead/lfol/rfol/llead/rlead.

        Args:
            veh_np: np.ndarray. Subset of dataset that only includes a given veh_id
                Note that the dataset should be sorted in terms of frame_id
            veh_dict: dict, str -> list or float. Dictionary that includes all
                information about the vehicle. Keys include: posmem, dt, speedmem, etc.
                This is updated within the function with lanemem/leadmem/etc.

        Returns:
            None

        """
        colnames = ['lanemem', 'leadmem', 'folmem', 'lfolmem', 'rfolmem', 'lleadmem', 'rleadmem']
        curr_vals = {col: None for col in colnames}
        final_mems = [[] for col in colnames]

        for idx in range(veh_np.shape[0]):
            row = veh_np[idx, :]
            for idx, col in enumerate(colnames):
                # preprocess val
                val = row[col_dict[col.replace("mem", "")]]
                if val == None_val:
                    val = None

                # if we're just starting or the saved value is different
                # than the current value, we update the mem
                if len(final_mems[idx]) == 0 or curr_vals[col] != val:
                    curr_vals[col] = val
                    final_mems[idx].append((val, int(row[col_dict['frame_id']])))
        # save to veh_dict
        for idx, col in enumerate(colnames):
            veh_dict[col] = final_mems[idx]

    def ema(np_vec, alpha=0.9):
        """Returns the exponential moving average of np_vec.

        Args:
            np_vec: np.ndarray, must be vector. The vector that we want to smooth
            alpha: float. The alpha parameter in exponential moving average
                x[i] = alpha * x[i-1] + (1 - alpha) * np_vec[i]
        Returns:
            res: list w/same len as np_vec, with exponentially smoothed values
        """
        assert(len(np_vec.shape) == 1)
        res = [np_vec[0]]
        curr_val = np_vec[0]
        for i in range(1, np_vec.shape[0]):
            curr_val = curr_val * alpha + (1 - alpha) * np_vec[i]
            res.append(curr_val)
        return res

    # generate veh_dict with vehicledata objects
    all_veh_dict = {}
    veh_ids, veh_inds, veh_counts = np.unique(dataset[:, col_dict['veh_id']],
                                              return_index=True, return_counts=True)
    for count, veh_id in enumerate(veh_ids):
        veh_dict = {}
        veh_np = dataset[veh_inds[count]:veh_inds[count]+veh_counts[count]]
        pos_mem = ema(veh_np[:, col_dict['pos']], alpha)
        veh_dict['posmem'] = list(pos_mem)

        speed_mem = [(pos_mem[i + 1] - pos_mem[i]) / dt for i in range(len(pos_mem) - 1)]  # re differentiate
        speed_mem.append(speed_mem[-1])
        veh_dict['speedmem'] = speed_mem

        veh_dict['start'] = int(veh_np[0, col_dict['frame_id']])
        veh_dict['end'] = int(veh_np[-1, col_dict['frame_id']])
        veh_dict['vehlen'] = veh_np[0, col_dict['veh_len']]
        veh_dict['dt'] = dt
        veh_dict['vehid'] = veh_id

        convert_to_mem(veh_np, veh_dict)
        all_veh_dict[veh_id] = VehicleData(vehdict=all_veh_dict, **veh_dict)
    return all_veh_dict


class VehicleData:
    """Holds trajectory data for a single vehicle.

    Attributes:
        posmem: list of floats giving the position, where the 0 index corresponds to time = start
        speedmem: list of floats giving the speed, where the 0 index corresponds to time = start
        vehid: unique vehicle ID (float) for hashing. Note that 0 is not a valid vehid.
        start: first time index vehicle data is available
        end: the last time index vehicle data is available
        dt: time discretization used (discretization is assumed to be constant)
        vehlen: physical length of vehicle
        leadmem: VehMem object for the leader, leader.pos,  leader.speed, leader.len information.
            VehMem objects are indexed by start, not 0. See VehMem.
        lanemem: VehMem object for the lane. See VehMem.
        folmem: VehMem object for the follower, follower.pos,  follower.speed, follower.len information.
            See VehMem
        lfolmem: Like folmem, but for left follower (follower if vehicle changed into left lane)
        rfolmem: Like folmem, but for right follower (follower if vehicle changed into right lane)
        rleadmem: like leadmem, but for left leader (leader if vehicle changed into left lane)
        lleadmem: like leadmem, but for right leader (leader if vehicle changed into right lane)
        longest_lead_times: tuple of starting, ending time index for the longest interval with
            leader(s) not None.
        leads: list of unique lead IDs (does not include None)
    """

    def __init__(self, posmem=[], speedmem=[], vehid=None, start=None, dt=None, vehlen=None,
                 lanemem=[], leadmem=[], folmem=[], end=None, lfolmem=[], rfolmem=[],
                 rleadmem=[], lleadmem=[], vehdict=None):
        """Inits VehicleData and also makes the Vehmem objects used for the 'mem' data.

        Args:
            posmem: list of floats giving the position, where the 0 index corresponds to the position at
                start
            speedmem: list of floats giving the speed, where the 0 index corresponds to the speed at start
            vehid: unique vehicle ID (float) for hashing. Note that 0 is not a valid vehid.
            start: first time index vehicle data is available
            dt: time discretization used (discretization is assumed to be constant)
            vehlen: physical length of vehicle
            leadmem: list of tuples, where each tuple is (vehid, time) giving the time the ego vehicle
                first begins to follow the vehicle with id vehid. Note vehid=None is also possible.
            lanemem: list of tuples, where each tuple is (laneid, time) giving the time the ego vehicle
                first enters the lane with laneid
            folmem: list of tuples, where each tuple is (vehid, time) giving the time the ego vehicle first
                begins to act as follower for the vehicle with id vehid. Note vehid=None is also possible.
            end: the last time index vehicle data is available
            lfolmem: Like folmem, but for left follower (follower if vehicle changed into left lane)
            rfolmem: Like folmem, but for right follower (follower if vehicle changed into right lane)
            rleadmem: like leadmem, but for left leader (leader if vehicle changed into left lane)
            lleadmem: like leadmem, but for right leader (leader if vehicle changed into right lane)
            vehdict: dictionary containing all VehicleData, where keys are the vehid
        """
        self.posmem = posmem
        self.speedmem = speedmem
        self.vehid = vehid
        self.start = start
        self.end = end
        self.dt = dt
        self.len = vehlen
        self.lanemem = VehMem(lanemem, vehdict, start, end, is_lane=True)
        self.leadmem = VehMem(leadmem, vehdict, start, end)
        self.folmem = VehMem(folmem, vehdict, start, end)
        self.lfolmem = VehMem(lfolmem, vehdict, start, end)
        self.rfolmem = VehMem(rfolmem, vehdict, start, end)
        self.rleadmem = VehMem(rleadmem, vehdict, start, end)
        self.lleadmem = VehMem(lleadmem, vehdict, start, end)

        self.leads = self.get_unique_mem(leadmem)
        self.longest_lead_times = self.get_longest_lead_times()

    def get_longest_lead_times(self):
        """Find the longest time interval with leader is not None and return the starting/ending times."""
        # similar code could be used to list all intervals with leader not None - would we ever need that?
        if len(self.leads) == 0:
            return self.start, self.start
        longest = 0
        longest_start = -1
        longest_end = -1

        curr_start = None
        running_interval = 0
        # to deal with last element case
        temp = self.leadmem.data.copy()
        temp.append((None, self.end+1))

        for idx, lead_id_and_time in enumerate(temp):
            lead, start_time = lead_id_and_time

            if curr_start is None and lead is not None:
                # setting up
                curr_start = start_time
                running_interval = 0
            elif curr_start is not None:
                # update the running interval
                running_interval = start_time - curr_start

            # no leader, so reset
            if lead is None:
                if running_interval > longest:
                    longest = running_interval
                    longest_start = curr_start
                    longest_end = start_time-1
                curr_start = None
                running_interval = 0

        return int(longest_start), int(longest_end)

    def get_unique_mem(self, mem):
        """Returns a list of unique values in sparse 'mem' representation, not including None."""
        out = set([i[0] for i in mem])
        if None in out:
            out.remove(None)
        return list(out)

    def __hash__(self):
        """Vehicles/VehicleData are hashable by their unique vehicle ID."""
        return hash(self.vehid)

    def __eq__(self, other):
        """Used for comparing two vehicles with ==."""
        return self.vehid == other.vehid

    def __ne__(self, other):
        """Used for comparing two vehicles with !=."""
        return not self.vehid == other.vehid

    def __repr__(self):
        """Display for vehicle in interactive console."""
        return 'saved data for vehicle '+str(self.vehid)

    def __str__(self):
        """Convert to a str representation."""
        return self.__repr__()


def convert_to_data(vehicle):
    """Converts a (subclassed) Vehicle to VehicleData."""
    # should convert Vehicle objects to their vehid. Need to convert Lanes to some index?
    raise NotImplementedError
    return VehicleData(vehicle)


class VehMem:
    """Implements memory of lane, lead, fol, rfol, lfol, rlead, llead.

    Attributes:
        data: sparse representation of memory
        start: start time of VehicleData
        end: end of VehicleData
        pos: VehMemPosition, positions of the VehMem which can be directly indexed/sliced using times
            (pos, speed, len are not for lanemem)
        speed: VehMemPosition, speeds of the VehMem which can be directly indexed/sliced using times
        len: VehMemPosition, len of the VehMem which can be directly indexed/sliced using times
    """

    def __init__(self, vehmem, vehdict, start, end, is_lane=False):
        """Inits VehMem and also makes the corresponding pos, speed, len objects if is_lane is False.

        Args:
            vehmem: sparse representation of one of the  lanemem, leadmem, folmem, rfolmem, lfolmem, rleadmem,
                lleadmem
            vehdict: dictionary containing all VehicleData
            start: start time of the VehicleData which has vehmem
            end: end time of the VehicleData which has vehmem
            is_lane: True if vehmem gives lanemem, otherwise False. If True, we will make the pos, speed, len
        """
        self.data = vehmem
        self.start = start
        self.end = end
        if not is_lane:
            self.pos = VehMemPosition(vehmem, vehdict, start, end)
            self.speed = VehMemSpeed(vehmem, vehdict, start, end)
            self.len = VehMemLen(vehmem, vehdict, start, end)

    def __getitem__(self, times):
        """Index memory using times, as if it was in its dense representation. Also accepts slice input.

        Behaves as if the memory was in a dense representation, and accepted times instead of indices.
        Note that to change times into indices, you would subtract the times by start. Similarly to
        change indices into times, you would add start to the indices.

        E.g. self.data = [[0,5], [2, 8], [6, 12]], self.start = 5, self.end = 17
        self[5] = 0, self[8] = 2, self[17] = 6
        self[4], self[18], self[0], self[-1] all throw index errors.
        self[5:10] = self[:10] = [0, 0, 0, 2, 2]  (return dense representation between times 5 and 9)
        self[10:] = self[10:18] = [2, 2, 6, 6, 6, 6, 6, 6]
        self[:] = [0, 0, 0, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6]  (return entire dense representation)
        self[18:] = self[:4] = []  (behaves as slicing a regular list would)
        self[17:] = self[17:18] = [6]
        self[11:10] = []   (no steps for slices - must always advance by 1 time index)
        self[5:10:2] = self[5:10]   (no steps for slices - must always advance by 1 time index)
        """
        if type(times) == slice:
            data = self.data
            start, stop = times.start, times.stop
            if not start:
                start = self.start
            elif start < self.start:
                start = self.start
            elif start > self.end:
                return []
            if not stop:
                stop = self.end+1
            elif stop > self.end+1:
                stop = self.end+1
            elif stop < self.start+1:
                return []

            startind, stopind = mem_binary_search(data, start, return_ind=True), \
                mem_binary_search(data, stop, return_ind=True)
            out = []

            timeslist = (start, *(data[i][1] for i in range(startind+1, stopind+1)), stop)
            for count, i in enumerate(range(startind, stopind+1)):
                out.extend([data[i][0]]*(timeslist[count+1]-timeslist[count]))
            return out
        else:
            if times < self.start or times > self.end:
                raise IndexError
            return mem_binary_search(self.data, times)

    def intervals(self, start=None, stop=None):
        """Converts sparse representation into an interval representation.

        See mem_to_interval - the difference is that this will behave as slicing does for
        times outside of [self.start, self.end].
        E.g. self.data = [[1,3], [2,6]], self.start = 3, self.end = 10
        self.intervals() = self.intervals(3, 10) = [[1, 3, 6], [2, 6, 11]]
        self.intervals(4, 8) = [[1, 4, 6], [2, 6, 9]]
        self.intervals(2, 11) = self.intervals(3, 10) = [[1, 3, 6], [2, 6, 11]]
        """
        if not start:
            start = self.start
        elif start < self.start:
            start = self.start
        elif start > self.end:
            return []
        if not stop:
            stop = self.end
        elif stop > self.end:
            stop = self.end
        elif stop < self.start:
            return []
        return mem_to_interval(self.data, start, stop)

    def __repr__(self):
        """Ipython Console represention."""
        return 'VehMem: '+str(self.data)


class VehMemPosition:
    """Implements slicing/indexing for the position of a (l/r)-(lead/fol)-mem

    Attributes:
        data: the sparse representation of the (l/r)-(lead/fol)-mem
        vehdict: dictionary of all VehicleData - this is what the position data is read from
        start: start of the vehicle which has the mem
        end: end of the vehicle which has the mem
    """
    def __init__(self, vehmem, vehdict, start, end):
        """Inits VehMemPosition - note vehdict must contain VehicleData for any vehicles in vehmem."""
        self.data = vehmem
        self.vehdict = vehdict
        self.start = start
        self.end = end

    def __getitem__(self, times):
        """Index or slice memory as if it was in its dense representation."""
        if type(times) == slice:
            data = self.data
            vehdict = self.vehdict
            start, stop = times.start, times.stop
            if not start:
                start = self.start
            elif start < self.start:
                start = self.start
            elif start > self.end:
                return []
            if not stop:
                stop = self.end+1
            elif stop > self.end+1:
                stop = self.end+1
            elif stop < self.start+1:
                return []

            startind, stopind = mem_binary_search(data, start, return_ind=True), \
                mem_binary_search(data, stop, return_ind=True)
            out = []

            timeslist = (start, *(data[i][1] for i in range(startind+1, stopind+1)), stop)
            for count, i in enumerate(range(startind, stopind+1)):
                curveh = data[i][0]
                if curveh is not None:
                    vehdata = vehdict[curveh]
                    out.extend(self.myslice(vehdata, timeslist[count], timeslist[count+1]))
                else:
                    out.extend([None]*(timeslist[count+1]-timeslist[count]))
            return out
        else:
            if times < self.start or times > self.end:
                raise IndexError
            vehdata = self.vehdict[mem_binary_search(self.data, times)]
            return self.index(vehdata, times)

    def myslice(self, vehdata, start, stop):
        curstart = vehdata.start
        return vehdata.posmem[start-curstart:stop-curstart]

    def index(self, vehdata, time):
        return vehdata.posmem[time - vehdata.start]

    def __repr__(self):
        return str(self.__getitem__(slice(self.start, self.end+1)))


class VehMemSpeed(VehMemPosition):
    """Returns speed data instead of position data."""
    def myslice(self, vehdata, start, stop):
        curstart = vehdata.start
        return vehdata.speedmem[start-curstart:stop-curstart]

    def index(self, vehdata, time):
        return vehdata.speedmem[time - vehdata.start]


class VehMemLen(VehMemPosition):
    """Returns lengths instead of position data."""
    def myslice(self, vehdata, start, stop):
        return (vehdata.len,)*(stop-start)

    def index(self, vehdata, time):
        return vehdata.len


def mem_binary_search(arr, time, return_ind=False):
    """
    Performs binary search on array, but assumes that the array is filled with tuples, where
    the first element is the value and the second is the time. We're sorting utilizing the time field
    and returning the value at the specific time.
    Args:
        arr: sorted array with (val, start_time)
        time: the time that we are looking for
        return_ind: If True, return the index corresponding to time in arr
    Returns:
        the value that took place at the given time
    """
    # if len(arr) == 0 or time < arr[0][1]:
    #     return None
    start, end = 0, len(arr)
    while (end > start+1):
        mid = (end + start) // 2
        if time >= arr[mid][1]:
            start = mid
        else:
            end = mid

    if return_ind:
        return start
    else:
        return arr[start][0]


def interval_binary_search(X,time):
    #finds index m such that the interval X[m], X[m+1] contains time.
    #X = array
    #time = float
    lo = 0
    hi = len(X)-2
    m = (lo + hi) //  2
    while (hi - lo) > 1:
        if time < X[m]:
            hi = m
        else:
            lo = m
        m = (lo + hi) // 2
    return lo


def mem_to_interval(vehmem, start, stop):
    """Converts memory between times start and stop to an interval representation.

    E.g. vehmem = [[1,3], [2,6]], start = 4, stop = 8
    interval representation = [[1,4,6], [2,6,9]]
    The interval representation gives the start, end times so it is more convenient to use.
    Note that if you gives start/end times outside of the memory, it will just use the closest memory.
    E.g. vehmem = [[1,3], [2,6]], start = 2, stop = 1000
    interval representation = [[1, 2, 6], [2, 6, 1001]]
    """
    startind, stopind = mem_binary_search(vehmem, start, return_ind=True), \
        mem_binary_search(vehmem, stop, return_ind=True)

    timeslist = (start, *(vehmem[i][1] for i in range(startind+1, stopind+1)), stop+1)
    return [[vehmem[startind+i][0], timeslist[i], timeslist[i+1]] for i in range(len(timeslist)-1)]


def get_lead_data(veh, meas, platooninfo, rp=None, dt=.1):
    """Returns lead vehicle trajectory and possibly relaxation

    Args:
        veh: vehicle to obtain leader for
        meas: from makeplatooninfo
        platooninfo: from makeplatooninfo
        rp: if not None, return the relaxation amounts calculated from the measurments, not including mergers,
            with a relaxation parameter of rp. The relaxation is calculated using r_constant and makerinfo,
            which will give slightly different results than havsim.simulation.relaxation

    Returns:
        leadpos: (times,) numpy array of lead position - lead length (may refer to multiple vehicles)
        leadspeed: (times,) numpy array of lead speed (may refer to multiple vehicles)
        relax: (if rp is not None) (times,1) numpy array of relaxation amounts
    """
    t0, t1, t2, t3 = platooninfo[veh][:4]
    leadinfo = makeleadinfo([veh], platooninfo, meas)

    lead = np.zeros((t2-t1+1,3))
    for j in leadinfo[0]:
        curlead, start, stop = j
        lead_t0 = platooninfo[curlead][0]
        lead[start-t1:stop+1-t1, :] = meas[curlead][start-lead_t0:stop+1-lead_t0, [2, 3, 6]]

    if rp is not None:
        rinfo = makerinfo([veh], platooninfo, meas, leadinfo, relaxtype='both', mergertype=None)
        relax, unused = r_constant(rinfo[0], [t1, t2], t3, rp, adj=False, h=dt)

        return lead[:,0] - lead[:,2], lead[:,1], relax[:t2-t1+1]

    return lead[:,0] - lead[:,2], lead[:,1]


def get_fixed_relaxation(veh, meas, platooninfo, rp, dt=.1):
    """Gets headway relaxation amounts determined apriori using makerinfo and r_constant for parameter rp."""
    t0, t1, t2, t3 = platooninfo[veh][:4]
    leadinfo = makeleadinfo([veh], platooninfo, meas)
    rinfo = makerinfo([veh], platooninfo, meas, leadinfo, relaxtype='both', mergertype=None)
    relax, unused = r_constant(rinfo[0], [t1, t2], t3, rp, adj=False, h=dt)
    return relax[:t2-t1+1]


def makeplatooninfo(dataset, simlen = 50):
#	This looks at the entire dataset, and makes a platooninfo entry for every single vehicle. The platooninfo tells us during which times we calibrate a vehicle. It also contains
#the vehicles intial conditions, as well as the vehicles leaders, and any followers the vehicle has. note that we modify the folowers it has during the time
#we use the makeplatoon function.
#The function also makes the outputs leaders, G (follower network), simcount, curlead, totfollist, followers, curleadlist
#All of those things we need to run the makeplatoon function, and all of those are modified everytime makeplatoon runs.
#
#input:
#    dataset: source of data,
#
#    dataind: column indices for data entries. e.g. [3,4,9,8,6,2,5] for reconstructed. [5,11,14,15,8,13,12] for raw/reextracted. (optional)
#    NOTE THAT FIRST COLUMN IS VEHICLE ID AND SECOND COLUMN IS FRAME ID. THESE ARE ASSUMED AND NOT INCLUDED IN DATAIND
#            0- position,
#            1 - speed,
#            2 - leader,
#            3 - follower.
#            4 - length,
#            5 - lane,
#            6 - length
            #7 -lane
            #8 - acceleration
#            e.g. data[:,dataind[0]] all position entries for entire dataset, data[data[:,1]] all frame IDs for entire dataset
#
#    simlen = 50: this is the minimum number of observations a vehicle has a leader for it to be calibrated.
#
#output:
#    meas: all measurements with vehID as key, values as dataset[dataset==meas.val()] i.e. same format as dataset
            #rows are observations.
            #columns are:
            #0 - id
            #1 - time
            #2 - position
            #3 - speed
            #4 - leader
            #5 - follower
            #6 - length
            #7 - lane
            #8 - acceleration
#
#    platooninfo: dictionary with key as vehicle ID, (excludes lead vehicle)
#    value is array containing information about the platoon and calibration problem
#        0 - t_nstar (first recorded time, t_nstar = t_n for makefollowerchain)
#        1 - t_n (first simulated time)
#        2 - T_nm1 (last simulated time)
#        3 - T_n (last recorded time)
#        4 - array of lead vehicles
#        5 - position IC
#        6 - speed IC
#        7 - [len(followerlist), followerlist], where followerlist is the unique simulated followers of the vehicle. note followerlist =/= np.unique(LCinfo[:,2])
#
#    leaders: list of vehicle IDs which are not simulated. these vehicles should not have any leaders, and their times should indicate that they are not simulated (t_n = T_nm1)
#
#    simcount: count of how many vehicles in the dataset have 1 or more followers that need to be simulated. If simcount is equal to 0, then
#    there are no more vehicles that can be simulated. Simcount is not the same as how many vehicles need to be simulated. In above example, simcount is 1.
#    If vehicles 2 and 4 are both simulated, the simcount will drop to 0, and all vehicles that can be simulated have been simulated.
#
#    curlead: this is needed for makeplatoon. We assign it as None for initialization purposes. This is the most recent vehicle added as a leader to curleadlist
#
#    totfollist: this represents ALL possible followers that may be able to be added based on all the vehicles currently in leaders. list. needed for makeplatoon
#
#    followers: this is NOT all possible followers, ONLY of any vehicle that has been assigned as 'curlead' (so it is the list of all vehicles that can follow anything in curleadlist).
#    The purpose of this variable in addition to totfollist is that we want to prioritize vehicles that have their leader in the current platoon.
#
#    curleadlist: this is needed for makeplatoon. the current list of leaders that we try to add followers for
	##########
    #inputs - dataset. it should be organized with rows as observations and columns have the following information
    #dataind: 0 - vehicle ID, 1- time,  2- position, 3 - speed, 4 - leader, 5 - follower. 6 - length, 7 - lane, 8 - acceleration
    #the data should be sorted. Meaning that all observations for a vehicle ID are sequential (so all the observations are together)
    #additionally, within all observations for that vehicle time should be increasing.

    #simlen = 50 - vehicles need to have a leader for at least this many continuous observations to be simulated. Otherwise we will not simulate.

    #this takes the dataset, changes it into a dictionary where the vehicle ID is the key and values are observations.

    vehlist, vehind, vehct = np.unique(dataset[:,0], return_index =True, return_counts = True) #get list of all vehicle IDs. we will operate on each vehicle.

    meas = {} #data will go here
    platooninfo = {} #output
    masterlenlist = {} #this will be a dict with keys as veh id, value as length of vehicle. needed to make vehlen, part of platooninfo. not returned
    leaders = [] #we will initialize the leader information which is needed for makeplatoon

    for z in range(len(vehlist)): #first pass we can almost do everything
        i = vehlist[z] #this is what i used to be
        curveh = dataset[vehind[z]:vehind[z]+vehct[z],:] #current vehicle data
#        LCinfo = curveh[:,[1,dataind[2], dataind[3]]] #lane change info #i'm taking out the LCinfo from the platooninfo since the information is all in meas anyway
        lanedata = np.nonzero(curveh[:,4])[0] #row indices of LCinfo with a leader
        #the way this works is that np.nonzero returns the indices of LCinfo with a leader. then lanedata increases sequentially when there is a leader,
        #and has jumps where there is a break in the leaders. so this is the exact form we need to use checksequential.
        mylen = len(lanedata)
        lanedata = np.append(lanedata, lanedata) #we need to make this into a np array because of the way checksequential works so we'll just repeat the column
        lanedata = lanedata.reshape((mylen,2) , order = 'F')
        unused, indjumps = checksequential(lanedata) #indjumps returns indices of lanedata, lanedata itself are the indices of curveh.
        t_nstar = int(curveh[0,1]) #first time measurement is known
        T_n = int(curveh[-1,1]) #last time measurement is known

        masterlenlist[i] = curveh[0,6] #add length of vehicle i to vehicle length dictionary
        meas[i] = curveh #add vehicle i to data

        if np.all(indjumps == [0,0]) or curveh[0,6]==0: #special case where vehicle has no leaders will cause index error; we cannot simulate those vehicles
            #also if the length of the vehicle is equal to 0 we can't simulate the vehicle. If we can't simulate the vehicle, t_nstar = t_n = T_nm1
            t_n = t_nstar  #set equal to first measurement time in that case.
            T_nm1 = t_nstar
            platooninfo[i] = [t_nstar, t_n, T_nm1, T_n, [], curveh[0,2], curveh[0, 3], []]
#            platooninfo[i] = [t_nstar, t_n, T_nm1, T_n, [], [0, []]]
            continue

        t_n = int(curveh[lanedata[indjumps[0],0],1]) #simulated time is longest continuous episode with a leader, slice notation is lanedata[indjumps[0]]:lanedata[indjumps[1]]
        T_nm1 = int(curveh[lanedata[indjumps[1]-1,0],1])

        if (T_nm1 - t_n) < simlen: #if the simulated time is "too short" (less than 5 seconds) we will not simulate it (risk of overfitting/underdetermined problem)
            t_n = t_nstar
            T_nm1 = t_nstar


        platooninfo[i] = [t_nstar, t_n, T_nm1, T_n, [], curveh[t_n-t_nstar,2], curveh[t_n-t_nstar, 3], []] #put in everything except for vehicle len and the follower info
#        platooninfo[i] = [t_nstar, t_n, T_nm1, T_n, [], [0, []]]
    for i in vehlist: #second pass we need to construct vehlen dictionary for each vehicle ID. we will also put in the last entry of platooninfo, which gives info on the followers
#        vehlen = {} #initialize vehlen which is the missing entry in platooninfo for each vehicle ID
        curinfo = platooninfo[i] #platooninfo for current veh
        t_nstar, t_n, T_nm1 = curinfo[0], curinfo[1], curinfo[2]
        leaderlist = list(np.unique(meas[i][t_n-t_nstar:T_nm1-t_nstar+1,4])) #unique leaders
        if 0 in leaderlist: #don't care about 0 entry remove it
            leaderlist.remove(0)
        for j in leaderlist: #iterate over each leader
#            if j == 0.0: #vehID = 0 means no vehicle (in this case, no leader)
#                continue
#            vehlen[j] = meas[j][0,dataind[4]] #put in vehicle length of each leader during simulated times
            #now we will construct the last entry of platooninfo
            platooninfo[j][-1].append(i) #vehicle j has vehicle i as a follower.
        platooninfo[i][4] = leaderlist #put in the leader information

        #now we have identified all the necessary information to setup the optimization problem.
        #first thing to do is to identify the vehicles which are not simulated. These are our lead vehicles.
        if (T_nm1 - t_n) == 0: #if vehicle not simulated that means it is always a leader
#            curfollowers = np.unique(LCinfo[:,2]) #all unique followers of leader. NOTE THAT JUST BECAUSE IT IS A FOLLOWER DOES NOT MEAN ITS SIMULATED
            leaders.append(i)


    #explanation of how makeplatoon works:
    #the main problem you can run into when forming platoons is what I refer to as "circular dependency". this occurs when veh X has veh Y as BOTH a leader AND follower.
    #This is a problem because veh X and Y depend on each other, and you have to arbitrarily pick one as only a leader in order to resolve the loop
    #this can occur when a follower overtakes a leader. I'm not sure how common that is, but in the interest of generality we will handle these cases.
    #the other thing to keep in mind is related to efficiency. since we need to iterate over lists of vehicles twice to form platoons (O(n**2)), if the lists of candidate
    #vehicles, or leaders, becomes long, forming the platoons can potentially be very inefficient. size m vehicle platoon, size n lists, potentially m*n**2.
    #however, if you are smart about how you form the platoons you can keep the follower list and leader list fairly short, so n will be small.
    #We want to take out leaders once all their followers are in platoons. this keeps the leader list short.
    #to keep the follower list short, you need to check for circular dependency, and also search depth first instead of breadth. (e.g. try to go deep in single lane)
    #end explanation

    #now all the "pre-processing" has been completed.

    #now we will initialize the platoon formationation algorithm

    #first make sure there are vehicles which can be used as lead vehicles
    #actually you don't necessarily have to do this because you can resolve the circular dependency.
    if len(leaders)==0:
        print('no leaders identified in data. data is either loaded incorrectly, or lead vehicles have circular dependency')
        print('we will automatically get a leader')
        newlead = None
        newleadt_nstar= float('inf')#initialized as arbitrarily large value
        for i in vehlist:
            if platooninfo[i][0] < newleadt_nstar:
                newleadt_nstar = platooninfo[i][0]
                newlead = i
        platooninfo[newlead][1] = platooninfo[newlead][0] #define the new leader as a vehicle that is never simulated
        platooninfo[newlead][2] = platooninfo[newlead][0] #define the new leader as a vehicle that is never simulated
        leaders.append(newlead)






    curlead = None #initialize curlead as None

    for i in leaders: #this will ensure that every vehicle designed as a leader has no leaders and no vehicle has a follower which is designated as a leader.
        chklead = platooninfo[i][4].copy() #copy because we will potentially be modifying this pointer
        for j in chklead:
            platooninfo[j][-1].remove(i) #remove leader from the follower list of the leader's leaders; meaning the leader is not a follower
            platooninfo[i][4].remove(j) #the leader should not have any leaders.

    #want to make sure there are not any leaders with no followers since we don't want that.
    leadersc = leaders.copy()
    for i in leadersc:#initialize the simulation with the trajectory of the leaders
        if len(platooninfo[i][-1]) ==0: #in case there are any leaders without followers #probably don't need this if but meh we'll leave it in
            leaders.remove(i)

    simcount = 0 #simcount keeps track of how many vehicles still have followers we can simulate. Once no vehicles have followers we can simulate, it means
    #we must have simulated every vehicle that we can (thus our job is done)
    for i in vehlist:
        if len(platooninfo[i][-1])>0:
            simcount += 1

    totfollist = []
    followers = []
    curleadlist = []
        #we need to initialize totfollist before we move into the while loop
    for i in leaders:
        #add in all the followers into totfollist, unless i = curlead
#        if i == curlead:  #don't need this part when it's in makeplatooninfo because curlead is just None
#            continue #don't append the thing if i == curlead because it will be done in the beginning of the while loop
        for j in platooninfo[i][-1]:
            totfollist.append(j) #append all of the followers for vehicle i
    totfollist = list(set(totfollist)) #make list of followers unique


     #We return the objects we constructed, and can begin to form the platoons iteratively using makeplatoon.
    return meas, platooninfo, leaders, simcount, curlead, totfollist, followers, curleadlist

def checksequential(data, dataind = 1, pickfirst = False):
#	checks that given data are all sequential in time (i.e. each row of data advances one frameID)
	#If the data are not sequential, it finds all different sequential periods, and returns the longest one
	#if pickfirst is true, it will always pick the first interval.
	#This function is used by both makeplatooninfo and makefollowerchain to check if we have a continuous period of having a leader.
	#This essentially is using the np.nonzero function
	#note - assumes the data is arranged so that higher row indices correspond to later times
	#input-
	#    data: nparray, with quantity of interest in column dataind i.e. data[:,dataind]
	#    data is arranged so that data[:,dataind] has values that increase by 1 [0,1,2, ...] BUT THERE MAY BE JUMPS e.g. [1 3] or [1 10000] are jumps. [1 2 3 4] is sequential
	#    we "check the data is sequential" by finding the longest period with no jumps. return the sequential data, and also the indices for sequential data (indjumps)
	#    data[indjumps[0]:indjumps[1],:] would give the sequential data.
	#    dataind: the column index of the data that we will be checking is sequential
	#
	#    pickfirst = False: if True, we will always pick the first sequential period, even if it isn't the longest. Otherwise, we always pick the longest.
	#
	#
	#output -
	#    sequentialdata: the sequential part of the data. i.e. the data with only the sequential part returned
	#    indjumps: the indices of the jumps
	#    note that indjumps[seqind]:indjumps[seqind+1] gives correct slice indexing. So the actual times indices are indjumps[seqind], indjumps[seqind+1]-1


    #returns data which is sequential and the slice indices used to give this modified data
    #note that indjumps[seqind]:indjumps[seqind+1] gives correct slice indexing. So the actual times indices are indjumps[seqind], indjumps[seqind+1]-1
    l = data.shape[0] #data must be an np array with rows as observations! cannot be a list/regular array or have columns as observations

    if l ==0: #check if data is empty. if this happens we return the special value of [0,0] for indjumps. [0,0] cannot be returned any other way.
        indjumps = [0,0]
        return data, indjumps
    if (-data[0,dataind]+data[-1,dataind]+1) == l: #very quick check if data is totally sequential we can return it immediately
        indjumps = [0,l]
        return data, indjumps
    if l <= 10:
        pass
        # print('warning: very short measurements') #this is just to get a feel for how many potentially weird simulated vehicles are in a platoon.
        #if you get a bunch of print out it might be because the data has some issue or there may be a bug in other code
    timejumps = data[1:l,dataind] - data[0:l-1,dataind]
    timejumps = timejumps - 1 #timejumps is nonzero only if there is a gap in the datastream
    indjumps = np.nonzero(timejumps) #non-zero indices of timejumps
    lenmeas = np.append(indjumps, [l-1]) - np.insert(indjumps,0,-1) #array containing number of measurements in each sequential period
    seqind = np.argmin(-lenmeas) #gets index of longest sequential period
    indjumps = indjumps[0] + 1 #prepare indjumps so we can get the different time periods from it easily
    indjumps = np.append(indjumps, l)
    indjumps = np.insert(indjumps,0,0)

    if pickfirst: #pick first = true always returns the first sequential period, regardless of length. defaults to false
        data = data[indjumps[0]:indjumps[1],:]
        return data, indjumps[[0, 1]]

    data = data[indjumps[seqind]:indjumps[seqind+1],:]
    #i don't know why I return only indjumps with specific seqind instead of just all of indjumps. But that's how it is so I will leave it
    return data, indjumps[[seqind, seqind+1]]

def sequential(data, dataind = 1, slices_format = True):

    #returns indices where data is no longer sequential after that index. So for example for indjumps = [5,10,11] we have data[0:5+1], data[5+1:10+1], data[10+1:11+1], data[11+1:]
    #as being sequential
    #if slices_format = True, then it returns the indjumps so that indjumps[i]:indjumps[i+1] gives the correct slices notation
    l = data.shape[0] #data must be an np array with rows as observations! cannot be a list/regular array or have columns as observations

    if l ==0: #check if data is empty. if this happens we return the special value of [0,0] for indjumps. [0,0] cannot be returned any other way.
        #this should probably return None instead
        indjumps = [0,0]
        return indjumps
    if (-data[0,dataind]+data[-1,dataind]+1) == l: #very quick check if data is totally sequential we can return it immediately
        indjumps = [0,l]
        return indjumps

    timejumps = data[1:l,dataind] - data[0:l-1,dataind]
    timejumps = timejumps - 1 #timejumps is nonzero only if there is a gap in the datastream
    indjumps = np.nonzero(timejumps) #non-zero indices of timejumps


    if slices_format:
        indjumps = indjumps[0] + 1 #prepare indjumps so we can get the different time periods from it easily
        indjumps = np.append(indjumps, l)
        indjumps = np.insert(indjumps,0,0)

    return indjumps

def indtotimes(indjumps,data, dataind = 1):
    #takes indjumps, data and returns the times corresponding to the boundaries of the sequential periods
    #there are len(indjumps) - 1 total blocks, and each block is characterized as a tuple of two values of the beginning and ending time + 1

    #if you want to go from times back to inds it's easy, first you subtract off the start time, then you add the starting indices of the start time (included as third in the tuple )
    out = []
    for i in range(len(indjumps)-1):
        starttime = data[indjumps[i],dataind]
        endtime = data[indjumps[i+1]-1,dataind]+1
        temp = (starttime, endtime, indjumps[i]) #indjumps[i] included to keep track of how many measurements total
        out.append(temp)

    return out

def indtopos(indjumps, data, dataind = 2):
    #takes indjumps, data and returns the positions corresponding to boundaries of sequential periods
    #each sequential section of data has a len 2 tuple of beginning, ending position
    out = []
    for i in range(len(indjumps)-1):
        startpos  = data[indjumps[i],dataind]
        endpos = data[indjumps[i+1]-1,dataind]
        temp = (startpos, endpos)
        out.append(temp)
    return out


def makeleadinfo(platoon, platooninfo, sim, *args):
    #gets ALL lead info for platoon
    #requires platooninfo as input as well
    #leaders are computed ONLY OVER SIMULATED TIMES (t_n - T_nm1).
    #requires platooninfo so it can know the simulated times.
    #also requires sim to know the leader at each observation

    #may be faster to use np.unique with return_inverse and sequential

    #EXAMPLE:
#    platoon = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
#
#    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20.
#    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i].)
    leadinfo = []
    for i in platoon: #iterate over each vehicle in the platoon
        curleadinfo = [] #for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon

        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4] #get times for current vehicle
        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
        curlead = leadlist[0] #initialize current leader
        curleadinfo.append([curlead, t_n]) #initialization
        for j in range(len(leadlist)):
            if leadlist[j] != curlead: #if there is a new leader
                curlead = leadlist[j] #update the current leader
                curleadinfo[-1].append(t_n+j-1) #last time (in frameID) the old leader is observed
                curleadinfo.append([curlead,t_n+j]) #new leader and the first time (in frameID) it is observed.
        curleadinfo[-1].append(t_n+len(leadlist)-1) #termination

        leadinfo.append(curleadinfo)

    return leadinfo

def makefolinfo(platoon, platooninfo, sim, *args, allfollowers = True, endtime = 'Tn'):
    #same as leadinfo but it gives followers instead of leaders.
    #followers computed ONLY OVER SIMULATED TIMES + BOUNDARY CONDITION TIMES (t_n - T_nm1 + T_nm1 - T_n)
    #allfollowers = True -> gives all followers, even if they aren't in platoon
    #allfollowers = False -> only gives followers in the platoon (needed for adjoint calculation, adjoint variables depend on followers, not leaders.)
    #endtime = 'Tn' calculates followers between t_n, T_n, otherwise calculated between t_n, T_nm1,
    #so give endtime = 'Tnm1' and it will not compute followers over boundary condition times

    #EXAMPLE
    ##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
##    allfollowers = False-
##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers IN THE PLATOON
    #all followers = True -
#    [[[7,11,20]], [[8, 11, 15],[9,16,20]]] #vehicle 7 has 8 and 9 as followers
    folinfo = []
    for i in platoon:
        curfolinfo = []
        t_nstar, t_n, T_nm1, T_n= platooninfo[i][0:4]
        if endtime == 'Tn':
            follist = sim[i][t_n-t_nstar:T_n-t_nstar+1,5] #list of followers
        else:
            follist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,5]
        curfol = follist[0]
        if curfol != 0: #you need to get the time of the follower to make sure it is actually being simulated in that time
            #(this is for the case when allfollower = False)
            foltn = platooninfo[curfol][1]
        else:
            foltn = math.inf
        unfinished = False

        if allfollowers and curfol != 0:
            curfolinfo.append([curfol,t_n])
            unfinished = True
        else:
            if curfol in platoon and t_n >= foltn: #if the current follower is in platoons we initialize
                curfolinfo.append([curfol,t_n])
                unfinished = True

        for j in range(len(follist)): #check what we just made to see if we need to put stuff in folinfo
            if follist[j] != curfol: #if there is a new follower
                curfol = follist[j]
                if curfol != 0:
                    foltn = platooninfo[curfol][1]
                else:
                    foltn = math.inf
                if unfinished: #if currrent follower entry is not finished
                    curfolinfo[-1].append(t_n+j-1) #we finish the interval
                    unfinished = False
                #check if we need to start a new fol entry
                if allfollowers and curfol != 0:
                    curfolinfo.append([curfol,t_n+j])
                    unfinished = True
                else:
                    if curfol in platoon and t_n+j >= foltn: #if new follower is in platoons
                        curfolinfo.append([curfol,t_n+j]) #start the next interval
                        unfinished = True
        if unfinished: #if currrent follower entry is not finished
            curfolinfo[-1].append(t_n+len(follist)-1) #finish it

        folinfo.append(curfolinfo)



    return folinfo

def makeleadfolinfo(platoons, platooninfo, sim, *args, relaxtype = 'both', mergertype = 'avg', merge_from_lane = 7, merge_lane = 6):
    #new makeleadfolinfo function which integrates the previous versions
    #inputs -
    #platoons : platoon you want to calibrate
    #platooninfo - output from makeplatooninfo
    #meas - measurements in usual format
    #relaxtype = 'pos', 'neg', 'both', 'none'  - choose between positive, negative, and pos/negative relaxation amounts added. 'none' is no relax.
    #mergertype = 'avg', 'none', 'remove'- 'avg' calculates the relaxation amount using average headway
    #if 'none' will not get merger relaxation amounts, but NOTE that some mergers are actually treated as lane changes and these are still kept.
    #if 'remove' will actually go through and remove those mergers treated as lane changes (this happens when you had a leader in the on-ramp, and then merged before your leader)
    #merge_from_lane = 7 - if using merger anything other than 'none', you need to specify the laneID corresponding to the on-ramp
    #merge_lane = 6 - if using merger anything other than 'none' you need to specify the laneID you are merging into

    #outputs -
    #leadinfo - list of lists with the relevant lead info (lists of triples leadID, starttime, endtime )
    #leadinfo lets you get the lead vehicle trajectory of a leader in a vectorized way.
    #folinfo - same as leadinfo, but for followers instead of leaders.
    #rinfo - gets times and relaxation amounts. Used for implementing relaxation phenomenon, which prevents
    #unrealistic behaviors due to lane changing, and improves lane changing dynamics

##EXAMPLE:
##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
##
##    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20.
##    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i]. leadinfo[i] is a list of lists, so leadinfo is a list of lists of lists.)
##
##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers in the platoon

    #legacy info-
    #makeleadfolinfo_r - 'pos' 'none'
    #makeleadfolinfo_r2 - 'neg', 'none'
    #makeleadfolinfo_r3 - 'both', 'none'
    #makeleadfolinfo_r4 - 'both', 'avg'
    #makeleadfolinfo_r5 - 'both', 'last'
    #makeleadfolinfo_r6 - 'both', 'remove'

    #in the original implementation everything was done at once which I guess saves some work but makes it harder to change/develop.
    #in this refactored version everything is modularized which is a lot nicer but slower. These functions are for calibration, and for calibration
    #all the time (>99.99%) is spent simulating, the time you spend doing makeleadfolinfo is neglible. Hence this design makes sense.

    leadinfo = makeleadinfo(platoons, platooninfo, sim)
    folinfo = makefolinfo(platoons, platooninfo, sim, allfollowers = False)
    rinfo = makerinfo(platoons, platooninfo, sim, leadinfo, relaxtype = relaxtype, mergertype = mergertype, merge_from_lane = merge_from_lane, merge_lane = merge_lane)

    return leadinfo, folinfo, rinfo

def makerinfo(platoons, platooninfo, sim, leadinfo, relaxtype = 'both',mergertype = 'avg', merge_from_lane = 7, merge_lane = 6):
    """Rule for merger is not consistent with newer relaxation."""
    if relaxtype =='none':
        return [[] for i in platoons]

    rinfo = []
    for i in platoons:
        currinfo = []
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        leadlist = sim[i][t_n-t_nstar:T_nm1-t_nstar+1,4] #this gets the leaders for each timestep of the current vehicle\
        curlead = leadlist[0]

        for j in range(len(leadlist)):
            if leadlist[j] != curlead:
                newlead = leadlist[j]
                oldlead = curlead

                #####relax constant calculation
                newt_nstar = platooninfo[newlead][0]
                oldt_nstar = platooninfo[oldlead][0]
                try:
                    olds = sim[oldlead][t_n+j-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j-1; this is the headway
                except:
                    olds = sim[oldlead][t_n+j-1-oldt_nstar,2] - sim[oldlead][0,6] - sim[i][t_n+j-t_nstar,2] + .1*sim[oldlead][t_n+j-1-oldt_nstar,3]

                news = sim[newlead][t_n+j-newt_nstar,2] - sim[newlead][0,6] - sim[i][t_n+j-t_nstar,2] #the time is t_n+j
                ########

                #pos/neg relax amounts
                gam = olds - news
                if relaxtype == 'both':
                    currinfo.append([t_n+j,gam])

                    if mergertype == 'remove':
                        if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane:
                            currinfo.pop(-1)
                elif relaxtype == 'pos':
                    if gam > 0:
                        currinfo.append([t_n+j,gam])

                        if mergertype == 'remove':
                            if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane:
                                currinfo.pop(-1)
                elif relaxtype == 'neg':
                    if gam < 0:
                        currinfo.append([t_n+j,gam])

                        if mergertype == 'remove':
                            if sim[i][t_n+j-t_nstar,7]==merge_lane and sim[i][t_n+j-1-t_nstar,7]==merge_from_lane:
                                currinfo.pop(-1)

                curlead = newlead
        rinfo.append(currinfo)
    #merger cases
    if mergertype == 'avg':
        rinfo = merge_rconstant(platoons,platooninfo,sim,leadinfo,rinfo,200, merge_from_lane, merge_lane)

    return rinfo


def merge_rconstant(platoons, platooninfo, sim, leadinfo, rinfo, relax_constant = 100,merge_from_lane= 7,merge_lane = 6, datalen =9,h=.1):
    for i in range(len(platoons)):
        curveh = platoons[i]
        t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
        lanelist = np.unique(sim[curveh][:t_n-t_nstar,7])


        if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][t_n-t_nstar,7]==merge_lane: #get a merge constant when a vehicle's simulation starts when they enter the highway #
            lead = np.zeros((T_n+1-t_n,datalen)) #initialize the lead vehicle trajectory
            for j in leadinfo[i]:
                curleadid = j[0] #current leader ID
                leadt_nstar = int(sim[curleadid][0,1]) #t_nstar for the current lead, put into int
                lead[j[1]-t_n:j[2]+1-t_n,:] = sim[curleadid][j[1]-leadt_nstar:j[2]+1-leadt_nstar,:] #get the lead trajectory from simulation
            headway =  lead[:,2]-sim[curveh][t_n-t_nstar:,2]-lead[:,6]
            headway = headway[:T_nm1+1-t_n]
            #calculate the average headway when not close to lane changing events
            headwaybool = np.ones(len(headway),dtype=bool)
            for j in rinfo[i]:
                headwaybool[j[0]-t_n:j[0]-t_n+relax_constant]=0

            headway = headway[headwaybool]
            if len(headway)>0: #if there are any suitable headways we can use then do it
                preheadway = np.mean(headway)

                postlead = sim[curveh][t_n-t_nstar,4]
                postleadt_nstar = platooninfo[postlead][0]

                posthd = sim[postlead][t_n-postleadt_nstar,2]-sim[postlead][t_n-postleadt_nstar,6]-sim[curveh][t_n-t_nstar,2]

                curr = preheadway - posthd
                rinfo[i].insert(0,[t_n,curr])
            #another strategy to get the headway in the case that there aren't any places we can estimate it from

            else:
                #it never reaches this point unless in a special case
                leadlist = np.unique(sim[curveh][:t_n-t_nstar,4])
                if len(leadlist)>1 and 0 in leadlist: #otherwise if we are able to use the other strategy then use that

                    cursim = sim[curveh][:t_n-t_nstar,:].copy()
                    cursim = cursim[cursim[:,7]==merge_from_lane]
                    cursim = cursim[cursim[:,4]!=0]

                    curt = cursim[-1,1]
                    curlead = cursim[-1,4]
                    leadt_nstar = platooninfo[curlead][0]

                    prehd = sim[curlead][curt-leadt_nstar,2]-sim[curlead][curt-leadt_nstar,6]-cursim[-1,2] #headway before

                    postlead = sim[curveh][t_n-t_nstar,4]
                    postleadt_nstar = platooninfo[postlead][0]

                    posthd = sim[postlead][t_n-postleadt_nstar,2]-sim[postlead][t_n-postleadt_nstar,6]-sim[curveh][t_n-t_nstar,2]
                    curr=prehd-posthd

                    rinfo[i].insert(0,[t_n,curr])

                else: #if neither strategy can be used then we can't get a merger r constant for the current vehicle.
                    continue

    return rinfo


def arraytraj(meas,followerchain, presim = False, postsim = False,datalen=9):
    #puts output from makefollerchain/makeplatooninfo into a dict where the key is frame ID, value is array of vehicle and their position, speed
    #we can include the presimulation (t_nstar to t_n) as well as postsimulation (T_nm1 to T_n) but including those are optional
    #this will put in the whole trajectory based off of the times in followerchain
    t = 1 #index for first time
    T = 2 #index for last time
    if presim: #if presim is true we want the presimulation, so we start with t_nstar
        t = 0
    if postsim:
        T = 3
    t_list = [] #t_n list
    T_list = [] #T_n list
    for i in followerchain.values():
        t_list.append(i[t])
        T_list.append(i[T])

    T_n = int(max(T_list))  #T_{n} #get maximum T
    t_1 = int(min(t_list)) #t_1 #get minimum T
    mytime = range(t_1, T_n+1) #time range of data
    platoontraj = {k: np.empty((0,datalen)) for k in mytime}  #initialize output
    #fix this part, also make something to use this output to fix the lead/follow stuff in ngsim data
    for i in followerchain.keys():

        curmeas = meas[i] #i is the vehicle id, these are the current measuremeents
        t_nstar = followerchain[i][0]
        t_n = followerchain[i][t] #first time we are using for current vehicle
        T_n = followerchain[i][T] #last time we are using for current vehicle
        curmeas = curmeas[t_n-t_nstar:T_n-t_nstar+1,:] #get only the times requested
        curtime = range(t_n,T_n+1)
        count = 0
        for j in curtime:
            platoontraj[j] = np.append(platoontraj[j], np.reshape(curmeas[count,:],(1,datalen)), axis=0)
            count += 1
    return platoontraj, mytime


def platoononly(platooninfo, platoon):
    #makes it so platooninfo only contains entries for platoon
    #platoon can be a platoon in form [ 1, 2, 3, etc.] or a list of those.
    ans = {}

    if type(platoon[0]) == list:  #list of platoons
        useplatoon = []
        for i in platoon:
            useplatoon.extend(i[:])
    else:
        useplatoon = platoon

    for i in useplatoon[:]:
        ans[i] = platooninfo[i]
    return ans



def greedy_set_cover(subsets, parent_set):
    #copy pasted from stack exchange
    parent_set = set(parent_set)
    max = len(parent_set)
    # create the initial heap. Note 'subsets' can be unsorted,
    # so this is independent of whether remove_redunant_subsets is used.
    heap = []
    for s in subsets:
        # Python's heapq lets you pop the *smallest* value, so we
        # want to use max-len(s) as a score, not len(s).
        # len(heap) is just proving a unique number to each subset,
        # used to tiebreak equal scores.
        heapq.heappush(heap, [max-len(s), len(heap), s])
    results = []
    result_set = set()
    while result_set < parent_set:
#        logging.debug('len of result_set is {0}'.format(len(result_set)))
        best = []
        unused = []
        while heap:
            score, count, s = heapq.heappop(heap)
            if not best:
                best = [max-len(s - result_set), count, s]
                continue
            if score >= best[0]:
                # because subset scores only get worse as the resultset
                # gets bigger, we know that the rest of the heap cannot beat
                # the best score. So push the subset back on the heap, and
                # stop this iteration.
                heapq.heappush(heap, [score, count, s])
                break
            score = max-len(s - result_set)
            if score >= best[0]:
                unused.append([score, count, s])
            else:
                unused.append(best)
                best = [score, count, s]
        add_set = best[2]
#        logging.debug('len of add_set is {0} score was {1}'.format(len(add_set), best[0]))
        results.append(add_set)
        result_set.update(add_set)
        # subsets that were not the best get put back on the heap for next time.
        while unused:
            heapq.heappush(heap, unused.pop())
    return results


def fin_dif_wrapper(p,args, *eargs, eps = 1e-8, **kwargs):
    #returns the gradient for function with call signature obj = objfun(p, *args)
    #note you should pass in 'objfun' as the last entry in the tuple for args
    #so objfun = args[-1]
    #uses first order forward difference with step size eps to compute the gradient
    out = np.zeros((len(p),))
    objfun = args[-1]
    obj = objfun(p,*args)
    for i in range(len(out)):
        curp = p.copy()
        curp[i] += eps
        out[i] = objfun(curp,*args)
    return (out-obj)/eps


def approx_hess(p, args, kwargs, eps = 1e-8, gradfn = fin_dif_wrapper, curgrad = None, **unused):
    #input the current point p, function to calculate the gradient gradfn with call signature
    #grad = gradfn(p,*args,*kwargs)
    #and we will compute the hessian using a forward difference approximation.
    #this will use n+1 gradient evaluations to calculate the hessian.
    #you can pass in the current grad if you have it, this will save 1 gradient evaluation.
    n = len(p)
    hess = np.zeros((n,n))
    if curgrad is None:
        grad = gradfn(p,*args,*kwargs) #calculate gradient for the unperturbed parameters
    else:
        grad = curgrad
#    grad = np.asarray(grad) #just pass in a np array not a list...if you want to pass in list then you need to convert to np array here.
    for i in range(n):
        pe = p.copy()
        pe[i] += eps #perturbed parameters for parameter n
        gradn = gradfn(pe,*args,**kwargs) #gradient for perturbed parameters
#        gradn = np.asarray(gradn) #just pass in a np array not a list...if you want to pass in list then you need to convert to np array here.
        hess[:,i] = gradn-grad #first column of the hessian without the 1/eps

    hess = hess*(1/eps)
    hess = .5*(hess + np.transpose(hess))
    return hess


def chain_metric(platoon, platooninfo, meas, k=.9, metrictype='lead'):
    #metric that defines how good a platoon is
    #refer to platoon formation pdf for exact definition
    res = 0
    for i in platoon:
        T = set(range(platooninfo[i][1], platooninfo[i][2]+1))
        res += c_metric(i, platoon, T, platooninfo, meas, k=k, metrictype=metrictype)
    return res


def c_metric(veh, platoon, T, platooninfo, meas, k=.9, metrictype='lead', depth=0, targetsdict = {}):
    #defines how good a single vehicle in a specific time is.
    #refer to platoon formation pdf for exact definition

    veh = int(veh)
    if veh in targetsdict:
        targetsList = targetsdict[veh]
    else:
        if metrictype == 'lead':
            leadinfo = makeleadinfo([veh], platooninfo, meas)
            targetsList = leadinfo[0]
        else:
            folinfo = makefolinfo([veh], platooninfo, meas)
            # targetsList = folinfo[0]
            temp = folinfo[0]
            targetsList = []
            for i in temp:
                Tnstart = platooninfo[i[0]][1]
                Tnm1 = platooninfo[i[0]][2]
                start = max(Tnstart, i[1])
                end = min(Tnm1, i[2])
                if start<end:
                    targetsList.append([i[0], start, end])
        targetsdict[veh] = targetsList

    res = len(getL(veh, platoon, T, targetsList))
    leads = getLead(veh, platoon, T, targetsList)

    for i in leads:
        res += k * c_metric(i, platoon, getTimes(veh, i, T, targetsList), platooninfo, meas, k=k, metrictype=metrictype, depth=depth + 1,targetsdict = targetsdict)
    return res

def getL(veh, platoon, T, targetsList):
    L = set([])
    temp = set([])
    for i in targetsList:
        if i[0] not in platoon:
            continue
        temp.update(range(i[1], i[2]+1))
    L = T.intersection(temp)
    return L

def getTimes(veh, lead, T, targetsList):
    temp = set([])
    for i in targetsList:
        if i[0] == lead:
            temp.update(range(i[1], i[2]+1))
    temp = T.intersection(temp)
    return temp

def getLead(veh, platoon, T, targetsList):
    leads = []
    for i in targetsList:
        if i[0] in platoon and (i[1] in T or i[2] in T):
            leads.append(i[0])
    leads = list(set(leads))
    return leads



def cirdep_metric(platoonlist, platooninfo, meas, k=.9, metrictype='veh'):
    #platoonlist - list of platoons

    #type = veh checks for circular dependencies. For every vehicle which is
    #causing a circular dependency it outputs:
    # tuple of [vehicle, list of lead vehicles, list of lead vehicles platoon indices], vehicle platoon index
    #where vehicle is the vehicle with the circular dependency, which it has becuase of lead vehicles.

    #type = num quantifies how bad a circular dependency is by computing
    #the change to chain metric when adding the lead vehicle to the platoon
    #with the circular dependency. Output is a list of floats, same length as platoonlist.
    if metrictype == 'veh':
        cirList = []
        after = set([])
        veh2platoon = {} #converts vehicle to platoon index
        for i in range(len(platoonlist)):
            for j in platoonlist[i]:
                veh2platoon[j] = i
        for i in range(len(platoonlist)):
            after.update(platoonlist[i])
        for i in range(len(platoonlist)):
            after -= set(platoonlist[i])
            leadinfo = makeleadinfo(platoonlist[i], platooninfo, meas)
            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                circleadveh = leaders.intersection(after)
                if len(circleadveh)>0:
                    cirList.append(([platoonlist[i][j], list(circleadveh), [veh2platoon[k] for k in circleadveh]], i))
        return cirList
    elif metrictype == 'num':
        res = 0
        cirList = []
        after = set([])
        leader_violate_map = defaultdict(list)
        for i in range(len(platoonlist)): #get set of all vehicles
            after.update(platoonlist[i])
        for i in range(len(platoonlist)): #i is current platoon
            after -= set(platoonlist[i]) #remove vehicles from current platoon
            leadinfo= makeleadinfo(platoonlist[i], platooninfo, meas)
            temp = []

            for j in range(len(platoonlist[i])):
                leaders = [k[0] for k in leadinfo[j]]
                leaders = set(leaders)
                leaders_after = leaders.intersection(after) #leaders_after are any leaders of i which are not yet calibrated
                if len(leaders_after) > 0:
                    # temp.append((list(leaders_after), i))

                    violated_leaders = [v for v in leaders_after if i not in leader_violate_map[v]]
                    if violated_leaders: # Remove duplicated invoking of chain matric
                        temp.append((violated_leaders, i))
                    for l in leaders_after:
                        leader_violate_map[l].append(i)

                else:
                    temp.append(None)
            cirList.append(temp)
        res = []
        for i in cirList:
            if i == None:
                res.append(0)
            else:
                temp = 0
                for j in i:
                    if j:
                        for l in j[0]:
                            T = set(range(platooninfo[l][1], platooninfo[l][3]+1))
                            temp += c_metric(l, platoonlist[j[1]], T, platooninfo, meas=meas, k=k, metrictype='follower')
            res.append(temp)
        return res


def is_pareto_efficient(costs, return_mask = True): #copy pasted from stack exchange. finds pareto front assuming lower is better
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient

def boundaryspeeds(meas, entrylanes, exitlanes, timeind, outtimeind, car_ids=None, my_mod = False):
    #car_ids is a list of vehicle IDs, only use those values in meas

    #entrylanes, exitlanes are list of laneIDs at which the boundary speeds will be calculated
    #i.e. entry speeds are calculated for entrylanes, exit speeds for exitlanes
    #timeind is time (measured in seconds) for data
    #e.g. .1 seconds for ngsim
    #outtimeind is time for output timeseries

    #output is nested list of speeds for each lane in entryspeeds/exitspeeds
    #nested list of (first time, last time) tuples for each lane in entry/exittimes

    #gets entry/exit speeds empirically from trajectory data; documentation is in boundaryspeeds assignment pdf
    #my_mod - calculates the entry headways as well and outputs the resulting flows - DO NOT USE


    # filter meas based on car ids, merge the result into a single 2d array
    if car_ids is None:
        data = np.concatenate(list(meas.values()))
    else:
        data = np.concatenate([meas[car_id] for car_id in car_ids])

    # sort observations based on lane number, then time, then position
    data = data[np.lexsort((data[:, 2], data[:, 1], data[:, -2]))]

    # get the index for the entry/exit data row index for each lane and time
    _, index, count = np.unique(data[:, [-2, 1]], axis=0, return_index=True, return_counts=True)
    index_rev = index + count - 1
    entry_data = data[index]  # all observations for entry speeds
    exit_data = data[index_rev]  # all observations for exit speeds

    # now aggregate the data according to outtimeind / timeind
    interval = outtimeind / timeind
    entryspeeds = list()
    entrytimes = list()
    exitspeeds = list()
    exittimes = list()

    for entrylane in entrylanes:
        # filter entry data according to lane number, then take only 2 columns: time and speed
        entry_data_for_lane = entry_data[entry_data[:, -2] == entrylane]
        if my_mod: #calculates headways as well to output flows
            #you don't want to actually compute flow like this because you will always get an overestimate
            #calculate headways, same length as the speeds
            myheadways = list()
            for i in range(len(entry_data_for_lane)):
                curlead,curtime, vehlen = entry_data_for_lane[i,[4,1,6]]
                if curlead == 0:
                    curhd = 40 #magic number just guess a headway (net headway = tail to tail)
                else:
                    curleadlen, leadt_nstar = meas[curlead][0,[6,1]]
                    curhd = meas[curlead][int(curtime - leadt_nstar), 2] - curleadlen - entry_data_for_lane[i,2] + vehlen
                myheadways.append(curhd)
            #same as for no modification
            entry_data_for_lane = entry_data_for_lane[:,[1,3]]
            entryspeed, entrytime = interpolate(entry_data_for_lane, interval)
            #compute headways using same interpolation function
            entry_data_for_lane[:,1] = myheadways
            entryhd, unused = interpolate(entry_data_for_lane, interval)
            entryspeed = list(np.divide(entryspeed,entryhd)) #flows = speed / (net) headway
        else:
            entry_data_for_lane = entry_data_for_lane[:,[1,3]]
            entryspeed, entrytime = interpolate(entry_data_for_lane, interval)

        entryspeeds.append(entryspeed)
        entrytimes.append(entrytime)

    for exitlane in exitlanes:
        # filter exit data according to lane number, then take only 2 columns: time and speed
        exit_data_for_lane = exit_data[exit_data[:, -2] == exitlane][:, [1, 3]]
        exitspeed, exittime = interpolate(exit_data_for_lane, interval)
        exitspeeds.append(exitspeed)
        exittimes.append(exittime)

    return entryspeeds, entrytimes, exitspeeds, exittimes


def interpolate(data, interval=1.0):
    # entry/exit data: 2d array with 2 columns: time and speed for a lane
    #second column is the one we act on; not necessarily have to be speed
    # interval: aggregation units. = new timeind / old timeind
    # returns: (aggregated_speed_list, (start_time_of_first_interval, start_time_of_last_interval))

    if not len(data):
        return list(), ()
    if len(np.shape(data)) == 1:
        newdata = np.zeros((len(data),2))
        newdata[:,1] = data
        newdata[:,0] = list(range(len(data)))
        data = newdata
    speeds = list()
    cur_ind = 0
    cur_time = data[0, 0]
    remained = interval
    speed = 0.0
    while cur_ind < len(data) - 1:
        if remained + cur_time < data[cur_ind + 1, 0]:
            speed += data[cur_ind, 1] * remained
            cur_time += remained
            remained = 0.0
        else:
            speed += data[cur_ind, 1] * (data[cur_ind + 1, 0] - cur_time)
            remained -= (data[cur_ind + 1, 0] - cur_time)
            cur_time = data[cur_ind + 1, 0]
            cur_ind += 1
        if remained == 0.0:
            speeds.append(speed / interval)
            remained = interval
            speed = 0.0
    speed += remained * data[-1, 1]
    speeds.append(speed / interval)
    return speeds, (data[0, 0], data[0, 0] + (len(speeds) - 1) * interval)


def interp1ds(X,Y,times):
    #given time series data X, Y (such that each (X[i], Y[i]) tuple is an observation),
    #and the array times, interpolates the data onto times.

    #X, Y, and times all need to be sorted in terms of increasing time. X and times need to have a constant time discretization
    #runtime is O(n+m) where X is len(n) and times is len(m)

    #uses 1d interpolation. This is similar to the functionality of scipy.interp1d. Name is interp1ds because it does linear interpolation in 1d (interp1d) on sorted data (s)

    #e.g. X = [1,2,3,4,5]
    #Y = [2,3,4,5,6]
    #times = [3.25,4,4.75]

    #out = [4.25,5,5.75]

    if times[0] < X[0] or times[-1] > X[-1]:
        print('Error: requested times are outside measurements')
        return None

    Xdt = X[1] - X[0]
    timesdt = times[1]-times[0]
    change = timesdt/Xdt

    m = interval_binary_search(X,times[0])
    out = np.zeros(len(times))
    curind = m + (times[0]-X[m])/Xdt

    leftover = curind % 1
    out[0] = Y[m] + leftover*(Y[m+1]-Y[m])

    for i in range(len(times)-1):
        curind = curind + change #update index

        leftover = curind % 1 #leftover needed for interpolation
        ind = int(curind // 1) #cast to int because it is casted to float automatically

        out[i+1] = Y[ind] + leftover*(Y[ind+1]-Y[ind])

    return out


def getentryflows(meas, entrylanes,  timeind, outtimeind):
    #meas - data in normal format
    #entrylanes - list of laneIDs
    #timeind - discretization (in real time) for data
    #outtimeind - discretization (in real time) for output
    times = {i:[] for i in entrylanes}
    for i in meas.keys():
        curtime, lane = meas[i][0,[1,7]]
        if lane not in entrylanes:
            continue
        times[lane].append(curtime)

    interval = outtimeind / timeind
    entryflows = list()
    entrytimes = list()

    for i in entrylanes:
        times[i].sort()

        curdata = np.zeros((int(times[i][-1] - times[i][0]),2)) #initialize output
        curdata[:,0] = range(int(times[i][0]), int(times[i][-1]))
        firsttime = times[i][0]
        for count, j in enumerate(times[i][:-1]):
            nexttime, curtime = times[i][count+1], j
            curflow = 1/((nexttime - curtime)*timeind)

            curdata[int(curtime - firsttime):int(nexttime - firsttime),1] = curflow
        entryflow, entrytime = interpolate(curdata, interval)
        entryflows.append(entryflow)
        entrytimes.append(entrytime)

    return entryflows, entrytimes


def calculateflows(meas, spacea, timea, agg, lane = None, method = 'area', h = .1):
    #meas = measurements, in usual format (dictionary where keys are vehicle IDs, values are numpy arrays
 	#spacea - reads as ``space A'' (where A is the region where the macroscopic quantities are being calculated).
        #list of lists, each nested list is a length 2 list which ... represents the starting and ending location on road.
        #So if len(spacea) >1 there will be multiple regions on the road which we are tracking e.g. spacea = [[200,400],[800,1000]],
        #calculate the flows in regions 200 to 400 and 800 to 1000 in meas.
 	#timea - reads as ``time A'', should be a list of the times (in the local time of thedata).
        #E.g. timea = [1000,3000] calculate times between 1000 and 3000.
 	#agg - aggregation length, float number which is the length of each aggregation interval.
        #E.g. agg = 300 each measurement of the macroscopic quantities is over 300 time units in the data,
        #so in NGSim where each time is a frameID with length .1s, we are aggregating every 30 seconds.
    #h specifies unit conversion - i.e. if 1 index in data = .1 of units you want, h = .1
        #e.g. ngsim has .1 seconds between measurements, so h = .1 yields units of seconds for time. no conversion for space units
    #area method (from laval paper), or flow method (count flow into space region, calculate space mean speed, get density from flow/speed)
        #area method is better

    #for each space region, value is a list of floats of the value at the correpsonding time interval
    q = [[] for i in spacea]
    k = [[] for i in spacea]

    starttime = [i[0,1] for i in meas.values()]
    starttime = int(min(starttime)) #first time index in data

    spacealist = []
    for i in spacea:
        spacealist.extend(i)
    # spacemin = min(spacealist)
    # spacemax = max(spacealist)
    # timemin = min(timea)
    # timemax = max(timea)

    intervals = []  #tuples of time intervals
    start = timea[0]
    end = timea[1]
    temp1 = start
    temp2 = start + agg
    while temp2 < end:
        intervals.append((temp1, temp2))
        temp1 = temp2
        temp2 += agg
    intervals.append((temp1, end))


    regions = [[([], []) for j in intervals] for i in spacea]
    #regions are indexed by space, then time. values are list of (position traveled, time elapsed) (list of float, list of float)

    flows = [[0 for j in intervals] for i in spacea] #used if method = 'flow', indexed by space, then time, int of how many vehicles enter region
    for vehid in meas:
        alldata = meas[vehid]

        #if lane is given we need to find the segments of data inside the lane
        if lane is not None:
            alldata = alldata[alldata[:,7]==lane] #boolean mask selects data inside lane
            inds = sequential(alldata) #returns indexes where there are jumps
            indlist = []
            for i in range(len(inds)-1):
                indlist.append([inds[i], inds[i+1]])
        else: #otherwise can just use everything
            indlist = [[0,len(alldata)]]

        for i in indlist:
            data = alldata[i[0]:i[1]] #select only current region of data - #sequential data for a single vehicle in correct lane if applicable
            if len(data) == 0:
                continue
#            region_contained = []
#            region_data = {}  # key: tid, sid

            for i in range(len(intervals)):
                start =  int(max(0, intervals[i][0] + starttime - data[0,1])) #indices for slicing data
                end = int(max(0, intervals[i][1] + starttime - data[0,1])) #its ok if end goes over for slicing - if both zero means no data in current interval

                if start == end:
                    continue
                curdata = data[start:end]

                for j in range(len(spacea)):
                    minspace, maxspace = spacea[j][0], spacea[j][1]
                    curspacedata = curdata[np.all([curdata[:,2] > minspace, curdata[:,2] < maxspace], axis = 0)]
                    if len(curspacedata) == 0:
                        continue
                    regions[j][i][0].append(curspacedata[-1,2] - curspacedata[0,2])
                    regions[j][i][1].append((curspacedata[-1,1] - curspacedata[0,1])*h)
                    if method == 'flow':
                        firstpos, lastpos = curdata[0,2], curdata[-1,2]
                        if firstpos < spacea[j][0] and lastpos > spacea[j][0]:
                            flows[j][i] += 1

    if method == 'area':
        for i in range(len(spacea)):
            for j in range(len(intervals)):
                area = (spacea[i][1] - spacea[i][0]) * (intervals[j][1] - intervals[j][0])
                q[i].append(sum(regions[i][j][0]) / area)
                k[i].append(sum(regions[i][j][1]) / area)
    elif method == 'flow':
        for i in range(len(spacea)):
            for j in range(len(intervals)):
                q[i].append(flows[i][j] / (h*(intervals[j][1] - intervals[j][0])))
                try:
                    k[i].append(sum(regions[i][j][0]) / sum(regions[i][j][1]))
                except:
                    k[i].append(0) #division by zero when region is empty

    return q, k


def r_constant(currinfo, frames, T_n, rp, adj = True, h = .1):
	#currinfo - output from makeleadfolinfo_r*
	#frames - [t_n, T_nm1], a list where the first entry is the first simulated time and the second entry is the last simulated time
	# T_n - last time vehicle is observed
	#rp - value for the relaxation, measured in real time (as opposed to discrete time)
	#adj = True - can output needed values to compute adjoint system
	#h = .1 - time discretization

    #given a list of times and gamma constants (rinfo for a specific vehicle = currinfo) as well as frames (t_n and T_nm1 for that specific vehicle) and the relaxation constant (rp). h is the timestep (.1 for NGSim)
    #we will make the relaxation amounts for the vehicle over the length of its trajectory
    #rinfo is precomputed in makeleadfolinfo_r. then during the objective evaluation/simulation, we compute these times.
    #note that we may need to alter the pre computed gammas inside of rinfo; that is because if you switch mutliple lanes in a short time, you may move to what looks like only a marginally shorter headway,
    #but really you are still experiencing the relaxation from the lane change you just took
    if len(currinfo)==0:
        relax = np.zeros(T_n-frames[0]+1)
        return relax, relax #if currinfo is empty we don't have to do anything

    out = np.zeros((T_n-frames[0]+1,1)) #initialize relaxation amount for the time between t_n and T_n
    out2 = np.zeros((T_n-frames[0]+1,1))
    outlen = 1

    maxind = frames[1]-frames[0]+1 #this is the maximum index we are supposed to put values into because the time between T_nm1 and T_n is not simulated. Plus 1 because of the way slices work.
    if rp<h: #if relaxation is too small for some reason
        rp = h #this is the smallest rp can be
#    if rp<h: #if relaxation is smaller than the smallest it can be #deprecated
#        return out, out2 #there will be no relaxation

    mylen = math.ceil(rp/h)-1 #this is how many nonzero entries will be in r each time we have the relaxation constant
    r = np.linspace(1-h/rp,1-h/rp*(mylen),mylen) #here are the relaxation constants. these are determined only by the relaxation constant. this gets multipled by the 'gamma' which is the change in headway immediately after the LC

    for i in range(len(currinfo)): #frames[1]-frames[0]+1 is the length of the simulation; this makes it so it will be all zeros between T_nm1 and T_n
        entry = currinfo[i] #the current entry for the relaxation phenomenon
        curind = entry[0]-frames[0] #current time is entry[0]; we start at frames[0] so this is the current index
        for j in range(outlen):
            if out2[curind,j] == 0:
                if curind+mylen > maxind: #in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)
                    out[curind:maxind,j] = r[0:maxind-curind]
                    out2[curind:maxind,j] = currinfo[i][1]
                else: #this is the normal case
                    out[curind:curind+mylen,j] = r
                    out2[curind:curind+mylen,j] = currinfo[i][1]
                break

        else:
            newout = np.zeros((T_n-frames[0]+1,1))
            newout2 = np.zeros((T_n-frames[0]+1,1))


            if curind+mylen > maxind: #in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)
                newout[curind:maxind,0] = r[0:maxind-curind]
                newout2[curind:maxind,0] = currinfo[i][1]
            else: #this is the normal case
                newout[curind:curind+mylen,0] = r
                newout2[curind:curind+mylen,0] = currinfo[i][1]

            out = np.append(out,newout,axis=1)
            out2 = np.append(out2,newout2,axis=1)
            outlen += 1

    #######calculate relaxation amounts and the part we need for the adjoint calculation #different from the old way
    relax = np.multiply(out,out2)
    relax = np.sum(relax,1)

    if adj:
        outd = -(1/rp)*(out-1) #derivative of out (note that this is technically not the derivative because of the piecewise nature of out/r)
        relaxadj = np.multiply(outd,out2) #once multiplied with out2 (called gamma in paper) it will be the derivative though.
        relaxadj = np.sum(relaxadj,1)
    else:
        relaxadj = relax

    return relax,relaxadj


