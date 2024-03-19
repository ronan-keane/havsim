"""Helper functions and utilities for loading/manipulating data."""
import numpy as np
import math

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
        curr_frame = dataset_sortframes[indices[count]:indices[count] + counts[count], :]

        # generate min/max index for each lane within this frame
        lane_id_to_indices = {}
        lane_ids, lane_indices, lane_counts = np.unique(curr_frame[:, col_dict['lane']], return_index=True,
                                                        return_counts=True)
        for countlane, lane_id in enumerate(lane_ids):
            lane_id_to_indices[lane_id] = (lane_indices[countlane],
                                           lane_indices[countlane] + lane_counts[countlane])

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
        assert (len(np_vec.shape) == 1)
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
        veh_np = dataset[veh_inds[count]:veh_inds[count] + veh_counts[count]]
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
        self.posmem = StateMem(posmem, start)
        self.speedmem = StateMem(speedmem, start)
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
        self.lcmems = [self.lfolmem, self.rfolmem, self.lleadmem, self.rleadmem]

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
        temp.append((None, self.end + 1))

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
                    longest_end = start_time - 1
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
        return 'saved data for vehicle ' + str(self.vehid)

    def __str__(self):
        """Convert to a str representation."""
        return self.__repr__()


def convert_to_data(vehicle):
    """Converts a (subclassed) Vehicle to VehicleData."""
    # should convert Vehicle objects to their vehid. Need to convert Lanes to some index?
    raise NotImplementedError
    return VehicleData(vehicle)


class StateMem:
    """Let posmem or speedmem etc. be slicable by times."""

    def __init__(self, posmem, start):
        self.data = posmem
        self.start = start
        self.end = start + len(posmem) - 1

    def __getitem__(self, times):
        if type(times) == slice:
            start, stop = times.start, times.stop
            if not start:
                start = self.start
            elif start < self.start:
                start = self.start
            elif start > self.end:
                return []
            if not stop:
                stop = self.end + 1
            elif stop > self.end + 1:
                stop = self.end + 1
            elif stop < self.start + 1:
                return []

            return self.data[start - self.start:stop - self.start]

        else:
            if times < self.start or times > self.end:
                raise IndexError
            return self.data[times - self.start]


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
                stop = self.end + 1
            elif stop > self.end + 1:
                stop = self.end + 1
            elif stop < self.start + 1:
                return []

            startind, stopind = mem_binary_search(data, start, return_ind=True), \
                                mem_binary_search(data, stop, return_ind=True)
            out = []

            timeslist = (start, *(data[i][1] for i in range(startind + 1, stopind + 1)), stop)
            for count, i in enumerate(range(startind, stopind + 1)):
                out.extend([data[i][0]] * (timeslist[count + 1] - timeslist[count]))
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
        return 'VehMem: ' + str(self.data)


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
                stop = self.end + 1
            elif stop > self.end + 1:
                stop = self.end + 1
            elif stop < self.start + 1:
                return []

            startind, stopind = mem_binary_search(data, start, return_ind=True), \
                                mem_binary_search(data, stop, return_ind=True)
            out = []

            timeslist = (start, *(data[i][1] for i in range(startind + 1, stopind + 1)), stop)
            for count, i in enumerate(range(startind, stopind + 1)):
                curveh = data[i][0]
                if curveh is not None:
                    vehdata = vehdict[curveh]
                    out.extend(self.myslice(vehdata, timeslist[count], timeslist[count + 1]))
                else:
                    out.extend([None] * (timeslist[count + 1] - timeslist[count]))
            return out
        else:
            if times < self.start or times > self.end:
                raise IndexError
            vehdata = self.vehdict[mem_binary_search(self.data, times)]
            return self.index(vehdata, times)

    def myslice(self, vehdata, start, stop):
        return vehdata.posmem[start:stop]

    def index(self, vehdata, time):
        return vehdata.posmem[time]

    def __repr__(self):
        return str(self.__getitem__(slice(self.start, self.end + 1)))


class VehMemSpeed(VehMemPosition):
    """Returns speed data instead of position data."""

    def myslice(self, vehdata, start, stop):
        return vehdata.speedmem[start:stop]

    def index(self, vehdata, time):
        return vehdata.speedmem[time]


class VehMemLen(VehMemPosition):
    """Returns lengths instead of position data."""

    def myslice(self, vehdata, start, stop):
        return (vehdata.len,) * (stop - start)

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
    while (end > start + 1):
        mid = (end + start) // 2
        if time >= arr[mid][1]:
            start = mid
        else:
            end = mid

    if return_ind:
        return start
    else:
        return arr[start][0]


def interval_binary_search(X, time):
    # finds index m such that the interval X[m], X[m+1] contains time.
    # X = array
    # time = float
    lo = 0
    hi = len(X) - 1
    m = (lo + hi) // 2
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

    timeslist = (start, *(vehmem[i][1] for i in range(startind + 1, stopind + 1)), stop + 1)
    return [[vehmem[startind + i][0], timeslist[i], timeslist[i + 1]] for i in range(len(timeslist) - 1)]


def checksequential(data, dataind=1, pickfirst=False):
    #	checks that given data are all sequential in time (i.e. each row of data advances one frameID)
    # If the data are not sequential, it finds all different sequential periods, and returns the longest one
    # if pickfirst is true, it will always pick the first interval.
    # This function is used by both makeplatooninfo and makefollowerchain to check if we have a continuous period of having a leader.
    # This essentially is using the np.nonzero function
    # note - assumes the data is arranged so that higher row indices correspond to later times
    # input-
    #    data: nparray, with quantity of interest in column dataind i.e. data[:,dataind]
    #    data is arranged so that data[:,dataind] has values that increase by 1 [0,1,2, ...] BUT THERE MAY BE JUMPS e.g. [1 3] or [1 10000] are jumps. [1 2 3 4] is sequential
    #    we "check the data is sequential" by finding the longest period with no jumps. return the sequential data, and also the indices for sequential data (indjumps)
    #    data[indjumps[0]:indjumps[1],:] would give the sequential data.
    #    dataind: the column index of the data that we will be checking is sequential
    #
    #    pickfirst = False: if True, we will always pick the first sequential period, even if it isn't the longest. Otherwise, we always pick the longest.
    #
    #
    # output -
    #    sequentialdata: the sequential part of the data. i.e. the data with only the sequential part returned
    #    indjumps: the indices of the jumps
    #    note that indjumps[seqind]:indjumps[seqind+1] gives correct slice indexing. So the actual times indices are indjumps[seqind], indjumps[seqind+1]-1

    # returns data which is sequential and the slice indices used to give this modified data
    # note that indjumps[seqind]:indjumps[seqind+1] gives correct slice indexing. So the actual times indices are indjumps[seqind], indjumps[seqind+1]-1
    l = data.shape[
        0]  # data must be an np array with rows as observations! cannot be a list/regular array or have columns as observations

    if l == 0:  # check if data is empty. if this happens we return the special value of [0,0] for indjumps. [0,0] cannot be returned any other way.
        indjumps = [0, 0]
        return data, indjumps
    if (-data[0, dataind] + data[
        -1, dataind] + 1) == l:  # very quick check if data is totally sequential we can return it immediately
        indjumps = [0, l]
        return data, indjumps
    if l <= 10:
        pass
        # print('warning: very short measurements') #this is just to get a feel for how many potentially weird simulated vehicles are in a platoon.
        # if you get a bunch of print out it might be because the data has some issue or there may be a bug in other code
    timejumps = data[1:l, dataind] - data[0:l - 1, dataind]
    timejumps = timejumps - 1  # timejumps is nonzero only if there is a gap in the datastream
    indjumps = np.nonzero(timejumps)  # non-zero indices of timejumps
    lenmeas = np.append(indjumps, [l - 1]) - np.insert(indjumps, 0,
                                                       -1)  # array containing number of measurements in each sequential period
    seqind = np.argmin(-lenmeas)  # gets index of longest sequential period
    indjumps = indjumps[0] + 1  # prepare indjumps so we can get the different time periods from it easily
    indjumps = np.append(indjumps, l)
    indjumps = np.insert(indjumps, 0, 0)

    if pickfirst:  # pick first = true always returns the first sequential period, regardless of length. defaults to false
        data = data[indjumps[0]:indjumps[1], :]
        return data, indjumps[[0, 1]]

    data = data[indjumps[seqind]:indjumps[seqind + 1], :]
    # i don't know why I return only indjumps with specific seqind instead of just all of indjumps. But that's how it is so I will leave it
    return data, indjumps[[seqind, seqind + 1]]


def sequential(data, dataind=1, slices_format=True):
    # returns indices where data is no longer sequential after that index. So for example for indjumps = [5,10,11] we have data[0:5+1], data[5+1:10+1], data[10+1:11+1], data[11+1:]
    # as being sequential
    # if slices_format = True, then it returns the indjumps so that indjumps[i]:indjumps[i+1] gives the correct slices notation
    l = data.shape[
        0]  # data must be an np array with rows as observations! cannot be a list/regular array or have columns as observations

    if l == 0:  # check if data is empty. if this happens we return the special value of [0,0] for indjumps. [0,0] cannot be returned any other way.
        # this should probably return None instead
        indjumps = [0, 0]
        return indjumps
    if (-data[0, dataind] + data[
        -1, dataind] + 1) == l:  # very quick check if data is totally sequential we can return it immediately
        indjumps = [0, l]
        return indjumps

    timejumps = data[1:l, dataind] - data[0:l - 1, dataind]
    timejumps = timejumps - 1  # timejumps is nonzero only if there is a gap in the datastream
    indjumps = np.nonzero(timejumps)  # non-zero indices of timejumps

    if slices_format:
        indjumps = indjumps[0] + 1  # prepare indjumps so we can get the different time periods from it easily
        indjumps = np.append(indjumps, l)
        indjumps = np.insert(indjumps, 0, 0)

    return indjumps


def makeleadinfo(platoon, platooninfo, sim, *args):
    # gets ALL lead info for platoon
    # requires platooninfo as input as well
    # leaders are computed ONLY OVER SIMULATED TIMES (t_n - T_nm1).
    # requires platooninfo so it can know the simulated times.
    # also requires sim to know the leader at each observation

    # may be faster to use np.unique with return_inverse and sequential

    # EXAMPLE:
    #    platoon = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
    #
    #    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20.
    #    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i].)
    leadinfo = []
    for i in platoon:  # iterate over each vehicle in the platoon
        curleadinfo = []  # for each vehicle, we get these and then these are appeneded at the end so we have a list of the info for each vehicle in the platoon

        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]  # get times for current vehicle
        leadlist = sim[i][t_n - t_nstar:T_nm1 - t_nstar + 1,
                   4]  # this gets the leaders for each timestep of the current vehicle\
        curlead = leadlist[0]  # initialize current leader
        curleadinfo.append([curlead, t_n])  # initialization
        for j in range(len(leadlist)):
            if leadlist[j] != curlead:  # if there is a new leader
                curlead = leadlist[j]  # update the current leader
                curleadinfo[-1].append(t_n + j - 1)  # last time (in frameID) the old leader is observed
                curleadinfo.append([curlead, t_n + j])  # new leader and the first time (in frameID) it is observed.
        curleadinfo[-1].append(t_n + len(leadlist) - 1)  # termination

        leadinfo.append(curleadinfo)

    return leadinfo


def makefolinfo(platoon, platooninfo, sim, *args, allfollowers=True, endtime='Tn'):
    # same as leadinfo but it gives followers instead of leaders.
    # followers computed ONLY OVER SIMULATED TIMES + BOUNDARY CONDITION TIMES (t_n - T_nm1 + T_nm1 - T_n)
    # allfollowers = True -> gives all followers, even if they aren't in platoon
    # allfollowers = False -> only gives followers in the platoon (needed for adjoint calculation, adjoint variables depend on followers, not leaders.)
    # endtime = 'Tn' calculates followers between t_n, T_n, otherwise calculated between t_n, T_nm1,
    # so give endtime = 'Tnm1' and it will not compute followers over boundary condition times

    # EXAMPLE
    ##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
    ##    allfollowers = False-
    ##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers IN THE PLATOON
    # all followers = True -
    #    [[[7,11,20]], [[8, 11, 15],[9,16,20]]] #vehicle 7 has 8 and 9 as followers
    folinfo = []
    for i in platoon:
        curfolinfo = []
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        if endtime == 'Tn':
            follist = sim[i][t_n - t_nstar:T_n - t_nstar + 1, 5]  # list of followers
        else:
            follist = sim[i][t_n - t_nstar:T_nm1 - t_nstar + 1, 5]
        curfol = follist[0]
        if curfol != 0:  # you need to get the time of the follower to make sure it is actually being simulated in that time
            # (this is for the case when allfollower = False)
            foltn = platooninfo[curfol][1]
        else:
            foltn = math.inf
        unfinished = False

        if allfollowers and curfol != 0:
            curfolinfo.append([curfol, t_n])
            unfinished = True
        else:
            if curfol in platoon and t_n >= foltn:  # if the current follower is in platoons we initialize
                curfolinfo.append([curfol, t_n])
                unfinished = True

        for j in range(len(follist)):  # check what we just made to see if we need to put stuff in folinfo
            if follist[j] != curfol:  # if there is a new follower
                curfol = follist[j]
                if curfol != 0:
                    foltn = platooninfo[curfol][1]
                else:
                    foltn = math.inf
                if unfinished:  # if currrent follower entry is not finished
                    curfolinfo[-1].append(t_n + j - 1)  # we finish the interval
                    unfinished = False
                # check if we need to start a new fol entry
                if allfollowers and curfol != 0:
                    curfolinfo.append([curfol, t_n + j])
                    unfinished = True
                else:
                    if curfol in platoon and t_n + j >= foltn:  # if new follower is in platoons
                        curfolinfo.append([curfol, t_n + j])  # start the next interval
                        unfinished = True
        if unfinished:  # if currrent follower entry is not finished
            curfolinfo[-1].append(t_n + len(follist) - 1)  # finish it

        folinfo.append(curfolinfo)

    return folinfo


def makeleadfolinfo(platoons, platooninfo, sim, *args, relaxtype='both', mergertype='avg', merge_from_lane=7,
                    merge_lane=6, make_folinfo=False):
    # new makeleadfolinfo function which integrates the previous versions
    # inputs -
    # platoons : platoon you want to calibrate
    # platooninfo - output from makeplatooninfo
    # meas - measurements in usual format
    # relaxtype = 'pos', 'neg', 'both', 'none'  - choose between positive, negative, and pos/negative relaxation amounts added. 'none' is no relax.
    # mergertype = 'avg', 'none', 'remove'- 'avg' calculates the relaxation amount using average headway
    # if 'none' will not get merger relaxation amounts, but NOTE that some mergers are actually treated as lane changes and these are still kept.
    # if 'remove' will actually go through and remove those mergers treated as lane changes (this happens when you had a leader in the on-ramp, and then merged before your leader)
    # merge_from_lane = 7 - if using merger anything other than 'none', you need to specify the laneID corresponding to the on-ramp
    # merge_lane = 6 - if using merger anything other than 'none' you need to specify the laneID you are merging into

    # outputs -
    # leadinfo - list of lists with the relevant lead info (lists of triples leadID, starttime, endtime )
    # leadinfo lets you get the lead vehicle trajectory of a leader in a vectorized way.
    # folinfo - same as leadinfo, but for followers instead of leaders.
    # rinfo - gets times and relaxation amounts. Used for implementing relaxation phenomenon, which prevents
    # unrealistic behaviors due to lane changing, and improves lane changing dynamics

    ##EXAMPLE:
    ##    platoons = [[],5,7] means we want to calibrate vehicles 5 and 7 in a platoon
    ##
    ##    leadinfo = [[[1,1,10],[2,11,20]],[[5,10,500]]] Means that vehicle 5 has vehicle 1 as a leader from 1 to 10, 2 as a leader from 11 to 20.
    ##    vehicle 7 has 3 as a leader from 10 to 500 (leadinfo[i] is the leader info for platoons[i]. leadinfo[i] is a list of lists, so leadinfo is a list of lists of lists.)
    ##
    ##    folinfo = [[[7,11,20]], [[]]] Means that vehicle 5 has vehicle 7 as a follower in the platoon from 11 to 20, and vehicle 7 has no followers in the platoon

    # legacy info-
    # makeleadfolinfo_r - 'pos' 'none'
    # makeleadfolinfo_r2 - 'neg', 'none'
    # makeleadfolinfo_r3 - 'both', 'none'
    # makeleadfolinfo_r4 - 'both', 'avg'
    # makeleadfolinfo_r5 - 'both', 'last'
    # makeleadfolinfo_r6 - 'both', 'remove'

    # in the original implementation everything was done at once which I guess saves some work but makes it harder to change/develop.
    # in this refactored version everything is modularized which is a lot nicer but slower. These functions are for calibration, and for calibration
    # all the time (>99.99%) is spent simulating, the time you spend doing makeleadfolinfo is neglible. Hence this design makes sense.

    leadinfo = makeleadinfo(platoons, platooninfo, sim)
    rinfo = makerinfo(platoons, platooninfo, sim, leadinfo, relaxtype=relaxtype, mergertype=mergertype,
                      merge_from_lane=merge_from_lane, merge_lane=merge_lane)

    if make_folinfo:
        folinfo = makefolinfo(platoons, platooninfo, sim, allfollowers=False)
        return leadinfo, folinfo, rinfo
    else:
        return leadinfo, rinfo


def makerinfo(platoons, platooninfo, sim, leadinfo, relaxtype='both', mergertype='avg', merge_from_lane=7,
              merge_lane=6):
    """Rule for merger is not consistent with newer relaxation."""
    if relaxtype == 'none':
        return [[] for i in platoons]

    rinfo = []
    for i in platoons:
        currinfo = []
        t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]
        leadlist = sim[i][t_n - t_nstar:T_nm1 - t_nstar + 1,
                   4]  # this gets the leaders for each timestep of the current vehicle\
        curlead = leadlist[0]

        for j in range(len(leadlist)):
            if leadlist[j] != curlead:
                newlead = leadlist[j]
                oldlead = curlead

                #####relax constant calculation
                newt_nstar = platooninfo[newlead][0]
                oldt_nstar = platooninfo[oldlead][0]
                try:
                    olds = sim[oldlead][t_n + j - oldt_nstar, 2] - sim[oldlead][0, 6] - sim[i][
                        t_n + j - t_nstar, 2]  # the time is t_n+j-1; this is the headway
                except:
                    olds = sim[oldlead][t_n + j - 1 - oldt_nstar, 2] - sim[oldlead][0, 6] - sim[i][
                        t_n + j - t_nstar, 2] + .1 * sim[oldlead][t_n + j - 1 - oldt_nstar, 3]

                news = sim[newlead][t_n + j - newt_nstar, 2] - sim[newlead][0, 6] - sim[i][
                    t_n + j - t_nstar, 2]  # the time is t_n+j
                ########

                # pos/neg relax amounts
                gam = olds - news
                if relaxtype == 'both':
                    currinfo.append([t_n + j, gam])

                    if mergertype == 'remove':
                        if sim[i][t_n + j - t_nstar, 7] == merge_lane and sim[i][
                            t_n + j - 1 - t_nstar, 7] == merge_from_lane:
                            currinfo.pop(-1)
                elif relaxtype == 'pos':
                    if gam > 0:
                        currinfo.append([t_n + j, gam])

                        if mergertype == 'remove':
                            if sim[i][t_n + j - t_nstar, 7] == merge_lane and sim[i][
                                t_n + j - 1 - t_nstar, 7] == merge_from_lane:
                                currinfo.pop(-1)
                elif relaxtype == 'neg':
                    if gam < 0:
                        currinfo.append([t_n + j, gam])

                        if mergertype == 'remove':
                            if sim[i][t_n + j - t_nstar, 7] == merge_lane and sim[i][
                                t_n + j - 1 - t_nstar, 7] == merge_from_lane:
                                currinfo.pop(-1)

                curlead = newlead
        rinfo.append(currinfo)
    # merger cases
    if mergertype == 'avg':
        rinfo = merge_rconstant(platoons, platooninfo, sim, leadinfo, rinfo, 200, merge_from_lane, merge_lane)

    return rinfo


def merge_rconstant(platoons, platooninfo, sim, leadinfo, rinfo, relax_constant=100, merge_from_lane=7, merge_lane=6,
                    datalen=9, h=.1):
    for i in range(len(platoons)):
        curveh = platoons[i]
        t_nstar, t_n, T_nm1, T_n = platooninfo[curveh][0:4]
        lanelist = np.unique(sim[curveh][:t_n - t_nstar, 7])

        if merge_from_lane in lanelist and merge_lane not in lanelist and sim[curveh][
            t_n - t_nstar, 7] == merge_lane:  # get a merge constant when a vehicle's simulation starts when they enter the highway #
            lead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
            for j in leadinfo[i]:
                curleadid = j[0]  # current leader ID
                leadt_nstar = int(sim[curleadid][0, 1])  # t_nstar for the current lead, put into int
                lead[j[1] - t_n:j[2] + 1 - t_n, :] = sim[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,
                                                     :]  # get the lead trajectory from simulation
            headway = lead[:, 2] - sim[curveh][t_n - t_nstar:, 2] - lead[:, 6]
            headway = headway[:T_nm1 + 1 - t_n]
            # calculate the average headway when not close to lane changing events
            headwaybool = np.ones(len(headway), dtype=bool)
            for j in rinfo[i]:
                headwaybool[j[0] - t_n:j[0] - t_n + relax_constant] = 0

            headway = headway[headwaybool]
            if len(headway) > 0:  # if there are any suitable headways we can use then do it
                preheadway = np.mean(headway)

                postlead = sim[curveh][t_n - t_nstar, 4]
                postleadt_nstar = platooninfo[postlead][0]

                posthd = sim[postlead][t_n - postleadt_nstar, 2] - sim[postlead][t_n - postleadt_nstar, 6] - \
                         sim[curveh][t_n - t_nstar, 2]

                curr = preheadway - posthd
                rinfo[i].insert(0, [t_n, curr])
            # another strategy to get the headway in the case that there aren't any places we can estimate it from

            else:
                # it never reaches this point unless in a special case
                leadlist = np.unique(sim[curveh][:t_n - t_nstar, 4])
                if len(leadlist) > 1 and 0 in leadlist:  # otherwise if we are able to use the other strategy then use that

                    cursim = sim[curveh][:t_n - t_nstar, :].copy()
                    cursim = cursim[cursim[:, 7] == merge_from_lane]
                    cursim = cursim[cursim[:, 4] != 0]

                    curt = cursim[-1, 1]
                    curlead = cursim[-1, 4]
                    leadt_nstar = platooninfo[curlead][0]

                    prehd = sim[curlead][curt - leadt_nstar, 2] - sim[curlead][curt - leadt_nstar, 6] - cursim[
                        -1, 2]  # headway before

                    postlead = sim[curveh][t_n - t_nstar, 4]
                    postleadt_nstar = platooninfo[postlead][0]

                    posthd = sim[postlead][t_n - postleadt_nstar, 2] - sim[postlead][t_n - postleadt_nstar, 6] - \
                             sim[curveh][t_n - t_nstar, 2]
                    curr = prehd - posthd

                    rinfo[i].insert(0, [t_n, curr])

                else:  # if neither strategy can be used then we can't get a merger r constant for the current vehicle.
                    continue

    return rinfo


def arraytraj(meas, followerchain, mytime=None, timesteps=8):
    # puts output from makefollerchain/makeplatooninfo into a dict where the key is frame ID, value is array of vehicle and their position, speed
    # we can include the presimulation (t_nstar to t_n) as well as postsimulation (T_nm1 to T_n) but including those are optional
    # this will put in the whole trajectory based off of the times in followerchain
    if mytime is None:
        t_list = []  # t_n list
        T_list = []  # T_n list
        for i in followerchain.values():
            t_list.append(i[0])
            T_list.append(i[3])
        T_n = int(max(T_list))  # T_{n} #get maximum T
        t_1 = int(min(t_list))  # t_1 #get minimum T
        mytime = list(range(t_1, T_n + 1))  # time range of data

    platoontraj = {k: [] for k in mytime}
    for i in followerchain.keys():
        curmeas = meas[i]
        t_n, T_n = followerchain[i][0], followerchain[i][3]
        curtime = range(max(t_n, mytime[0]), min(T_n, mytime[-1])+1)
        curmeas = add_lane_interp_to_data(curmeas, curtime, timesteps=timesteps)
        for t in curtime:
            platoontraj[t].append(curmeas[t-t_n, [2, 7, 3, 0, 6]])
    for t in mytime:
        cur = platoontraj[t]
        if len(cur) > 0:
            platoontraj[t] = np.stack(cur, axis=0)
        else:
            platoontraj[t] = np.empty((0, 5))
    return platoontraj, mytime


def add_lane_interp_to_data(meas, curtime, timesteps=8):
    # changes discrete lane positions to be continuous approximation, where lane change occurs over timesteps timesteps
    if len(curtime) == 0:
        return meas
    start, end = curtime.start, curtime.stop
    t_n = int(meas[0, 1])

    new_meas = meas.copy()
    lanedata = new_meas[start-t_n:end-t_n, 7]
    lanejumps = lanedata[1:] - lanedata[:-1]
    new_lane_inds = np.nonzero(lanejumps)[0]
    for i in new_lane_inds:
        init_lane = lanedata[i]
        delta = lanejumps[i]
        delta_pos = delta/timesteps
        new_lanes = [init_lane + delta_pos*j for j in range(1, timesteps)]
        if i + timesteps - 1 > len(lanedata)-1:
            new_lanes = new_lanes[:len(lanedata)-i-1]
            lanedata[i+1:] = np.array(new_lanes)
        else:
            lanedata[i+1:i+timesteps] = np.array(new_lanes)
    return new_meas


def platoononly(platooninfo, platoon):
    # makes it so platooninfo only contains entries for platoon
    # platoon can be a platoon in form [ 1, 2, 3, etc.] or a list of those.
    ans = {}

    if type(platoon[0]) == list:  # list of platoons
        useplatoon = []
        for i in platoon:
            useplatoon.extend(i[:])
    else:
        useplatoon = platoon

    for i in useplatoon[:]:
        ans[i] = platooninfo[i]
    return ans


def calculateflows(meas, spacea, timea, agg, lane=None, method='area', h=.1,
                   time_units=3600, space_units=1000):
    # meas = measurements, in usual format (dictionary where keys are vehicle IDs, values are numpy arrays
    # spacea - reads as ``space A'' (where A is the region where the macroscopic quantities are being calculated).
    # list of lists, each nested list is a length 2 list which ... represents the starting and ending location on road.
    # So if len(spacea) >1 there will be multiple regions on the road which we are tracking e.g. spacea = [[200,400],[800,1000]],
    # calculate the flows in regions 200 to 400 and 800 to 1000 in meas.
    # timea - reads as ``time A'', should be a list of the times (in the local time of thedata).
    # E.g. timea = [1000,3000] calculate times between 1000 and 3000.
    # agg - aggregation length, float number which is the length of each aggregation interval.
    # E.g. agg = 300 each measurement of the macroscopic quantities is over 300 time units in the data,
    # so in NGSim where each time is a frameID with length .1s, we are aggregating every 30 seconds.
    # h specifies unit conversion - i.e. if 1 index in data = .1 of units you want, h = .1
    # e.g. ngsim has .1 seconds between measurements, so h = .1 yields units of seconds for time. no conversion for space units
    # area method (from laval paper), or flow method (count flow into space region, calculate space mean speed, get density from flow/speed)
    # area method is better

    # for each space region, value is a list of floats of the value at the correpsonding time interval
    q = [[] for i in spacea]
    k = [[] for i in spacea]

    # starttime = [i[0,1] for i in meas.values()]
    # starttime = int(min(starttime)) #first time index in data

    spacealist = []
    for i in spacea:
        spacealist.extend(i)
    # spacemin = min(spacealist)
    # spacemax = max(spacealist)
    # timemin = min(timea)
    # timemax = max(timea)

    intervals = []  # tuples of time intervals
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
    # regions are indexed by space, then time. values are list of (position traveled, time elapsed) (list of float, list of float)

    flows = [[0 for j in intervals] for i in
             spacea]  # used if method = 'flow', indexed by space, then time, int of how many vehicles enter region
    for vehid in meas:
        alldata = meas[vehid]

        # if lane is given we need to find the segments of data inside the lane
        if lane is not None:
            # e.g. if data is in lane 2 for times (0-10), lane 1 for times (11-20), and we want lane 1 data,
            # we want to look at the times (10-20), so we don't lose the interval (10-11)
            temp = alldata[alldata[:, 7] == lane]  # boolean mask selects data inside lane
            if len(temp) == 0:
                continue
            inds = sequential(temp)  # returns indexes where there are jumps
            indlist2 = []
            for i in range(len(inds) - 1):
                indlist2.append([inds[i], inds[i + 1]])
            # new code to get the correct intervals
            mint = alldata[0, 1]
            indlist = []
            for i in indlist2:
                temp_start, temp_end = temp[i[0], 1], temp[i[1] - 1, 1]
                temp_start = max(temp_start - 1, mint)
                indlist.append([int(temp_start - mint), int(temp_end - mint + 1)])

        else:  # otherwise can just use everything
            indlist = [[0, len(alldata)]]

        for i in indlist:
            data = alldata[i[0]:i[
                1]]  # select only current region of data - #sequential data for a single vehicle in correct lane if applicable
            if len(data) == 0:
                continue
            #            region_contained = []
            #            region_data = {}  # key: tid, sid

            for i in range(len(intervals)):
                # start =  int(max(0, intervals[i][0] + starttime - data[0,1])) #indices for slicing data
                # end = int(max(0, intervals[i][1] + starttime - data[0,1])) #its ok if end goes over for slicing - if both zero means no data in current interval
                start = int(max(0, intervals[i][0] - data[0, 1]))  # indices for slicing data
                end = int(max(0, intervals[i][1] + 1 - data[
                    0, 1]))  # its ok if end goes over for slicing - if both zero means no data in current interval

                curdata = data[start:end]
                if len(curdata) == 0:
                    continue

                for j in range(len(spacea)):
                    minspace, maxspace = spacea[j][0], spacea[j][1]
                    # curspacedata = curdata[np.all([curdata[:,2] > minspace, curdata[:,2] < maxspace], axis = 0)]
                    # if len(curspacedata) == 0:
                    #     continue
                    # new code to interpolate onto regions
                    minind = interval_binary_search(curdata[:, 2], minspace)
                    maxind = interval_binary_search(curdata[:, 2], maxspace)
                    maxind = min(len(curdata) - 1, maxind + 1)
                    if minind == maxind:
                        continue
                    curspacedata = curdata[minind:maxind + 1, [1, 2]]  # array of times, positions
                    if curdata[maxind, 2] < minspace or curdata[minind, 2] > maxspace:
                        continue
                    if curdata[minind, 2] < minspace:
                        left = curdata[minind, 2]
                        right = curdata[minind + 1, 2]
                        interp = (minspace - left) / (right - left)
                        curspacedata[0, 0] += interp
                        curspacedata[0, 1] = minspace
                    if curdata[maxind, 2] > maxspace:
                        left = curdata[maxind - 1, 2]
                        right = curdata[maxind, 2]
                        interp = (maxspace - left) / (right - left)
                        curspacedata[-1, 0] = curspacedata[-2, 0] + interp
                        curspacedata[-1, 1] = maxspace

                    regions[j][i][0].append(curspacedata[-1, 1] - curspacedata[0, 1])
                    regions[j][i][1].append((curspacedata[-1, 0] - curspacedata[0, 0]))
                    if method == 'flow':
                        firstpos, lastpos = curspacedata[0, 1], curspacedata[-1, 1]
                        if firstpos <= spacea[j][0] and lastpos > spacea[j][0]:
                            flows[j][i] += 1

    if method == 'area':
        for i in range(len(spacea)):
            for j in range(len(intervals)):
                area = (spacea[i][1] - spacea[i][0]) * (intervals[j][1] - intervals[j][0])
                q[i].append(sum(regions[i][j][0]) / area / h * time_units)
                k[i].append(sum(regions[i][j][1]) / area * space_units)
    elif method == 'flow':
        for i in range(len(spacea)):
            for j in range(len(intervals)):
                q[i].append(flows[i][j] / (intervals[j][1] - intervals[j][0]))
                try:
                    k[i].append(sum(regions[i][j][0]) / sum(regions[i][j][1]))
                except:
                    k[i].append(0)  # division by zero when region is empty

    return q, k


def r_constant(currinfo, frames, T_n, rp, adj=True, h=.1):
    # currinfo - output from makeleadfolinfo_r*
    # frames - [t_n, T_nm1], a list where the first entry is the first simulated time and the second entry is the last simulated time
    # T_n - last time vehicle is observed
    # rp - value for the relaxation, measured in real time (as opposed to discrete time)
    # adj = True - can output needed values to compute adjoint system
    # h = .1 - time discretization

    # given a list of times and gamma constants (rinfo for a specific vehicle = currinfo) as well as frames (t_n and T_nm1 for that specific vehicle) and the relaxation constant (rp). h is the timestep (.1 for NGSim)
    # we will make the relaxation amounts for the vehicle over the length of its trajectory
    # rinfo is precomputed in makeleadfolinfo_r. then during the objective evaluation/simulation, we compute these times.
    # note that we may need to alter the pre computed gammas inside of rinfo; that is because if you switch mutliple lanes in a short time, you may move to what looks like only a marginally shorter headway,
    # but really you are still experiencing the relaxation from the lane change you just took
    if len(currinfo) == 0:
        relax = np.zeros(T_n - frames[0] + 1)
        return relax, relax  # if currinfo is empty we don't have to do anything

    out = np.zeros((T_n - frames[0] + 1, 1))  # initialize relaxation amount for the time between t_n and T_n
    out2 = np.zeros((T_n - frames[0] + 1, 1))
    outlen = 1

    maxind = frames[1] - frames[
        0] + 1  # this is the maximum index we are supposed to put values into because the time between T_nm1 and T_n is not simulated. Plus 1 because of the way slices work.
    if rp < h:  # if relaxation is too small for some reason
        rp = h  # this is the smallest rp can be
    #    if rp<h: #if relaxation is smaller than the smallest it can be #deprecated
    #        return out, out2 #there will be no relaxation

    mylen = math.ceil(
        rp / h) - 1  # this is how many nonzero entries will be in r each time we have the relaxation constant
    r = np.linspace(1 - h / rp, 1 - h / rp * (mylen),
                    mylen)  # here are the relaxation constants. these are determined only by the relaxation constant. this gets multipled by the 'gamma' which is the change in headway immediately after the LC

    for i in range(
            len(currinfo)):  # frames[1]-frames[0]+1 is the length of the simulation; this makes it so it will be all zeros between T_nm1 and T_n
        entry = currinfo[i]  # the current entry for the relaxation phenomenon
        curind = entry[0] - frames[0]  # current time is entry[0]; we start at frames[0] so this is the current index
        for j in range(outlen):
            if out2[curind, j] == 0:
                if curind + mylen > maxind:  # in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)
                    out[curind:maxind, j] = r[0:maxind - curind]
                    out2[curind:maxind, j] = currinfo[i][1]
                else:  # this is the normal case
                    out[curind:curind + mylen, j] = r
                    out2[curind:curind + mylen, j] = currinfo[i][1]
                break

        else:
            newout = np.zeros((T_n - frames[0] + 1, 1))
            newout2 = np.zeros((T_n - frames[0] + 1, 1))

            if curind + mylen > maxind:  # in this case we can't put in the entire r because we will get into the shifted end part (and also possibly get an index out of bounds error)
                newout[curind:maxind, 0] = r[0:maxind - curind]
                newout2[curind:maxind, 0] = currinfo[i][1]
            else:  # this is the normal case
                newout[curind:curind + mylen, 0] = r
                newout2[curind:curind + mylen, 0] = currinfo[i][1]

            out = np.append(out, newout, axis=1)
            out2 = np.append(out2, newout2, axis=1)
            outlen += 1

    #######calculate relaxation amounts and the part we need for the adjoint calculation #different from the old way
    relax = np.multiply(out, out2)
    relax = np.sum(relax, 1)

    if adj:
        outd = -(1 / rp) * (
                    out - 1)  # derivative of out (note that this is technically not the derivative because of the piecewise nature of out/r)
        relaxadj = np.multiply(outd,
                               out2)  # once multiplied with out2 (called gamma in paper) it will be the derivative though.
        relaxadj = np.sum(relaxadj, 1)
    else:
        relaxadj = relax

    return relax, relaxadj


def crash_confidence(crashes, n_sims, vmt_sim, z=1.96, inverse=True):
    """Calculates confidence interval of a crash rate. Assumes number of crashes per simulation is poisson distributed.

    Args:
        crashes: total number of crashes
        n_sims: total number of (identically distributed) simulations
        vmt_sim: average number of miles driven per simulation
        z: z-score corresponding to (1 - \alpha/2) percentile, where \alpha is the confidence interval.
        inverse: if True, return inverse crash rate (miles/event). Otherwise, return crash rate (event/miles)
    Returns:
        mean: average crash rate (events/miles)
        low: lower confidence interval of crash rate (events/miles)
        high: upper confidence interval of crash rate (events/miles)
    """
    crashes = crashes if crashes > 0 else 0.69
    mean = crashes/n_sims
    if crashes < 20:
        temp = crashes/n_sims + z**2/(2*n_sims)
        temp2 = z/2/n_sims*(4*crashes + z**2)**.5
        if inverse:
            return vmt_sim/mean, vmt_sim/(temp+temp2), vmt_sim/(temp-temp2)
        else:
            return mean/vmt_sim, (temp-temp2)/vmt_sim, (temp+temp2)/vmt_sim
    else:
        temp = crashes**.5*z/n_sims
        if inverse:
            return vmt_sim/mean, vmt_sim/(mean + temp), vmt_sim/(mean - temp)
        else:
            return mean/vmt_sim, (mean - temp)/vmt_sim, (mean + temp)/vmt_sim


def count_leadmem(veh, timeind):
    if timeind < veh.start:
        return 0
    for count, leadmem in enumerate(veh.leadmem[:-1]):
        if leadmem[1] <= timeind < veh.leadmem[count + 1][1]:
            break
    else:
        count = len(veh.leadmem) - 1
    return count


def add_leaders(veh_list, start, end):
    """For a list of vehicles, add all leaders and leaders of leaders within times [start, end].

    Args:
        veh_list: list of Vehicles
        start: int time index
        end: int time index
    Returns:
        platoon: list containing all Vehicles in veh_list, and all leaders in times [start, end].
    """
    platoon = veh_list.copy()
    for veh in veh_list:
        for mem in veh.leadmem[count_leadmem(veh, start):count_leadmem(veh, end)+1]:
            if hasattr(mem[0], 'vehid'):
                platoon.append(mem[0])
                for mem2 in mem[0].leadmem[count_leadmem(mem[0], start):count_leadmem(mem[0], end)+1]:
                    if hasattr(mem2[0], 'vehid'):
                        platoon.append(mem2[0])
    return list(set(platoon))


def indtodata(indjumps, data, dataind=2):
    out = []
    for i in range(len(indjumps)-1):
        startpos  = data[indjumps[i],dataind]
        endpos = data[indjumps[i+1]-1,dataind]
        temp = (startpos, endpos)
        out.append(temp)
    return out
