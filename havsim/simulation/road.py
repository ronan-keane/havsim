from havsim.simulation.road_networks import Lane, downstream_wrapper, get_inflow_wrapper, increment_inflow_wrapper

def compute_route(start_road, start_pos, exit):
    """
    start_road: object for the start road
    start_pos: position on the start road
    exit: name for the exit road (string)
    """
    # The exit must be of str type
    assert isinstance(exit, str)
    # A map from (in_road, out_road) to (shortest_dist, pos, last_in_road_out_road_pair_in_the_shortest_path)
    # (in_road, out_road) uniquely identify a connection point between two roads
    dist = dict()
    dist[(None, start_road)] = (0, start_pos, None)

    # Visited (in_road, out_road) pair
    visited = set()

    def get_road_name(rd):
        if isinstance(rd, str):
            return rd
        return rd.name

    def get_min_distance_road_info():
        ret = None
        min_dist = float('inf')
        for k, v in dist.items():
            if k not in visited and v[0] < min_dist:
                min_dist = v[0]
                ret = k, v
        return ret

    while True:
        cur = get_min_distance_road_info()
        if cur is None:
            break
        (in_road, out_road), (shortest_dist, pos, last_road_info) = cur

        # If the road is an exit
        if isinstance(out_road, str):
            if out_road == exit:
                res = []
                # Construct the shortest path backwards
                while last_road_info is not None:
                    if not res or res[-1] != get_road_name(out_road):
                        res.append(get_road_name(out_road))
                    in_road, out_road = last_road_info
                    last_road_info = dist[last_road_info][2]
                res.reverse()
                # Ignore the start road on the path
                return res
        else:
            for neighbor_name, neighbor_info in out_road.connect_to.items():
                # We only deal with unvisited neighbors
                if neighbor_name not in visited:
                    if neighbor_info[1] == 'continue':
                        connect_pos_neighbor = neighbor_info[0]
                    else:
                        connect_pos_neighbor = neighbor_info[0][0]
                    if connect_pos_neighbor >= pos:
                        new_dist_neighbor = shortest_dist + connect_pos_neighbor - pos
                        neighbor_entry = out_road, (neighbor_name if neighbor_info[-1] is None else neighbor_info[-1])
                        if neighbor_entry not in dist or dist[neighbor_entry][0] > new_dist_neighbor:
                            if neighbor_info[-1] is None:
                                pos_neighbor = 0
                            else:
                                pos_neighbor = connect_pos_neighbor + out_road[0].roadlen[neighbor_name]
                            dist[neighbor_entry] = (new_dist_neighbor, pos_neighbor, (in_road, out_road))
        visited.add((in_road, out_road))
    # If we reach here, it means the exit is not reachable, reach empty route
    return []


def add_or_get_merge_anchor_index(lane, pos):
    """
    Add a merge anchor at pos if needed, return the index of the merge anchor
    """
    if not hasattr(lane, "merge_anchors"):
        lane.merge_anchors = []
    for ind, e in enumerate(lane.merge_anchors):
        if (e[1] is None and pos == lane.start) or e[1] == pos:
            return ind
    lane.merge_anchors.append([lane.anchor, None if pos == lane.start else pos])
    return len(lane.merge_anchors) - 1


def connect_lane_left_right(left_lane, right_lane, left_connection, right_connection):
    if left_lane is None or right_lane is None:
        return

    if left_connection[0] == left_lane.start:
        left_lane.connect_right[0] = (left_lane.start, right_lane)
    else:
        assert left_connection[0] > left_lane.start
        left_lane.connect_right.append((left_connection[0], right_lane))
        merge_anchor_ind = add_or_get_merge_anchor_index(right_lane, right_connection[0])
        left_lane.events.append(
            {'event': 'update lr', 'right': 'add', 'left': None, 'right anchor': merge_anchor_ind,
             'pos': left_connection[0]})

    if left_connection[1] < left_lane.end:
        left_lane.connect_right.append((left_connection[1], None))
        left_lane.events.append(
            {'event': 'update lr', 'right': 'remove', 'left': None,
             'pos': left_connection[1]})

    if right_connection[0] == right_lane.start:
        right_lane.connect_left[0] = (right_lane.start, left_lane)
    else:
        assert right_connection[0] > right_lane.start
        right_lane.connect_left.append((right_connection[0], left_lane))
        merge_anchor_ind = add_or_get_merge_anchor_index(left_lane, left_connection[0])
        right_lane.events.append(
            {'event': 'update lr', 'left': 'add', 'right': None, 'left anchor': merge_anchor_ind,
             'pos': right_connection[0]})

    if right_connection[1] < right_lane.end:
        right_lane.connect_left.append((right_connection[1], None))
        right_lane.events.append(
            {'event': 'update lr', 'left': 'remove', 'right': None,
             'pos': right_connection[1]})

    # Sort lane events by position
    left_lane.connect_right.sort(key=lambda d: d[0])
    right_lane.connect_left.sort(key=lambda d: d[0])
    left_lane.events.sort(key=lambda d: d['pos'])
    right_lane.events.sort(key=lambda d: d['pos'])


class Road:
    def __init__(self, num_lanes, length, name, connections=None):
        """
        num_lanes: number of lanes (int)
        length: if float/int, the length of all lanes. If a list, each entry gives the (start, end) tuple
          for the corresponding lane
        name: name of the road
        connections: If None, we assume the lanes are connected where possible. Otherwise, It is a list where
          each entry is a tuple of ((left start, left end), (right start, right end)) giving the connections
        """
        self.num_lanes = num_lanes
        self.name = name

        # Validate connections arg
        if connections is not None:
            assert len(connections) == num_lanes

        # Validate and canonicalize length arg
        if isinstance(length, int) or isinstance(length, float):
            assert length > 0
            length = [(0, length) for _ in range(num_lanes)]
        else:
            assert isinstance(length, list) and len(length) == num_lanes and isinstance(length[0], tuple)

        # Construct lane objects for the road
        self.lanes = []
        # All lanes will share the same roadlen dictionary,
        # this is convenient for updating roadlen
        roadlen = {name: 0}
        for i in range(num_lanes):
            lane = Lane(start=length[i][0], end=length[i][1], road=self, laneind=i)
            # Overwrite the default roadlen to be the shared one
            lane.roadlen = roadlen
            self.lanes.append(lane)

        # Connect adjacent lanes
        for i in range(num_lanes - 1):
            left_lane = self.lanes[i]
            right_lane = self.lanes[i + 1]
            if connections is not None:
                left_connection = connections[i][1]
                right_connection = connections[i + 1][0]
            else:
                left_connection = right_connection = (
                    max(left_lane.start, right_lane.start), min(left_lane.end, right_lane.end))
            connect_lane_left_right(left_lane, right_lane, left_connection, right_connection)

    def connect(self, new_road, self_indices=None, new_road_indices=None, is_exit=False):
        """
        new_road: road object to make the connection to. If passing in a string, it's assumed
            to be the name of the exit road (need to mark is_exit as True)
        self_indices: a list of indices of the current road for making the connection
        new_road_indices: a list of indices of the new road to connect to
        is_exit: whether the new road is an exit. An exit road won't have a road object.
        """

        if self_indices is None:
            self_indices = list(range(self.num_lanes))
        self_indices.sort()

        # We should check that self_indices is a continuous list of numbers, because the laneind in lane's
        # connections attribute makes this assumption (if 'continue', it's a tuple of 2 ints, giving the
        # leftmost and rightmost lanes which will continue to the desired lane. if 'merge', it's the laneind
        # of the lane we need to be on to merge.)
        def is_continuously_increasing(nums):
            for i in range(len(nums) - 1):
                if nums[i] + 1 != nums[i+1]:
                    return False
            return True
        assert is_continuously_increasing(self_indices)

        # If passing in a string, new_road is assumed to be the name of the exit
        if isinstance(new_road, str):
            assert is_exit
            all_lanes_end = tuple([self.lanes[i].end for i in self_indices])
            # It's assumed that all exits have the same end position
            assert all_lanes_end and min(all_lanes_end) == max(all_lanes_end)
            # We don't need to update roadlen for exit type roads
            for i in self_indices:
                self.lanes[i].events.append({'event': 'exit', 'pos': self.lanes[i].end})
            for i in range(self.num_lanes):
                cur_lane = self.lanes[i]
                cur_lane.connections[new_road] = (all_lanes_end[0], 'continue', (self_indices[0],self_indices[-1]), None, None)
        else:
            if new_road_indices is None:
                new_road_indices = list(range(new_road.num_lanes))

            new_road_indices.sort()
            # It is assumed that self_indices and new_road_indices have the same length
            assert len(self_indices) == len(new_road_indices)

            all_self_lanes_end = tuple([self.lanes[i].end for i in self_indices])
            all_new_lanes_start = tuple([new_road.lanes[i].start for i in new_road_indices])
            # It's assumed that all self lanes have the same end position and all new lanes
            # have the same start position
            assert all_self_lanes_end and min(all_self_lanes_end) == max(all_self_lanes_end)
            assert all_new_lanes_start and min(all_new_lanes_start) == max(all_new_lanes_start)
            # Since roadlen dict is shared across all lanes, we only need to update it via one of
            # the lanes
            self.lanes[0].roadlen[new_road.name] = all_self_lanes_end[0] - all_new_lanes_start[0]
            new_road.lanes[0].roadlen[self.name] = all_new_lanes_start[0] - all_self_lanes_end[0]

            # Update connections attribute for all lanes
            new_connection = (all_self_lanes_end[0], 'continue',(min(self_indices), max(self_indices)), None, new_road)
            for i in range(self.num_lanes):
                if new_road.name not in self.lanes[i].connections:
                    self.lanes[i].connections[new_road.name] = new_connection
                else:
                    if self.lanes[i].connections[new_road.name][1] == 'continue':
                        left, right = self.lanes[i].connections[new_road.name][2]
                        old_dist = 0 if i in range(left, right + 1) else min(abs(left - i), abs(right - i))
                    else:
                        left_or_right = self.lanes[i].connections[new_road.name][2]
                        old_dist = abs(i - left_or_right)
                    new_dist = 0 if i in range(self_indices[0], self_indices[-1] + 1) else min(abs(i - self_indices[0]), abs(i - self_indices[-1]))
                    if new_dist < old_dist:
                        self.lanes[i].connections[new_road.name] = new_connection

            # Update lane events, connect_to, merge anchors
            for self_ind, new_road_ind in zip(self_indices, new_road_indices):
                self_lane = self.lanes[self_ind]
                new_lane = new_road[new_road_ind]
                # Update connect_to attribute for the self_lane
                self_lane.connect_to = new_lane

                # Update merge anchors for current track
                new_anchor = self_lane.anchor
                lane_to_update = new_lane
                while lane_to_update is not None:
                    lane_to_update.anchor = new_anchor
                    for merge_anchor in lane_to_update.merge_anchors:
                        merge_anchor[0] = new_anchor
                        if merge_anchor[1] is None:
                            merge_anchor[1] = lane_to_update.start
                    if hasattr(lane_to_update, "connect_to"):
                        lane_to_update = lane_to_update.connect_to
                        # Assert that connect_to won't store an exit lane
                        assert not isinstance(lane_to_update, str)
                    else:
                        lane_to_update = None

                # Create lane events to be added for self_lane
                event_to_add = {'event': 'new lane', 'pos': self_lane.end}

                # Update left side
                self_lane_left = self_lane.get_connect_left(self_lane.end)
                new_lane_left = new_lane.get_connect_left(new_lane.start)
                if self_lane_left == new_lane_left:
                    event_to_add['left'] = None
                elif new_lane_left is None:
                    event_to_add['left'] = 'remove'
                elif self_lane_left is None:
                    event_to_add['left'] = 'add'
                    merge_anchor_pos = (new_lane.start if new_lane.road is new_lane_left.road
                                        else new_lane.start - new_lane.roadlen[new_lane_left.road.name])
                    merge_anchor_ind = add_or_get_merge_anchor_index(new_lane_left, merge_anchor_pos)
                    event_to_add['left anchor'] = merge_anchor_ind
                else:
                    event_to_add['left'] = 'update'

                # Update right side
                self_lane_right = self_lane.get_connect_right(self_lane.end)
                new_lane_right = new_lane.get_connect_right(new_lane.start)
                if self_lane_right == new_lane_right:
                    event_to_add['right'] = None
                elif new_lane_right is None:
                    event_to_add['right'] = 'remove'
                elif self_lane_right is None:
                    event_to_add['right'] = 'add'
                    merge_anchor_pos = (new_lane.start if new_lane.road is new_lane_right.road
                                        else new_lane.start - new_lane.roadlen[new_lane_right.road.name])
                    merge_anchor_ind = add_or_get_merge_anchor_index(new_lane_right, merge_anchor_pos)
                    event_to_add['right anchor'] = merge_anchor_ind
                else:
                    event_to_add['right'] = 'update'

                self_lane.events.append(event_to_add)

    def merge(self, new_road, self_index, new_lane_index, self_pos, new_lane_pos, side=None):
        """
        new_road: new road to be merged into
        self_index: index of the self road's lane
        new_lane_index: index of the new road's lane
        self_pos: a tuple indicating the start/end position of the merge connection for the current lane
        new_lane_pos: a tuple indicating the start/end position of the merge connection for the new lane
        side: 'l' or 'r', if not specified a side, will infer it automatically
        """
        if side == 'l':
            change_side = 'l_lc'
        elif side == 'r':
            change_side = 'r_lc'
        else:
            assert side is None
            change_side = 'l_lc' if self_index == 0 else 'r_lc'
        # Update connections attribute for all lanes
        new_connection = (self_pos, 'merge', self_index, change_side, new_road)
        for i in range(self.num_lanes):
            if new_road.name not in self.lanes[i].connections:
                self.lanes[i].connections[new_road.name] = new_connection
            else:
                if self.lanes[i].connections[new_road.name][1] == 'continue':
                    left, right = self.lanes[i].connections[new_road.name][2]
                    old_dist = 0 if i in range(left, right + 1) else min(abs(left - i), abs(right - i))
                else:
                    left_or_right = self.lanes[i].connections[new_road.name][2]
                    old_dist = abs(i - left_or_right)
                new_dist = abs(i - self_index)
                if new_dist < old_dist:
                    self.lanes[i].connections[new_road.name] = new_connection

        assert isinstance(self_pos, tuple) and isinstance(new_lane_pos, tuple)
        # Update roadlen
        self.lanes[0].roadlen[new_road.name] = self_pos[0] - new_lane_pos[0]
        new_road.lanes[0].roadlen[self.name] = new_lane_pos[0] - self_pos[0]
        # Update lane events and connect_left/connect_right for both lanes
        if change_side == 'l_lc':
            connect_lane_left_right(new_road[new_lane_index], self.lanes[self_index], new_lane_pos, self_pos)
        else:
            connect_lane_left_right(self.lanes[self_index], new_road[new_lane_index], self_pos, new_lane_pos)

    def set_downstream(self, downstream, self_indices=None):
        """
        downstream: dictionary of keyword args which defines call_downstream method
        self_indices: a list of lane indices to set downstream condition to
        """
        if downstream is not None:
            if self_indices is None:
                self_indices = list(range(self.num_lanes))
            else:
                assert isinstance(self_indices, list)
            for ind in self_indices:
                # Setup downstream conditions on all lanes on the same track
                cur_lane = self.lanes[ind].anchor.lane
                while True:
                    cur_lane.call_downstream = downstream_wrapper(**downstream).__get__(cur_lane, Lane)
                    if hasattr(cur_lane, 'connect_to') and isinstance(cur_lane.connect_to, Lane):
                        cur_lane = cur_lane.connect_to
                    else:
                        break

    def set_upstream(self, increment_inflow=None, get_inflow=None, new_vehicle=None, self_indices=None):
        """
        increment_inflow: dictionary of keyword args which defines increment_inflow method, or None
        get_inflow: dictionary of keyword args which defines increment_inflow method, or None
        new_vehicle: new_vehicle method, or None
        self_indices: a list of lane indices to set upstream condition to
        """
        if self_indices is None:
            self_indices = list(range(self.num_lanes))
        else:
            assert isinstance(self_indices, list)
        for ind in self_indices:
            lane = self.lanes[ind]
            if get_inflow is not None:
                lane.get_inflow = get_inflow_wrapper(**get_inflow).__get__(lane, Lane)
            if new_vehicle is not None:
                lane.new_vehicle = new_vehicle.__get__(lane, Lane)
            if increment_inflow is not None:
                lane.inflow_buffer = 0
                lane.newveh = None

                lane.increment_inflow = increment_inflow_wrapper(**increment_inflow).__get__(lane, Lane)

    def diverge(self):
        """
        TODO: implement it
        """
        pass

    def __getitem__(self, index):
        assert isinstance(index, int) and 0 <= index < self.num_lanes
        return self.lanes[index]
