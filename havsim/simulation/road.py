from havsim.simulation.road_networks import Lane


def add_merge_anchor(lane, pos):
    """
    Add a merge anchor at pos if needed, return the index of the merge anchor
    """
    lane.merge_anchors.append([lane.anchor, None if pos == lane.start else pos])
    return len(lane.merge_anchors) - 1


def connect_lane_left_right(left_lane, right_lane, left_connection, right_connection):
    if left_lane is None or right_lane is None:
        return

    if left_connection[0] == left_lane.start:
        if not left_lane.connect_right:
            left_lane.connect_right.append((left_lane.start, right_lane))
        else:
            assert left_lane.connect_right[0] == (left_lane.start, None)
            left_lane.connect_right[0] = (left_lane.start, right_lane)
    else:
        assert left_connection[0] > left_lane.start
        if not left_lane.connect_right:
            left_lane.connect_right.append((left_lane.start, None))
        left_lane.connect_right.append((left_connection[0], right_lane))
        merge_anchor_ind = add_merge_anchor(right_lane, right_connection[0])
        left_lane.events.append(
            {'event': 'update lr', 'right': 'add', 'left': None, 'right anchor': merge_anchor_ind,
             'pos': left_connection[0]})

    if left_connection[1] < left_lane.end:
        left_lane.connect_right.append((left_connection[1], None))
        left_lane.events.append(
            {'event': 'update lr', 'right': 'remove', 'left': None,
             'pos': left_connection[1]})

    if right_connection[0] == right_lane.start:
        if not right_lane.connect_left:
            right_lane.connect_left.append((right_lane.start, left_lane))
        else:
            assert right_lane.connect_left[0] == (right_lane.start, None)
            right_lane.connect_left[0] = (right_lane.start, left_lane)
    else:
        assert right_connection[0] > right_lane.start
        if not right_lane.connect_left:
            right_lane.connect_left.append((right_lane.start, None))
        right_lane.connect_left.append((right_connection[0], left_lane))
        merge_anchor_ind = add_merge_anchor(left_lane, left_connection[0])
        right_lane.events.append(
            {'event': 'update lr', 'left': 'add', 'right': None, 'left anchor': merge_anchor_ind,
             'pos': right_connection[0]})

    if right_connection[1] < right_lane.end:
        right_lane.connect_left.append((right_connection[1], None))
        right_lane.events.append(
            {'event': 'update lr', 'left': 'remove', 'right': None,
             'pos': right_connection[1]})

    # Sort lane events by position
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
        self.connect_to = {}

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

        # If passing in a string, new_road is assumed to be the name of the exit
        if isinstance(new_road, str):
            assert is_exit
            all_lanes_end = tuple([self.lanes[i].end for i in self_indices])
            # It's assumed that all exits have the same end position
            assert all_lanes_end and min(all_lanes_end) == max(all_lanes_end)
            self.connect_to[new_road] = (all_lanes_end[0], 'continue', self_indices, None, None)
            # We don't need to update roadlen for exit type roads
            for i in self_indices:
                self.lanes[i].events.append({'event': 'exit', 'pos': self.lanes[i].end})
        else:
            if new_road_indices is None:
                new_road_indices = list(range(new_road.num_lanes))

            # It is assumed that self_indices and new_road_indices have the same length
            assert len(self_indices) == len(new_road_indices)

            all_self_lanes_end = tuple([self.lanes[i].end for i in self_indices])
            all_new_lanes_start = tuple([new_road.lanes[i].start for i in new_road_indices])
            # It's assumed that all self lanes have the same end position and all new lanes
            # have the same start position
            assert all_self_lanes_end and min(all_self_lanes_end) == max(all_self_lanes_end)
            assert all_new_lanes_start and min(all_new_lanes_start) == max(all_new_lanes_start)
            self.connect_to[new_road] = (all_self_lanes_end[0], 'continue', self_indices, None, None)
            # Since roadlen dict is shared across all lanes, we only need to update it via one of
            # the lanes
            self.lanes[0].roadlen[new_road.name] = all_new_lanes_start[0] - all_self_lanes_end[0]
            for self_ind, new_road_ind in zip(self_indices, new_road_indices):
                # TODO: finish the impl for lane's connection
                self.lanes[self_ind].events.append({'event': 'new lane', 'pos': self.lanes[self_ind].end})

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
        # Update self road's connect_to
        self.connect_to[new_road.name] = (self_pos, 'merge', self_index, change_side, new_road)
        assert isinstance(self_pos, tuple) and isinstance(new_lane_pos, tuple)
        # Update roadlen
        self.lanes[0].roadlen[new_road.name] = new_lane_pos[0] - self_pos[0]
        new_road.lanes[0].roadlen[self.name] = self_pos[0] - new_lane_pos[0]
        # Update lane events and connect_left/connect_right for both lanes
        if change_side == 'l_lc':
            connect_lane_left_right(new_road[new_lane_index], self.lanes[self_index], new_lane_pos, self_pos)
        else:
            connect_lane_left_right(self.lanes[self_index], new_road[new_lane_index], self_pos, new_lane_pos)

    def diverge(self):
        """
        TODO: implement it
        """
        pass

    def __getitem__(self, index):
        if isinstance(index, str):
            if index == 'name':
                return self.name
            elif index == 'connect to':
                return self.connect_to
            elif index == 'laneinds':
                return self.num_lanes
            else:
                assert 0
        assert isinstance(index, int) and 0 <= index < self.num_lanes
        return self.lanes[index]
