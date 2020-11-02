from havsim.simulation.road_networks import Lane, connect_lane_left_right


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

    def connects(self, new_road, self_indices=None, new_road_indices=None, is_exit=False):
        """
        new_road: road object to make the connection to. If passing in a string, it's assumed
            to be the name of the exit road (need to mark is_exit as True)
        self_indices: a list of indices of the current road for making the connection
        new_road_indices: a list of indices of the new road to connect to
        is_exit: whether the new road is an exit. An exit road won't have an road object.
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
                self.lanes[i].connects(new_lane=new_road, connect_type='exit')
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
                self.lanes[self_ind].connects(new_road.lanes[new_road_ind], connect_type='continue')

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
