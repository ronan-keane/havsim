from havsim.simulation.road_networks import Lane


class Road:
    def __init__(self, num_lanes, length, name, road_length_offset, connections=None):
        """
        num_lanes: number of lanes (int)
        length: if float/int, the length of all lanes. If a list, each entry gives the (start, end) tuple
          for the corresponding lane
        name: name of the road
        road_length_offset: a dictionary used to calculate head ways or position offset for lane changes
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
            assert isinstance(length, list) and len(length) == num_lanes

        # Construct lane objects for the road
        self.lanes = []
        for i in range(num_lanes):
            lane = Lane(start=length[i][0], end=length[i][1], road=self, laneind=i, connect_left=[], connect_right=[])
            lane.roadlen = road_length_offset
            self.lanes.append(lane)

        # Connect adjacent lanes
        for i in range(num_lanes):
            lane = self.lanes[i]
            left_lane = self.lanes[i - 1] if i > 0 else None
            right_lane = self.lanes[i + 1] if i + 1 < num_lanes else None
            if connections is not None:
                connection = connections[i]
            else:
                left_connection = (lane.start, None) if left_lane is None else (
                    max(left_lane.start, lane.start), min(left_lane.end, lane.end))
                right_connection = (lane.start, None) if right_lane is None else (
                    max(right_lane.start, lane.start), min(right_lane.end, lane.end))
                connection = (left_connection, right_connection)
            Road._connect_lane(lane, left_lane, right_lane, connection)

    @staticmethod
    def _connect_lane(lane, left_lane, right_lane, connection):
        if left_lane is None:
            lane.connect_left.append(connection[0])
        else:
            if connection[0][0] > lane.start:
                lane.connect_left.append((lane.start, None))
            lane.connect_left.append((connection[0][0], left_lane))
            if connection[0][1] < lane.end:
                lane.connect_left.append((connection[0][1], None))
        if right_lane is None:
            lane.connect_right.append(connection[1])
        else:
            if connection[1][0] > lane.start:
                lane.connect_right.append((lane.start, None))
            lane.connect_right.append((connection[1][0], right_lane))
            if connection[1][1] < lane.end:
                lane.connect_right.append((connection[1][1], None))

    def connects(self, new_road, self_indices, new_road_indices):
        # TODO: implement this
        # this would only work for 'continue' type
        # how to add the 'exit' connection ? different method for that?
        assert len(self_indices) == len(new_road_indices)
        # would just call the lane.connects method

    def __getitem__(self, index):
        assert isinstance(index, int) and 0 <= index < self.num_lanes
        return self.lanes[index]


# Lane should have a new 'connects' method as well?
class Lane:
    def connects(self, newlane):  # new connects method for Lane can handle 'merge' type or 'continue' type
        pass