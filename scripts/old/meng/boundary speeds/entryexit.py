"""
@author: rlk268@cornell.edu
"""

import pickle
import numpy as np
from havsim.calibration.algs import makeplatoonlist
from havsim.plotting import platoonplot, animatetraj

#path_reconngsim = '/Users/yidan/Downloads/reconngsim.pkl'
#path_highd26 = '/Users/yidan/Downloads/highd26.pkl'
#
## ngsim data
#with open(path_reconngsim, 'rb') as f:
#    reconngsim = pickle.load(f)[0]
## highd data
#with open(path_highd26, 'rb') as f:
#    highd = pickle.load(f)[0]

meas, platooninfo = makeplatoonlist(data, 1, False)  # form meas for ngsim data
# note time discretization = .1 seconds for ngsim, .04 seconds for highd

# %% toy example
# create toy example as just a small portion of the full data
platoon = [875.0, 903.0, 908.0]  # dataset only with these vehicles
toymeas = {}
for i in platoon:
    toymeas[i] = meas[i].copy()
toymeas[875][:, 4] = 0
toymeas[908][:, 5] = 0
# plot all the vehicles, you can click on lines to see which vehicle IDs the trajectories correspond to
#platoonplot(meas, None, platooninfo, platoon=platoon, colorcode=False, lane=3)
# you can view the vehicle trajectories in this animation if you want
#ani = animatetraj(meas, platooninfo, platoon=platoon)

# expected output:

# for lane 4 - note the first 3 observations of vehicle 875 are in lane 4, so these 3 observations define the entry/exit speeds
# for lane 4, which are defined only for those 3 times.

# for lane 3 -
# exit speeds will be vehicle 875 starting from its third observation, vehicle 903 from 3083 - 3104, vehicle 908 from 3105 - 3139
# entry speeds are going to be the speeds of vehicle 875 from times 2555 - 2573,
# speeds of vehicle 903 from times 2574 - 2620, vehicle 908 from 2621 - 2660

# please verify what I've written above is correct and manually form the output for testing purposes
# hint: you can figure out what the times are supposed to be since we know 875 defines the exit speeds until it's last observation,
# then as soon as 875 last observation time is passed, 903 takes over the exit speeds, which continues until its last observation, etc.

"""
Record what the output should be for boundaryspeeds(toymeas,[3,4],[3,4],.1,.1)
"""


# ================ My solution below ===================
# first of all, your result on lane 4 is correct while the result on lane 3 is not.
# on lane 4, exit/entry speeds would be the speed of car 875 from 2555 to 2557 (both inclusive, same below)
# on lane 3
# entry speeds:
# speed of car 875 from 2558 to 2573 -> speed of car 903 from 2574 to 2620 -> speed of car 908 from 2621 to 3139
# exit speeds:
# speed of car 875 from 2558 to 3082 -> speed of car 903 from 3083 to 3104 -> speed of car 908 from 3105 to 3139
# (so you got the entry speeds wrong for lane 3)

# After rethinking the algo, I now believe that using the original data does not provide much of
# a performance improvement. So for now I just stick to the original function signature except adding car_ids param
# to take a subset of data from meas

# Also, since the timestamp is always integer type for any timeind, I assume that 1 unit diff in time is
# equivalent to timeind. As a result, only outtimeind/timeind matters for our function


def boundaryspeeds(meas, entrylanes, exitlanes, timeind, outtimeind, car_ids=None):
    #car_ids is a list of vehicle IDs, only use those values in meas 
    
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
        entry_data_for_lane = entry_data[entry_data[:, -2] == entrylane][:, [1, 3]]
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
    # interval: aggregation units.
    # returns: (aggregated_speed_list, (start_time_of_first_interval, start_time_of_last_interval))
    if not len(data):
        return list(), ()
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


def test_interpolate():
    assert interpolate(np.array([])) == ([], ())
    assert interpolate(np.array([[1, 5]])) == ([5], (1, 1))
    assert interpolate(np.array([[1, 5], [2, 6], [3, 7]]), 1) == ([5.0, 6.0, 7.0], (1, 3))
    assert interpolate(np.array([[1, 5], [2, 6], [3, 7]]), interval=2.5) == ([(5 + 6 + 7 / 2.0) / 2.5], (1, 1.0))
    assert interpolate(np.array([[1, 5], [2, 6], [4, 7]]), interval=2.5) == (
        [(5 + 6 + 3) / 2.5, (3 + 7 + 7) / 2.5], (1, 3.5))


def test_boundaryspeeds():
    result_to_test = boundaryspeeds(meas, [3, 4], [3, 4], .1, .1, car_ids=[875.0, 903.0, 908.0])
    # according to the result we compute manually above
    car875 = toymeas[875]
    car903 = toymeas[903]
    car908 = toymeas[908]
    lane3entry = list(car875[np.isin(car875[:, 1], range(2558, 2574))][:, 3]) + list(
        car903[np.isin(car903[:, 1], range(2574, 2621))][:, 3]) + list(
        car908[np.isin(car908[:, 1], range(2621, 3140))][:, 3])
    lane3exit = list(car875[np.isin(car875[:, 1], range(2558, 3083))][:, 3]) + list(
        car903[np.isin(car903[:, 1], range(3083, 3105))][:, 3]) + list(
        car908[np.isin(car908[:, 1], range(3105, 3140))][:, 3])
    lane4entry = list(car875[np.isin(car875[:, 1], range(2555, 2558))][:, 3])
    lane4exit = list(lane4entry)
    expected_result = [lane3entry, lane4entry], [(2558, 3139), (2555, 2557)], [lane3exit, lane4exit], [(2558, 3139),
                                                                                                       (2555, 2557)]
    assert result_to_test == expected_result


test_interpolate()
test_boundaryspeeds()
