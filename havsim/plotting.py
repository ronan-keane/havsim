"""
where all the plotting functions go

@author: rlk268@cornell.edu
"""
# TODO fix code style, add documentation and examples
import numpy as np
import copy
import math

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib import cm
import pywt
from statistics import harmonic_mean
from matplotlib.widgets import PolygonSelector
import palettable

import havsim.helper as helper
import havsim.calibration as hc


def plotColorLines(X, Y, SPEED, speed_limit, colormap = 'speeds', ind = 0):
    """X and Y are x/y data to plot, SPEED gives the color for each data pair."""

    #helper for platoonplot
    axs = plt.gca()
    c = SPEED
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # if speed_limit:
    # 	norm = plt.Normalize(speed_limit[0], speed_limit[1])
    # else:
    # 	norm = plt.Normalize(c.min(), c.max())
    norm = plt.Normalize(speed_limit[0], speed_limit[1])
    if colormap =='speeds':
        lc = LineCollection(segments, cmap=palettable.colorbrewer.diverging.RdYlGn_4.mpl_colormap, norm=norm)
        lc.set_linewidth(1)
    elif colormap =='times':
        cmap_list = [palettable.colorbrewer.sequential.Blues_9.mpl_colormap, palettable.colorbrewer.sequential.Oranges_9.mpl_colormap,
                     palettable.colorbrewer.sequential.Greens_9.mpl_colormap, palettable.colorbrewer.sequential.Greys_9.mpl_colormap]

#        lc = LineCollection(segments, cmap=plt.get_cmap('viridis'), norm=norm)
        if ind > len(cmap_list)-1:
            ind = len(cmap_list)-1
        lc = LineCollection(segments, cmap=cmap_list[ind], norm=norm)
        lc.set_linewidth(1)

#    lc = LineCollection(segments, cmap=cm.get_cmap('RdYlBu'), norm=norm)
    lc.set_array(c)
    line = axs.add_collection(lc)
    return line


def plot_format(all_vehicles, laneinds):
    # changes format from the output of simulation module into a format consistent with plotting functions
    # all_vehicles - set of all vehicles to convert
    # laneinds - dictionary where lanes are keys, values are the index we give them

    #outputs - meas and platooninfo, no follower or acceleration column in meas
    meas = {}
    platooninfo = {}
    for veh in all_vehicles:
        vehid = veh.vehid
        starttime = veh.starttime  # start and endtime are in real time, not slices time
        if not veh.endtime:
            endtime = veh.starttime + len(veh.speedmem) -1
        else:
            endtime = veh.endtime
        curmeas = np.empty((endtime - starttime + 1, 9))
        curmeas[:,0] = veh.vehid
        curmeas[:,1] = list(range(starttime, endtime+1))
        curmeas[:,2] = veh.posmem
        curmeas[:,3] = veh.speedmem
        curmeas[:,6] = veh.len

        # lane indexes
        memlen = len(veh.lanemem)
        for count, lanemem in enumerate(veh.lanemem):
            time1 = lanemem[1]
            if count == memlen - 1:
                time2 = endtime + 1
            else:
                time2 = veh.lanemem[count+1][1]
            curmeas[time1-starttime:time2-starttime,7] = laneinds[lanemem[0]]

        # leaders
        memlen = len(veh.leadmem)
        for count, leadmem in enumerate(veh.leadmem):
            time1 = leadmem[1]
            if count == memlen - 1:
                time2 = endtime + 1
            else:
                time2 = veh.leadmem[count+1][1]

            if leadmem[0] is None:
                useind = 0
            else:
                useind = leadmem[0].vehid
            curmeas[time1-starttime:time2-starttime,4] = useind

        # times for platooninfo
        lanedata = curmeas[:,[1,4]]
        lanedata = lanedata[lanedata[:,1] != 0]
        unused, indjumps = helper.checksequential(lanedata)
        if np.all(indjumps == [0,0]):
            time1 = starttime
            time2 = starttime
        else:
            time1 = lanedata[indjumps[0],0]
            time2 = lanedata[indjumps[1]-1,0]

        # make output
        platooninfo[vehid] = [starttime, time1, time2, endtime]
        meas[vehid] =  curmeas

    return meas, platooninfo



def platoonplot(meas, sim, platooninfo, platoon=[], newfig=True, clr=['C0', 'C1'],
                fulltraj=True, lane=None, opacity=.4, colorcode=True, speed_limit=[], timerange=[None, None]):  # plot platoon in space-time
    # meas - measurements in np array, rows are observations
    # sim - simulation in same format as meas. can pass in None and only meas will be shown, or can pass in the data and they will be plotted together
    # in different colors.
    # platooninfo (platooninfo) - dictionary containing information on each vehicle ID
    # platoon - default is [], in which case all keys of platooninfo are plotted. If passed in as a platoon (list of vehicle ID as [1:] so first entry not included)
    # only those vehicles will be plotted.

    # newfig = True - if True will create a new figure, otherwise it will use the current figure
    # clr = 'C0', assuming Colors = False, clr will control what colors will be used. Default is ['C0','C1'] which are the default matplotlib colors
    # this is used is sim is not None and colorcode = False
    # fulltraj = True controls how much of each trajectory to plot

    # lane = None - If passed in as a laneID, the parts of trajectories not in the lane ID given will be made opaque
    # colorcode = True - if colorcode is True, sim must be None, and we will plot the trajectories
    # colorcoded based on their speeds. It looks nice!
    # speed_limit = [] - only used when colorcode is True, if empty we will find the minimum and maximum speeds
    # and colorcode based on those speeds. Otherwise you can specify the min/max, and anything below/above
    # those limits will be colorcoded according to the limits
    # timerange = [None, None] - If fulltraj is True, this parameter is ingored
    # Otherwise, if values are passed in, only plot the trajectories in the provided time range

    # plots a platoon of vehicles in space-time plot.
    # features - can click on vehicles to display their IDs. Can compare meas and sim when colorcode is False.
    # can specify a lane, and make trajectories outside of that lane opaque.
    # can colorcode trajectories based on their speeds to easily see shockwaves and other structures.

    if sim is not None:
        colorcode = False

    ind = 2
    artist2veh = []

    if platoon != []:
        platooninfo = helper.platoononly(platooninfo, platoon)
    followerlist = list(platooninfo.keys())  # list of vehicle ID
    if lane != None:
        for i in followerlist.copy():
            if lane not in np.unique(meas[i][:,7]):
                followerlist.remove(i)
    if newfig:
        fig = plt.figure()


    counter = 0
    mymin = 1e10
    mymax = 0
    for i in followerlist:
        curmin = min(meas[i][:, 3])
        curmax = max(meas[i][:, 3])
        if mymin > curmin:
            mymin = curmin
        if mymax < curmax:
            mymax = curmax

    if not speed_limit:
        speed_limit = [mymin, mymax]

    for i in followerlist:  # iterate over each vehicle
        veh = meas[i]
        veh = extract_relevant_data(veh, i, platooninfo, fulltraj, timerange)

        # If current vehicle's data is irrelavant to given time range, move on to the next one
        if veh is None:
            continue

        x = veh[:, 1]
        y = veh[:, ind]
        speed_list = veh[:, 3]

        LCind = generate_LCind(veh, lane)

        for j in range(len(LCind) - 1):
            kwargs = {}
            if lane is not None and veh[LCind[j], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': opacity}  # dashed line .4 opacity (60% see through)
                plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[0], **kwargs)
                artist2veh.append(counter)
            else:

                X = x[LCind[j]:LCind[j + 1]]
                Y = y[LCind[j]:LCind[j + 1]]
                SPEED = speed_list[LCind[j]:LCind[j + 1]]
                if colorcode:
                    line = plotColorLines(X, Y, SPEED, speed_limit=speed_limit)

                else:
                    plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[0], picker=5, **kwargs)
                    artist2veh.append(counter)

        counter += 1


    if sim != None:
        counter = 0
        for i in followerlist:  # iterate over each vehicle
            veh = sim[i]
            veh = extract_relevant_data(veh, i, platooninfo, fulltraj, timerange)

            if len(veh) == 0:
                continue

            x = veh[:, 1]
            y = veh[:, ind]

            LCind = generate_LCind(veh, lane)

            for j in range(len(LCind) - 1):
                kwargs = {}
                if veh[LCind[j], 7] != lane and lane != None:
                    kwargs = {'linestyle': '--', 'alpha': .4}  # dashed line .4 opacity (60% see through)
                plt.plot(x[LCind[j]:LCind[j + 1]], y[LCind[j]:LCind[j + 1]], clr[1], **kwargs)

            counter += 1

    find_artists = []
    nartists = len(artist2veh)

    def on_pick(event):
        nonlocal find_artists
        ax = event.artist.axes
        curind = ax.lines.index(event.artist)  # artist index

        if event.mouseevent.button == 1:  # left click selects vehicle
            # deselect old vehicle
            for j in find_artists:
                ax.lines[j].set_color('C0')
                if sim != None:
                    ax.lines[j+nartists].set_color('C1')

            # select new vehicle
            vehind = artist2veh[curind]  # convert from artist to vehicle index
            find_artists = np.asarray(artist2veh)
            find_artists = np.nonzero(find_artists == vehind)[0]  # all artist indices which are associated with vehicle

            for j in find_artists:
                ax.lines[j].set_color('C3')
                if sim != None:
                    ax.lines[j+nartists].set_color('C3')
            plt.title('Vehicle ID ' + str(list(followerlist)[vehind]))
            plt.draw()
        plt.draw()

    # recolor the selected artist and all other artists associated with the vehicle ID so you can see what line you clicked on
    # change the title to say what vehicle you selected.

    fig.canvas.callbacks.connect('pick_event', on_pick)
    axs = plt.gca()

    plt.xlabel('time (frameID )')
    plt.ylabel('space (ft)')

    if colorcode:
        fig.colorbar(line, ax=axs)

    axs.autoscale(axis='x')
    axs.autoscale(axis='y')

    return

def extract_relevant_data(veh, i, platooninfo, fulltraj, timerange):
    #if fulltraj is True, plot between t_nstar - T_n; plot between t_n and T_nm1 otherwise
    #trajectories additionally must be between timerange if possible
    t_nstar, t_n, T_nm1, T_n = platooninfo[i][0:4]

    if fulltraj:  # entire trajectory including pre simulation and shifted end
        start = t_nstar
        end = T_n
    else:  # only show trajectory which is simulated
        start = t_n
        end = T_nm1

    if timerange[0] != None:
        if timerange[0] <= end:
            if start > timerange[0]:
                pass
            else:
                start = timerange[0]
        else:
            start = None

    if timerange[1] != None:
        if end < timerange[1]:
            pass
        else:
            end = timerange[1]

    if start == None:
        return np.zeros((0,8))
    return veh[start-t_nstar:end-t_nstar+1, :]


def generate_LCind(veh, lane):
    if lane != None:
        # LCind is a list of indices where the lane the vehicle is in changes. Note that it includes the first and last index.
        LCind = np.diff(veh[:, 7])
        LCind = np.nonzero(LCind)[0] + 1
        LCind = list(LCind)
        LCind.insert(0, 0)
        LCind.append(len(veh[:, 7]))

    else:
        LCind = [0, len(veh[:,1])]

    return LCind

def overlap(interval1, interval2):
    #given two tuples of start - end times, computes overlap between them
    #can pass None as either of values in interval2 to get better data
    outint = interval1.copy()
    if interval2[0] != None:
        if interval2[0] <= interval1[1]:
            if interval2[0] > interval1[0]:
                outint[0] = interval2[0]
        else:
            return None
    if interval2[1] is not None:
        if interval2[1] < interval1[1]:
            outint[1] = interval2[1]

    return outint

def generate_changetimes(veh, col_index):
    #returns list of indices [ind] (from 0 index of whatever is passed in) where
    #veh[ind, col_index] is different from veh[ind-1, col_index]. Then to slice the different blocks,
    #you can use veh[ind[0]:ind[1], col_index] where blocks have the same value repeated.

    #this is a generalization of generate_LCind
    ind = np.diff(veh[:, col_index])
    ind = np.nonzero(ind)[0] + 1
    ind = list(ind)
    ind.insert(0, 0)
    ind.append(len(veh[:, col_index]))

    return ind


def plotflows(meas, spacea, timea, agg, type='FD', FDagg=None, lane = None, method = 'area', h = .1):
    """
	aggregates microscopic data into macroscopic quantities based on Edie's generalized definitions of traffic variables

	meas = measurements, in usual format (dictionary where keys are vehicle IDs, values are numpy arrays)

	spacea = reads as ``space A'' (where A is the region where the macroscopic quantities are being calculated).
    list of lists, each nested list is a length 2 list which ... represents the starting and ending location on road.
    So if len(spacea) >1 there will be multiple regions on the road which we are tracking e.g. spacea = [[200,400],[800,1000]],
    calculate the flows in regions 200 to 400 and 800 to 1000 in meas.

	timea = reads as ``time A'', should be a list of the times (in the local time of thedata).
    E.g. timea = [1000,3000] calculate times between 1000 and 3000.

	agg = aggregation length, float number which is the length of each aggregation interval.
    E.g. agg = 300 each measurement of the macroscopic quantities is over 300 time units in the data,
    so in NGSim where each time is a frameID with length .1s, we are aggregating every 30 seconds.

	type = `FD', if type is `FD', plot data in flow-density plane. Otherwise, plot in flow-time plane.

	FDagg = None - If FDagg is None and len(spacea) > 1, aggregate q and k measurements together.
    Otherwise if FDagg is an int, only show the q and k measurements for the corresponding spacea[int]

    lane = None - If lane is given, it only uses measurement in that lane.

    h = .1 - time discretizatino in data - passed in to calculateflows

    Note that if the aggregation intervals are too small the plots won't really make sense
    because a lot of the variation is just due to the aggregation. Increase either agg
    or spacea regions to prevent this problem.
	"""
    intervals = []
    start = timea[0]
    end = timea[1]
    temp1 = start
    temp2 = start + agg
    while temp2 < end:
        intervals.append((temp1, temp2))
        temp1 = temp2
        temp2 += agg
    intervals.append((temp1, end))

    q, k = helper.calculateflows(meas, spacea, timea, agg, lane = lane, method = method, h = h)
    time_sequence = []
    time_sequence_for_line = []

    if len(q) > 1 and FDagg != None:
        q = [q[FDagg]]
        k = [k[FDagg]]

    for i in range(len(q)):
        for j in range(len(intervals)):
            time_sequence.append(intervals[j][0])

    for i in range(len(intervals)):
        time_sequence_for_line.append(intervals[i][0])
    # unzipped_q = []
    # for i in q:
    #     unzipped_q += i
    # unzipped_k = []
    # for i in k:
    #     unzipped_k += i

    if type == 'FD':
        marker_list = ['o', 'x']
        #different marker types
        for count, curq in enumerate(q):
            curmarker = marker_list[count]
            curk = k[count]
            plt.scatter(curk, curq, c=time_sequence_for_line, cmap=cm.get_cmap('viridis'), marker = curmarker)
        # plt.scatter(unzipped_k, unzipped_q, c=time_sequence, cmap=cm.get_cmap('viridis'))
        plt.colorbar()
        plt.xlabel("density")
        plt.ylabel("flow")
        plt.show()

    elif type == 'line':
        for i in range(len(spacea)):
            plt.plot(time_sequence_for_line, q[i])
        print(q)
        plt.xlabel("time")
        plt.ylabel("flow")
        plt.show()

    return


def plotvhd(meas, sim, platooninfo, vehicle_id, draw_arrow=False, arrow_interval=10, effective_headway=False, rp=None, h=.1,
            datalen=9, timerange=[None, None], lane=None, delay=0, newfig=True, plot_color_line=False):
    # draw_arrow = True: draw arrows (indicating direction) along with trajectories; False: plot the trajectories only
    # effective_headway = False - if True, computes the relaxation amounts using rp, and then uses the headway + relaxation amount to plot instead of just the headway
    # rp = None - effective headway is true, rp is a float which is the parameter for the relaxation amount
    # h = .1 - data discretization
    # datalen = 9
    # timerange = [None, None] indicates the start and end timestamps that the plot limits
    # lane = None, the lane number that need highlighted: Trajectories in all other lanes would be plotted with opacity
    # delay = 0 - gets starting time for newell model
    # newfig = True - if True will create a new figure, otherwise it will use the current figure
    # plot_color_line = False; If set to true, plot all trajectories using colored lines based on timestamp

    ####plotting
    if newfig:
        plt.figure()
    plt.xlabel('space headway (ft)')
    plt.ylabel('speed (ft/s)')
    title_text = 'space-headway for vehicle ' + " ".join(list(map(str, (vehicle_id))))
    if lane is not None:
        title_text = title_text + ' on lane ' + str(lane)
    plt.title(title_text)
    ax = plt.gca()
    artist_list = []

    if sim is None:
        # If sim is None, plot meas for all vehicles in vehicle_id
        for count, my_id in enumerate(vehicle_id):
            ret_list = process_one_vehicle(ax, meas, sim, platooninfo, my_id, timerange, lane, plot_color_line, effective_headway, rp, h, datalen, delay, count = count )
            artist_list.extend(ret_list)
    else:
        # If both meas and sim are provided,
        # will plot both simulation and measurement data for the first vehicle in vehicle_id
        if len(vehicle_id) > 1:
            print('plotting first vehicle '+str(vehicle_id[0])+' only')
        ret_list = process_one_vehicle(ax, meas, sim, platooninfo, vehicle_id[0], timerange, lane, plot_color_line, effective_headway, rp, h, datalen, delay)
        artist_list.extend(ret_list)

    if plot_color_line:
        ax.autoscale(axis = 'x')
        ax.autoscale(axis = 'y')
    else:
        organize_legends()

    if draw_arrow:
        for art in artist_list:
            if plot_color_line:
                add_arrow(art, arrow_interval, plot_color_line = plot_color_line)
            else:
                add_arrow(art[0], arrow_interval)

    return

# This function will process and prepare xy-coordinates, color, labels, etc.
# necessary to plot trajectories for a given vehicle and then invoke plot_one_vehicle() function to do the plotting
def process_one_vehicle(ax, meas, sim, platooninfo, my_id, timerange, lane, plot_color_line, effective_headway=False, rp=None, h=.1, datalen=9, delay=0, count = 0):
    artist_list = []
    if effective_headway:
        leadinfo, folinfo, rinfo = helper.makeleadfolinfo([my_id], platooninfo, meas)
    else:
        leadinfo, folinfo, rinfo = helper.makeleadfolinfo([my_id], platooninfo, meas, relaxtype = 'none')

    t_nstar, t_n, T_nm1, T_n = platooninfo[my_id][0:4]

    # Compute and validate start and end time
    start, end = compute_validate_time(timerange, t_n, T_nm1, h, delay)

    frames = [t_n, T_nm1]
    relax, unused = helper.r_constant(rinfo[0], frames, T_n, rp, False, h)  # get the relaxation amounts for the current vehicle; these depend on the parameter curp[-1] only.
    meas_label = str(my_id)
    meas_color = next(ax._get_lines.prop_cycler)['color']

    headway = None
    if sim is not None:
        headway = compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, sim, my_id, relax)
        sim_color = next(ax._get_lines.prop_cycler)['color']
        meas_label = 'Measurements'
        ret_list = plot_one_vehicle(headway[:end + 1 - start], sim[my_id][start - t_nstar:end + 1 - t_nstar, 3],
                                            sim[my_id][start - t_nstar:end + 1 - t_nstar, 1],
                                            sim[my_id][start - t_nstar:end + 1 - t_nstar, 7],
                                            lane, plot_color_line, leadinfo, start, end, 'Simulation', sim_color, count = count)
        artist_list.extend(ret_list)

    trueheadway = compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, meas, my_id, relax)
    ret_list = plot_one_vehicle(trueheadway[:end + 1 - start], meas[my_id][start - t_nstar:end + 1 - t_nstar, 3],
                                            meas[my_id][start - t_nstar:end + 1 - t_nstar, 1],
                                            meas[my_id][start - t_nstar:end + 1 - t_nstar, 7],
                                            lane, plot_color_line, leadinfo, start, end, meas_label, meas_color, count = count)
    artist_list.extend(ret_list)
    return artist_list


def plot_one_vehicle(x_coordinates, y_coordinates, timestamps, lane_numbers, target_lane, plot_color_line, leadinfo, start, end, label, color, opacity=.4, count = 0):
    # If there is at least a leader change,
    # we want to separate data into multiple sets otherwise there will be horizontal lines that have no meanings
    # x_coordinates and y_coordinates will have the same length,
    # and x_coordinates[0] and y_coordinates[0] have the same time frame == start
    # lane_numbers list has corresponding lane number for every single y_coordinates (speed)
    temp_start = 0
    leader_id = leadinfo[0][0][0]
    artist_list = []

    ##############################
#    if plot_color_line:
#        lines = plotColorLines(x_coordinates, y_coordinates, timestamps, [timestamps[0], timestamps[-1]])
#        return artist_list
    ##############################

    for index in range(0, len(x_coordinates)):
        current_leader_id = find_current_leader(start + index, leadinfo[0])
        if current_leader_id != leader_id:
            # Detected a leader change, plot the previous set
            leader_id = current_leader_id

            # Check if should do color line plotting
            if plot_color_line:
                lines = plotColorLines(x_coordinates[temp_start:index], y_coordinates[temp_start:index], timestamps[temp_start:index], [start- 100, end+10], colormap = 'times', ind = count)
                artist_list.append((lines, [start-100, end+10]))
            else:
                kwargs = {}
                # Check if lane changed as well, if yes, plot opaque lines instead
                if lane_numbers[temp_start] != target_lane and target_lane is not None:
                    kwargs = {'alpha': opacity}  # .4 opacity (60% see through)
                art = plt.plot(x_coordinates[temp_start:index], y_coordinates[temp_start:index], label=label, color=color, linewidth = 1.2, **kwargs)
                artist_list.append(art)

            temp_start = index

    # Plot the very last set, if there is one
    if plot_color_line:
        lines = plotColorLines(x_coordinates[temp_start:], y_coordinates[temp_start:], timestamps[temp_start:], [start-100, end+10], colormap = 'times', ind = count)
        artist_list.append((lines, [start-100, end+10]))
    else:
        kwargs = {}
        if lane_numbers[temp_start] != target_lane and target_lane is not None:
            kwargs = {'alpha': opacity}  # .4 opacity (60% see through)
        art = plt.plot(x_coordinates[temp_start:], y_coordinates[temp_start:], label=label, color=color, linewidth = 1.2, **kwargs)
        artist_list.append(art)
    return artist_list

# This function is used to merge legends (when necessary) especially the same vehicle has multiple trajectories sections
# due to leader or lane changes
def organize_legends():
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    plt.legend(newHandles, newLabels)

def add_arrow(line, arrow_interval=20, direction='right', size=15, color=None, plot_color_line = False):
    """
    add an arrow to a line.

    line:           Line2D object
    arrow_interval: the min length on x-axis between two arrows, given a list of xdata,
                    this can determine the number of arrows to be drawn
    direction:      'left' or 'right'
    size:           size of the arrow in fontsize points
    color:          if None, line color is taken.
    plot_color_line = True - if True, line is a tuple of (line collection object, norm) not line2d object
    """
    if plot_color_line:
        line, norm = line[:] #norm is min/max float

        my_cmap = line.get_cmap()
        colorarray = line.get_array() #floats used to color data

        def color_helper(index): #gets the color (in terms of matplotlib float array format) from index
            myint = (colorarray[index]-1 - norm[0])/(norm[1] - norm[0]+1)
            return my_cmap(myint)

        temp = line.get_segments() #actual plotting data
        xdata = [temp[0][0][0]]
        ydata = [temp[0][0][1]]
        for i in temp:
            xdata.append(i[1][0])
            ydata.append(i[1][1])

    else:
        if color == None:
            color = line.get_color()

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        def color_helper(*args):
            return color
    curdist = 0
    line.axes.annotate('',
                xytext=(xdata[0], ydata[0]),
                xy=(xdata[1], ydata[1]),
                arrowprops=dict(arrowstyle="->", color=color_helper(0)),
                size=size
            )
    for i in range(len(xdata)-1):
        curdist += ((xdata[i+1] - xdata[i])**2 + (ydata[i+1] - ydata[i])**2 )**.5
        if curdist > arrow_interval:
            curdist += - arrow_interval
            start_ind = i

            if start_ind == 0 or start_ind == len(xdata) - 1:
                continue

#            if direction == 'right':
            end_ind = start_ind + 1
#            else:
#                end_ind = start_ind - 1
            line.axes.annotate('',
                xytext=(xdata[start_ind], ydata[start_ind]),
                xy=(xdata[end_ind], ydata[end_ind]),
                arrowprops=dict(arrowstyle="->", color=color_helper(start_ind)),
                size=size
            )



def animatevhd(meas, sim, platooninfo, platoon, lentail=20, timerange=[None, None],
               lane = None, opacity = .2, interval = 10, rp = None, h=.1, delay=0):
    # plot multiple vehicles in phase space (speed v headway)
    #meas, sim - data in key = ID, value = numpy array format, pass sim = None to plot one set of data
    #platooninfo
    # platoon - list of vehicles to plot
    # lentail = 20 - number of observations to show in the past
    # timerange = [usestart, useend]
    # rp = None - can add relaxation to the headway, if you pass a number this is used as relaxation amount
    # h = .1 - data discretization, deprecated
    # delay = 0 - gets starting time for newell model, deprecated
    #lane = None - can specify a lane to make trajectories opaque if not in desired lane
    #opacity = .2 - controls opacity (set = 0 to not show, if 1 its equivalent to lane = None)

    #I think this function has good general design for how a animation for traffic simulation should be structured in python
    #each vehicle has a dictionary, which contains relevant plotting data and any extra information (keyword args, start/end times, etc)
    #create a sorted list with tuples of the (times, dictionary, 'add' or 'remove') which represent when artists (vehicles)
    #will enter or leave animation. THen in animation, in each frame check if there are any artists to add or remove;
    #if you add a vehicle, create an artist (or potentially multiple artists) and add its reference to the dictionary
    #keep a list of all active dictionaries (vehicles) during animation - update artists so you can use blitting and
    #dont have to keep redrawing - its faster and animation is smoother this way.
    fig = plt.figure()
    plt.xlabel('space headway (ft)')
    plt.ylabel('speed (ft/s)')
    plt.title('space-headway for vehicle ' + " ".join(list(map(str, (platoon)))))
    plotsim = False if sim is None else True
    xmin, xmax, ymin, ymax = math.inf, -math.inf, math.inf, -math.inf

    startendtimelist = []

    for veh in platoon:
        t_nstar, t_n, T_nm1, T_n = platooninfo[veh][:4]
        #heuristic will speed up plotting if a large dataset is passed in
        if timerange[0] is not None:
            if T_nm1 < timerange[0]:
                continue
        if timerange[1] is not None:
            if t_n > timerange[1]:
                continue

        #compute headway, speed between t_n and T_nm1
        headway = compute_headway2(veh, meas, platooninfo, rp, h)
        speed = meas[veh][t_n-t_nstar:T_nm1-t_nstar+1,3]
        if plotsim:
            simheadway = compute_headway2(veh, sim, platooninfo, rp, h)
            simspeed = sim[veh][t_n-t_nstar:T_nm1-t_nstar,3]

        curxmin, curxmax, curymin, curymax = min(headway), max(headway), min(speed), max(speed)
        xmin, xmax, ymin, ymax = min([xmin, curxmin]), max([xmax, curxmax]), min([ymin, curymin]), max([ymax, curymax])

        #split up headway/speed into sections based on having a continuous leader
        #assume that sim and measurements have same leaders in this code
        ind = generate_changetimes(meas[veh][t_n-t_nstar:T_nm1-t_nstar+1,:], 4)
        for i in range(len(ind)-1):
            #each section has the relevant speed, headway, start and end times, and opaque.
            newsection = {}

            #start and end times are in real time (not slices indexing).
            start = ind[i] + t_n
            end = ind[i+1]-1 + t_n
            curlane = meas[veh][start - t_nstar, 7]
            times = overlap([start, end], timerange) #times of section to use, in real time
            if times == None:
                continue
            newsection['hd'] = headway[times[0] - t_n:times[1]+1-t_n ]
            newsection['spd'] = speed[times[0] - t_n:times[1]+1-t_n ]
            newsection['start'] = times[0]
            newsection['end'] = times[1]
            kwargs = {'color': 'C0'}
            if lane != None and curlane != lane:
                kwargs['alpha'] = opacity
            newsection['kwargs'] = kwargs
            newsection['veh'] = str(int(veh))

            if plotsim:
                #literally the same thing repeated
                newsimsection = {}
                newsimsection['hd'] = simheadway[times[0] - t_n:times[1]+1-t_n ]
                newsimsection['spd'] = simspeed[times[0] - t_n:times[1]+1-t_n ]
                newsimsection['start'] = times[0]
                newsimsection['end'] = times[1]
                kwargs = {'color': 'C1'}
                if lane != None and curlane != lane:
                    kwargs['alpha'] = opacity
                newsimsection['kwargs'] = kwargs
                newsimsection['veh'] = str(int(veh))

            startendtimelist.append((times[0], newsection, 'add'))
            startendtimelist.append((times[1]+lentail+1, newsection, 'remove'))
            if plotsim:
                startendtimelist.append((times[0], newsimsection, 'add'))
                startendtimelist.append((times[1]+lentail+1, newsimsection, 'remove'))

    #sort timelist
    startendtimelist.sort(key = lambda x: x[0]) #sort according to times
    ax = plt.gca()
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymin - 5, ymax + 5)
    seclist = []
    times = [startendtimelist[0][0], startendtimelist[-1][0]]
    frames = list(range(times[0], times[1]+1))
    usetimelist = None

    def init():
        nonlocal usetimelist
        nonlocal seclist
        artists = []
        for sec in seclist:
            sec['traj'].remove()
            sec['label'].remove()
            artists.append(sec['traj'])
            artists.append(sec['label'])
        seclist = []
        usetimelist = startendtimelist.copy()
        return artists
    def anifunc(frame):
        nonlocal seclist
        nonlocal usetimelist
        artists = []
        #add or remove vehicles as needed
        while len(usetimelist) > 0:
            nexttime = usetimelist[0][0]
            if nexttime == frame:
                time, sec, task = usetimelist.pop(0)
                if task == 'add':
                    #create artists and keep reference to it in the dictionary - keep dictionary in seclist - all active trajectories
                    traj = ax.plot([xmin,xmax],[ymin,ymax], **sec['kwargs'])[0]
                    label = ax.annotate(sec['veh'], (xmin,ymin), fontsize = 7)
                    sec['traj']  = traj
                    sec['label'] = label
                    seclist.append(sec)
                elif task == 'remove':
                    #remove artists
                    seclist.remove(sec)
                    sec['traj'].remove()
                    sec['label'].remove()

                    artists.append(sec['traj'])
                    artists.append(sec['label'])
            else:
                break

        for sec in seclist:
            #do updating here
            animatevhdhelper(sec, frame, lentail)
            artists.append(sec['traj'])
            artists.append(sec['label'])


        return artists

    ani = animation.FuncAnimation(fig, anifunc, init_func = init, frames = frames, blit = True, interval = interval, repeat=True)

    return ani


def animatevhdhelper(sec, time, lentail):
    starttime = sec['start']
    endtime = sec['end']
    if time > endtime:
        end = endtime
    else:
        end = time

    if time < starttime + lentail+1:
        start = starttime
    else:
        start = time - lentail

    sec['traj'].set_data(sec['hd'][start - starttime: end - starttime+1], sec['spd'][start - starttime:end - starttime+1])
    sec['label'].set_position((sec['hd'][end - starttime], sec['spd'][end - starttime]))
    return


def find_current_leader(current_frame, leadinfo):
    # leadinfo is already only about one vehicle id
    # leadinfo is of the form [[leader, start_frame, end_frame], [new_leader, end_frame+1, another_end_frame]]
    leader_id = leadinfo[0][0]
    for k in range(len(leadinfo)):
        if leadinfo[k][1] <= current_frame and current_frame <= leadinfo[k][2]:
            leader_id = leadinfo[k][0]
    # After validation of start and end frame, this function is guaranteed to return a valid result
    return leader_id

def compute_validate_time(timerange, t_n, T_nm1, h=.1, delay=0):
    # start time validation
    # If passed in as None, or any value outside [t_n, T_nm1], defaults to t_n
    if timerange[0] is None or timerange[0] < t_n or timerange[0] >= T_nm1:
        start = t_n
        if delay != 0:
            offset = math.ceil(delay / h)
            start = t_n + offset
    else:
        start = timerange[0]

    # end time validation
    # If passed in as None, or any value outside [t_n, T_nm1], or smaller than timerange[0], default to T_nm1
    if timerange[1] is None or timerange[1] < timerange[0] or timerange[1] > T_nm1:
        end = T_nm1
    else:
        end = timerange[1]
    return start, end

def compute_headway(t_nstar, t_n, T_n, datalen, leadinfo, start, dataset, veh_id, relax):
    lead = np.zeros((T_n + 1 - t_n, datalen))  # initialize the lead vehicle trajectory
    for j in leadinfo[0]:
        curleadid = j[0]  # current leader ID
        leadt_nstar = int(dataset[curleadid][0, 1])  # t_nstar for the current lead, put into int
        lead[j[1] - t_n:j[2] + 1 - t_n, :] = dataset[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,:]  # get the lead trajectory from simulation
    headway = lead[start - t_n:, 2] - dataset[veh_id][start - t_nstar:, 2] - lead[start - t_n:, 6] + relax[start - t_n:]
    return headway

def compute_headway2(veh, data, platooninfo, rp = None, h =.1):
    #compute headways from data and platooninfo, possibly adding relaxation if desired
    #different format than compute_headway

    relaxtype = 'both' if rp is not None else 'none'
    leadinfo, unused, rinfo = helper.makeleadfolinfo([veh], platooninfo, data, relaxtype = relaxtype)
    t_nstar, t_n, T_nm1, T_n = platooninfo[veh][:4]
    relax, unused = helper.r_constant(rinfo[0], [t_n, T_nm1], T_n, rp, False, h)

    lead = np.zeros((T_nm1 + 1 - t_n, 9))  # initialize the lead vehicle trajectory
    for j in leadinfo[0]:
        curleadid = j[0]  # current leader ID
        leadt_nstar = int(data[curleadid][0, 1])  # t_nstar for the current lead, put into int
        lead[j[1] - t_n:j[2] + 1 - t_n, :] = data[curleadid][j[1] - leadt_nstar:j[2] + 1 - leadt_nstar,:]  # get the lead trajectory from simulation
    headway = lead[:, 2] - data[veh][t_n - t_nstar:T_nm1 - t_nstar+1, 2] - lead[:, 6] + relax[:T_nm1+1-t_n]

    return headway


def compute_line_data(headway, i, lentail, dataset, veh_id, time):
    trajectory = (headway[i:i + lentail], dataset[veh_id][time + i:time + i + lentail, 3])
    label = (headway[i + lentail], dataset[veh_id][time + i + lentail, 3])

    # Compute x_min, y_min, x_max and y_max for the given data and return
    if lentail == 0:
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
    else:
        x_min = min(headway[i:i + lentail])
        x_max = max(headway[i:i + lentail])
        y_min = min(dataset[veh_id][time + i:time + i + lentail, 3])
        y_max = max(dataset[veh_id][time + i:time + i + lentail, 3])

    return trajectory, label, x_min, y_min, x_max, y_max


def animatetraj(meas, followerchain, platoon=[], usetime=[], presim=True, postsim=True, datalen=9, speed_limit = [],
                   show_ID = True, interval = 10):
    #plots vehicles platoon using data meas.

    # platoon = [] - if given as a platoon, only plots those vehicles in the platoon (e.g. [[],1,2,3] )
    # usetime = [] - if given as a list, only plots those times in the list (e.g. list(range(1,100)) )
    # presim = True - presim and postsim control whether the entire trajectory is displayed or just the simulated parts (t_nstar - T_n versus T-n - T_nm1)
    # postsim = True

    if platoon != []:
        followerchain = helper.platoononly(followerchain, platoon)
    platoontraj, mytime = helper.arraytraj(meas, followerchain, presim, postsim, datalen)
    if not usetime:
        usetime = mytime

    fig = plt.figure(figsize=(10, 4))  # initialize figure and axis
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1600), ax.set_xlabel('localY')
    ax.set_ylim(7.5, 0), ax.set_ylabel('laneID')

    scatter_pts = ax.scatter([], [], c=[], cmap=palettable.colorbrewer.diverging.RdYlGn_4.mpl_colormap, marker=">") #cm.get_cmap('RdYlBu')

    if speed_limit == []:
        maxspeed = 0
        minspeed = math.inf
        for i in followerchain.keys():
            curmax = max(meas[i][:,3])
            curmin = min(meas[i][:,3])
            if curmin < minspeed:
                minspeed = curmin
            if curmax > maxspeed:
                maxspeed = curmax
        norm = plt.Normalize(minspeed,maxspeed)
    else:
        norm = plt.Normalize(speed_limit[0], speed_limit[1])

    fig.colorbar(scatter_pts, cmap=cm.get_cmap('RdYlBu'), norm=norm, shrink=0.7)
    current_annotation_dict = {}

    def aniFunc(frame):
        artists = [scatter_pts]
        ax = plt.gca()
        curdata = platoontraj[usetime[frame]]
        X = curdata[:, 2]
        Y = curdata[:, 7]
        speeds = curdata[:, 3]
        ids = curdata[:, 0]
        existing_vids = list(current_annotation_dict.keys()).copy()

        # Go through ids list
        # If the annotation already exists, modify it via set_position
        # If the annotation doesn't exist before, introduce it via ax.annotate
        if show_ID:
            for i in range(len(ids)):
                vid = ids[i]
                if vid in current_annotation_dict.keys():
                    current_annotation_dict[vid].set_position((X[i], Y[i]))
                    existing_vids.remove(vid)
                else:
                    current_annotation_dict[vid] = ax.annotate(str(int(vid)), (X[i], Y[i]), fontsize=7)
                artists.append(current_annotation_dict[vid])

            # Afterwards, check if existing annotations need to be removed, process it accordingly
            if len(existing_vids) > 0:
                for vid in existing_vids:
                    artists.append(current_annotation_dict[vid])
                    current_annotation_dict[vid].remove()
                    del current_annotation_dict[vid]


        c = speeds
        pts = [[X[i], Y[i]] for i in range(len(X))]
        data = np.vstack(pts)
        scatter_pts.set_offsets(data)
        scatter_pts.set_array(c)
        return artists

    def init():
        artists = [scatter_pts]
        ax = plt.gca()
        if show_ID:
            for vid, annotation in list(current_annotation_dict.items()).copy():
                artists.append(annotation)
                annotation.remove()
                del current_annotation_dict[vid]
        curdata = platoontraj[usetime[0]]
        X = curdata[:, 2]
        Y = curdata[:, 7]
        speeds = curdata[:, 3]
        ids = curdata[:, 0]
        for i in range(len(ids)):
            current_annotation_dict[ids[i]] = ax.annotate(str(int(ids[i])), (X[i], Y[i]), fontsize=7)
            artists.append(current_annotation_dict[ids[i]])
        c = speeds
        pts = [[X[i], Y[i]] for i in range(len(X))]
        data = np.vstack(pts)
        scatter_pts.set(norm=norm)
        scatter_pts.set_offsets(data)
        scatter_pts.set_array(c)
        return artists

    out = animation.FuncAnimation(fig, aniFunc, init_func=init, frames=len(usetime), interval=interval, blit = True)

    return out


def wt(series, scale):
    out, out2 = pywt.cwt(series, scale, 'mexh')
    energy = np.sum(np.abs(out), 0)
    return energy


def plotspacetime(meas, platooninfo, timeint = 50, xint = 70, lane=1, use_avg='mean'):
    # meas - keys are vehicles, values are numpy arrays where rows are observations
    #platooninfo - created with meas
    # timeint - length of time in each aggregated speed (in terms of data units);
    # xint - length of space in each aggregated speed (in terms of data units)
    # use_avg = 'mean' - controls averaging for speeds. if 'mean' then does arithmetic mean. if 'harm' then harmonic mean.
    # lane = 1 - choose which lane of the data to plot.

    #aggregates data in meas and plots it in spacetime plot

    #get data with helper function
    X, Y, meanspeeds, vehbins = plotspacetime_helper(meas, timeint, xint, lane, use_avg)

    #plotting
    cmap = cm.RdYlBu  # RdYlBu is probably the best colormap overall for this
    cmap.set_bad('white', 1.)  # change np.nan into white color
    fig, current_ax = plt.subplots(figsize=(12, 8))
    plt.pcolormesh(X, Y, meanspeeds,
                   cmap=cmap)  # pcolormesh is similar to imshow but is meant for plotting whereas imshow is for actual images
    plt.xlabel('Time')
    plt.ylabel('Space')
    cbar = plt.colorbar()  # colorbar
    cbar.set_label('Speed')

def plotspacetime_helper(myinput, timeint, xint, lane, avg_type, return_discretization = False, return_vehlist = False):
    #myinput - data, in either raw form (numpy array) or dictionary
    # timeint - length of time in each aggregated speed (in terms of data units);
    # xint - length of space in each aggregated speed (in terms of data units)
    #lane - if not None, selects only observations in lane
    #avg_type - can be either 'mean' to use arithmetic mean, or 'harm' to use harmonic mean
    #return_discretization - boolean controls whether to add discretization of space, time (both are 1d np arrays) to output
    #return_vehlist - boolean controls whether to return a set containing all unique vehicle IDs for observations

    #returns -
    #X - np array where [i,j] index gives X (time) coordinate for times[i], space[j] (note we call space x)
    #Y - np array where [i,j] index gives Y (space) coordinate for times[i], space[j]
    #meanspeeds - np array giving average speed in subregion, indexed the same as X and Y
    #vehbins - np array gives set of vehicle IDs for subregion, indexed the same as X and Y
    #(optional) - x, 1d np array giving grid points for x
    #(optional) - times, 1d np array giving grid points for time
    #(optional) - vehlist, set containing all unique vehicle IDs for observations
    if type(myinput) == dict: #assume either dict or raw input
        data = np.concatenate(list(myinput.values()))
    else:
        data = myinput
    if lane != None:
        data = data[data[:, 7] == lane]  #data from lane
        #if you want to plot multiple lanes, can mask data before and pass lane = None

    t0 = min(data[:, 1])
    tend = max(data[:, 1]) + 1e-6
    x0 = min(data[:, 2])
    xend = max(data[:, 2]) + 1e-6
    #discretization
    times = np.arange(t0, tend, timeint)
    if times[-1] != tend:
        times = np.append(times, tend)
    x = np.arange(x0, xend, xint)
    if x[-1] != xend:
        x = np.append(x, xend)
    X, Y = np.meshgrid(times, x, indexing = 'ij')

    #type of average
    if avg_type == 'mean':
        meanfunc = np.mean
    elif avg_type == 'harm':
        meanfunc = harmonic_mean

    #speeds and veh are nested lists indexed by (time, space)
    #speeds are lists of speeds, veh are sets of vehicle IDs
    speeds = [[[] for j in range(len(x)-1)] for i in range(len(times)-1)]
    vehbins = [[set() for j in range(len(x)-1)] for i in range(len(times)-1)]
    for i in range(len(data)):  # put all observations into their bin
        curt, curx, curv, curveh = data[i,[1,2,3,0]]

        curtimebin = math.floor((curt - t0) / timeint)
        curxbin = math.floor((curx - x0) / xint)
        speeds[curtimebin][curxbin].append(curv)
        vehbins[curtimebin][curxbin].add(curveh)

    meanspeeds = X.copy()  # initialize output
    for i in range(len(times) - 1):  # populate output
        for j in range(len(x) - 1):
            cur = speeds[i][j]
            if len(cur) == 0:
                cur = np.nan
            else:
                cur = meanfunc(cur)
            meanspeeds[i,j] = cur

    out = (X, Y, meanspeeds, vehbins)
    if return_discretization:
        out = out + (x, times)
    if return_vehlist:
        vehlist = set(np.unique(data[:,0]))
        out = out + (vehlist, )
    return out

def selectoscillation(meas, platooninfo, timeint = 50, xint = 70, lane=1, use_avg='mean', region_shape='p'):
    # meas - keys are vehicles, values are numpy arrays where rows are observations
    #platooninfo - created with meas
    # timeint - length of time in each aggregated speed (in terms of data units);
    # xint - length of space in each aggregated speed (in terms of data units)
    # use_avg = 'mean' - controls averaging for speeds. if 'mean' then does arithmetic mean. if 'harm' then harmonic mean.
    # lane = 1 - choose which lane of the data to plot.
    # region_shape = 'p' - 'p' makes region shapes into parralelograms. no other options are available

    #returns - nothing (creates an interactive plot)

    #makes an interactive spacetime plot - you can add trajectories to the plot,
    #select regions of the data by specifying regions in space-time.
    #Then you can send the regions to selectvehID to examine the data in more detail

    #get data with helper function
    X, Y, meanspeeds, vehbins, x, times, vehlist = \
    plotspacetime_helper(meas, timeint, xint, lane, use_avg,
                         return_discretization = True, return_vehlist = True)
    x0, xend, t0, xint, = x[0], x[-1], times[0], x[1] - x[0]

    #plotting
    cmap = cm.RdYlBu  # RdYlBu is probably the best colormap overall for this
    cmap.set_bad('white', 1.)  # change np.nan into white color
    fig, current_ax = plt.subplots(figsize=(12, 8))
    plt.pcolormesh(X, Y, meanspeeds,
                   cmap=cmap)  # pcolormesh is similar to imshow but is meant for plotting whereas imshow is for actual images
    plt.xlabel('Time')
    plt.ylabel('Space')
    cbar = plt.colorbar()  # colorbar
    cbar.set_label('Speed')

    def my_callback(
            args):  # this can be used to modify the current shape (for example, to make it into a perfect parralelogram, or to round the points to the nearest bin, etc.)
        # this will make the shape into a parralelogram with horizontal top and bottom. the first two points define the left side length, and the third point defines
        # the length of the other side. So the fourth point is therefore unused.
        nonlocal vertlist

        if region_shape != 'p':  # the callback function only does something when a parralelogram shape ('p') is requested
            return

            # note that you have to do a 4 sided shape
        testx = mytoggle_selector.RS._xs  # when shape is completed we get the coordinates of its vertices
        testy = mytoggle_selector.RS._ys

        # get x and y displacements of the first line
        lenx = testx[0] - testx[1]
        leny = testy[0] - testy[1]

        # set third point
        testy[2] = testy[1]
        # set last point to complete the parralogram
        testy[3] = testy[2] + leny
        testx[3] = testx[2] + lenx
        # redraw the shape
        mytoggle_selector.RS._draw_polygon()
        vertlist[-1] = mytoggle_selector.RS.verts

        return

    def mytoggle_selector(event):
        # these keep track of which vehicles we are showing in the plot
        nonlocal vehhelper
        nonlocal plottedvehlist
        nonlocal lineobjects
        # keep track of the areas of oscillation identified
        nonlocal vertlist
        if event.key in ['A', 'a']:  #plot all vehicles in bin
            curx, cury = event.xdata, event.ydata  # get mouse position
            # translate the mouse position into the bin we want to use
            curtimebin = math.floor((curx - t0) / timeint)
            curxbin = math.floor((cury - x0) / xint)

            newvehs = list(vehhelper[curtimebin][curxbin])
            if len(newvehs) > 0:
                vehhelper[curtimebin][curxbin] = set()
                for vehid in newvehs:
                    if vehid in plottedvehlist: #vehicle already plotted
                        continue
                    #plot vehicle and add to plottedvehlist
                    plottedvehlist.add(vehid)
                    temp = meas[vehid]
                    indjumps = helper.sequential(temp)
                    for i in range(len(indjumps) - 1):
                        plotx, ploty = temp[indjumps[i]:indjumps[i + 1], 1], temp[indjumps[i]:indjumps[i + 1], 2]
                        newline = plt.plot(plotx, ploty, 'C0', scalex=False, scaley=False)
                        lineobjects.append(newline)
                    plt.draw()

        if event.key in ['D', 'd']:  # on this key press we remove all the trajectories
            # reset all the plotting stuff
            vehhelper = copy.deepcopy(
                vehbins)  # vehhelper is going to keep track of what vehicles in each bin have not yet been plotted
            plottedvehlist = set()  # list of vehicle trajectories weve plotted
            for i in range(len(lineobjects)):
                lineobjects[i][0].remove()  # remove all lines
            plt.draw()
            lineobjects = []

        if event.key in ['W', 'w']:  # put top side of the shape to the top
            testx = mytoggle_selector.RS._xs  # when shape is completed we get the coordinates of its vertices
            testy = mytoggle_selector.RS._ys

            lenx = testx[0] - testx[1]
            leny = testy[0] - testy[1]

            dy = xend - testy[0]
            dx = dy * lenx / leny
            # update top side
            testx[0] += dx
            testy[0] += dy
            testx[-2] += dx
            testy[-2] += dy
            testx[-1] = testx[0]
            testy[-1] = testy[0]

            mytoggle_selector.RS._draw_polygon()

        if event.key in ['X', 'x']:  # put bottom side of the shape to the bottom
            testx = mytoggle_selector.RS._xs  # when shape is completed we get the coordinates of its vertices
            testy = mytoggle_selector.RS._ys

            lenx = testx[0] - testx[1]
            leny = testy[0] - testy[1]

            dy = -x0 + testy[1]
            dx = dy * lenx / leny
            # update bottom side
            testx[1] += -dx
            testy[1] += -dy
            testx[2] += -dx
            testy[2] += -dy

            mytoggle_selector.RS._draw_polygon()

        if event.key == 'enter':  # enter means we are happy with the current region selected and want to choose another.
            if len(mytoggle_selector.RS.verts) == 4: #if shape is valid
                vertlist.append(None) #entry where new shape vertices will go

            mytoggle_selector.RS = PolygonSelector(current_ax, my_callback,
                                                   lineprops=dict(color='k', linestyle='-', linewidth=2, alpha=0.4),
                                                   markerprops=dict(marker='o', markersize=2, mec='k', mfc='k',
                                                                    alpha=0.4))
            plt.connect('key_press_event', mytoggle_selector)
            plt.show()

        if event.key in ['N', 'n']:
            #sort vehicles to form all vehicles which can be shown
            all_veh_list = hc.sortveh(lane, meas, vehlist)

            #form platoonlists for start,end vertices for all regions in vertlist
            platoonlist = list(range(len(vertlist)*2))
            for i in range(len(platoonlist)):
                regionind, vertind = i // 2, (i % 2)*2
                time, space = vertlist[regionind][vertind]
                curtimebin = math.floor((time - t0) / timeint)
                curxbin = math.floor((space - x0) / xint)
                curplatoon = list(vehbins[curtimebin][curxbin])
                if len(curplatoon) == 0:
                    curtimebin = [curtimebin]
                    curxbin = [curxbin]
                    while True:
                        #expand bins that we look for vehicles in
                        curtimebin.append(curtimebin[-1]+1)
                        curtimebin.add(0, curtimebin[0]-1)
                        curxbin.append(curxbin[-1]+1)
                        curxbin.add(0, curxbin[0]-1)
                        for j in curtimebin:
                            for k in curxbin:
                                try:
                                    curplatoon = list(vehbins[j][k])
                                except: #out of bounds
                                    continue
                                if len(curplatoon) >0:
                                    break
                platoonlist[i] = curplatoon

            selectvehID(meas, platooninfo, lane, all_veh_list, vertlist, platoonlist, ind = 0)

        if event.key in ['V', 'v']:
            print(vertlist)


    print('a to plot vehicle trajectories in area, d to clear all vehicle trajectories')
    print('click with mouse to start drawing polygon')
    print('enter to select a new region to identify, click + shift to drag, esc to start over with current region')
    print('w to snap shape to top, x to snap shape to bottom, v to print vertices of all completed shapes')
    print('when all regions of interest are identified, press n to move to next stage of the process')

    vehhelper = copy.deepcopy(
        vehbins)  # vehhelper is going to keep track of what vehicles in each bin have not yet been plotted
    plottedvehlist = set()  # list of vehicle trajectories weve plotted
    lineobjects = []  # list of line objects weve plotted (note that we are including the option to remove all the things )

    vertlist = [None]  # list of the vertex corners

    mytoggle_selector.RS = PolygonSelector(current_ax, my_callback,
                                           lineprops=dict(color='k', linestyle='-', linewidth=2, alpha=0.4),
                                           markerprops=dict(marker='o', markersize=2, mec='k', mfc='k', alpha=0.4))
    plt.connect('key_press_event', mytoggle_selector)
    plt.show()

    return


def selectvehID(meas, platooninfo, lane, all_veh_list, vertlist = None, platoonlist=None, ind=0, out = [[]]):
    # TODO - refactor this into a more general version where you can pass your plot_ax functions so you can
    # have an interactive plot for whatever you want. Put the current functionality into its own function

    # meas - data in dictionary format
    # platooninfo - for meas
    # lane - lane for data shown. Must be specified
    # all_veh_list - sorted list of all possible vehicles which can be shown
    # vertlist - list of verticies of region, each vertice is a tuple and there should usually be 4 corners. If None, there will be no region shown
    # platoonlist = None - Specify either
    #    list of platoons - ith platoon are the initial vehicles shown for the ith call of selectvehID
    #    platoon - initial vehicles shown
    #    None - default to showing first vehicle in all_veh_list
    # ind = 0 - index keeps count of what platoon we are on
    # out - if platoonlist is a nested list of platoons, selectvehID can be called multiple times,
    # out keeps track of the selected vehicles for each call

    # outputs - None, creates interactive plot
    # plot with 4 subplots, shows the space-time, speed-time, std. dev of speed, wavelet series of vehicles.
    # interactive plot can add vehicles before/after, select specific vehicles etc.


    #initialize vehicles shown
    if platoonlist != None:
        if type(platoonlist[0]) == list: #nested list of platoons
            try:
                platoon = platoonlist[ind]
            except:
                print('selected vehicles are '+str(out)) #no more platoonlists -> return list of selected vehicles
                return
        else:
            platoon = platoonlist #platoonlist is a platoon
    else:
        platoon = None #default to None
    if platoon != None:
        # If a platoon is passed in, and the size of input platoon is greater than 1
        # Go through all platoons and find out the earliest and the latest vehicle
        # All vehicles in between will be placed as initial vehicles

        if len(platoon) == 1:
            veh_list = platoon
            left_window_index = 0
            right_window_index = 0
        else:
            left_window_index = len(all_veh_list) + 1
            right_window_index = -1
            # Loop through all vehicles in platoon and find out the initial range
            for vehicle in platoon:
                if vehicle in all_veh_list:
                    left_window_index = min(left_window_index, all_veh_list.index(vehicle))
                    right_window_index = max(right_window_index, all_veh_list.index(vehicle))
                    # If the passed in vehicle is not in the designated time range, it is omitted

            # Add all vehicles between left_window_index and right_window_index (inclusive) to veh_list
            veh_list = []
            for index in range(left_window_index, right_window_index + 1):
                veh_list.append(all_veh_list[index])
    else:
        # No platoon is passed in, use the first vehicle as initial by default
        initial_index = 0
        veh_list = []
        veh_list.append(all_veh_list[initial_index])
        left_window_index = initial_index
        right_window_index = initial_index

    #current vertices for shape
    if vertlist != None:
        curvert = vertlist[ind // 2]  # region selected
        xvert = [i[0] for i in curvert] + [curvert[0][0]]
        yvert = [i[1] for i in curvert] + [curvert[0][1]]
    else:
        xvert, yvert = [], []
    #keep track of selected vehicles if called multiple times
    if ind == 0: #reset out the first time we call
        out = [[]]
    if len(out[-1]) == 2:
        out.append([None])
    else:
        out[-1].append(None)

    # Initiate plotting
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    ax4.get_xaxis().set_visible(False)

    scale = np.arange(15, 100)  # for wavelet transform

    # The centralized dictionary that stores {veh_id -> list of all 4 artists in ax1, ax2, ax3 and ax4}
    artist_dict = {}
    # The reverse mapping of the dictionary above that stores {artist -> veh_id}
    artist_to_veh_dict = {}
    # The centralized dictionary that stores {veh_id -> list of all artists in ax2 and ax3}
    # This is a separate mapping since its plotting is separated from above
    ax23_artist_dict = {}
    # The vehicle indices currently getting plotted for both ax2 and ax3 (they always show the same vehicle)
    current_ax23_veh_left_index = -1
    current_ax23_veh_right_index = -1
    #controls visibility of opaque (out of lane) trajectories for each axis
    visbool = {ax1:True, ax2:True, ax3:True, ax4:True}


    # Base plotting function used for ploting all 4 plots
    def plot_axis(plot, y_axis, veh, meas, spdstd=None, vis = True):
        nonlocal artist_to_veh_dict

        LCind = generate_LCind(meas[veh], lane)
        x = meas[veh][:, 1]
        y2 = meas[veh][:, 3]
        axis_artist_list = []

        for i in range(len(LCind) - 1):
            kwargs = {'picker':5}
            show1 = True
            if meas[veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
                show1 = vis
            if spdstd is not None:
                spdstd = np.append(spdstd, y2[LCind[i]:LCind[i + 1]])
            axis_artist = plot.plot(x[LCind[i]:LCind[i + 1]], y_axis[LCind[i]:LCind[i + 1]], 'C0', **kwargs)
            axis_artist[0].set_visible(show1)
            axis_artist_list.append(axis_artist)
            for art in axis_artist:
                artist_to_veh_dict[art] = veh

        return axis_artist_list, spdstd

    def plot_ax1(veh, meas, platooninfo):
        nonlocal artist_dict

        y1 = meas[veh][:, 2]
        spdstd = np.asarray([])
        artist_dict[veh] = [None] * 4
        ax1_artist_list, spdstd = plot_axis(ax1, y1, veh, meas, spdstd, vis = visbool[ax1])

        artist_dict[veh][0] = ax1_artist_list
        spdstd = np.std(spdstd)

        return spdstd

    def plot_ax2(veh, meas, platooninfo):
        nonlocal artist_dict
        nonlocal ax23_artist_dict
        nonlocal current_ax23_veh_left_index
        nonlocal current_ax23_veh_right_index
        ax2.cla()

        y2 = meas[veh][:, 3]
        veh_index = all_veh_list.index(veh)
        current_ax23_veh_left_index = veh_index
        current_ax23_veh_right_index = veh_index
        ax23_artist_dict[veh] = [None] * 2
        ax2_artist_list, temp = plot_axis(ax2, y2, veh, meas, vis = visbool[ax2])

        artist_dict[veh][1] = ax2_artist_list
        ax23_artist_dict[veh][0] = ax2_artist_list
        return

    def plot_ax3(veh, meas, platooninfo):
        nonlocal artist_dict
        nonlocal ax23_artist_dict
        nonlocal current_ax23_veh_left_index
        nonlocal current_ax23_veh_right_index
        ax3.cla()

        y2 = meas[veh][:, 3]
        energy = wt(y2, scale)
        veh_index = all_veh_list.index(veh)
        current_ax23_veh_left_index = veh_index
        current_ax23_veh_right_index = veh_index
        ax3_artist_list, temp = plot_axis(ax3, energy, veh, meas, vis = visbool[ax3])

        artist_dict[veh][2] = ax3_artist_list
        ax23_artist_dict[veh][1] = ax3_artist_list
        return

    def plot_ax4(veh, meas, platooninfo, spdstd):
        nonlocal artist_dict
        nonlocal artist_to_veh_dict

        x,y = all_veh_list.index(veh), spdstd
        ax4_artist = ax4.plot(x, y, 'C0.', picker=5)
        # ax4_annotate = ax4.annotate(str(veh), (x+.1, y), fontsize = 'small')
        # ax4_artist.append(ax4_annotate)
        artist_dict[veh][3] = ax4_artist
        for art in ax4_artist:
            artist_to_veh_dict[art] = veh

        return

    # Plot the 4 subplots
    for veh_index, veh in enumerate(veh_list):
        spdstd = plot_ax1(veh, meas, platooninfo)
        plot_ax2(veh, meas, platooninfo)
        plot_ax3(veh, meas, platooninfo)
        plot_ax4(veh, meas, platooninfo, spdstd)
        selected_veh = veh

    plt.suptitle('Left click on trajectory to select vehicle\nPress \'H\' to view key press options')

    boxartist = ax1.plot(xvert, yvert, 'k-', scalex=False, scaley=False, alpha=.4)  # draw box for trajectories

    def on_pick(event):
        nonlocal artist_dict
        nonlocal artist_to_veh_dict
        nonlocal selected_veh

        selected_veh = artist_to_veh_dict[event.artist]

        if event.mouseevent.button == 3:  # right click has no functionality currently
            pass
        else:
            # left click selects the trajectory to begin/end with
            out[-1][-1] = selected_veh
            # Reset color for all lines in ax1 and ax4
            for line in ax1.lines:
                line.set_color('C0')
            for line in ax4.lines:
                line.set_color('C0')
            boxartist[0].set_color('k')

            # Set color for selected vehicle
            ax1_artist_list = artist_dict[selected_veh][0]
            for ax1_artist in ax1_artist_list:
                for art in ax1_artist:
                    art.set_color('C1')

            ax4_artist_list = artist_dict[selected_veh][3]
            for ax4_artist in ax4_artist_list:
                ax4_artist.set_color('C1')

            plt.suptitle('Vehicle ID ' + str(selected_veh) + ' selected')
            plot_ax2(selected_veh, meas, platooninfo)
            plot_ax3(selected_veh, meas, platooninfo)

            fig.canvas.draw()
            return

    # Utility functions respond to various key press events
    def add_vehicle_ax14_key_press_response(new_veh):
        nonlocal selected_veh
        spdstd = plot_ax1(new_veh, meas, platooninfo)
        plot_ax2(new_veh, meas, platooninfo)
        plot_ax3(new_veh, meas, platooninfo)
        plot_ax4(new_veh, meas, platooninfo, spdstd)
        scale_plot(ax1, y_axis_padding=50, dont_use = boxartist[0])
        scale_plot(ax4, 0.5, 0.25)
        plt.draw()
        selected_veh = new_veh

    def remove_vehicle_ax14_key_press_response(veh_tbr):
        ax1_artist_list = artist_dict[veh_tbr][0]
        for ax1_artist in ax1_artist_list:
            for art in ax1_artist:
                art.remove()
                del art
        ax4_artist_list = artist_dict[veh_tbr][3]
        for ax4_artist in ax4_artist_list:
            ax4_artist.remove()
            del ax4_artist
        # scale_plot(ax1, y_axis_padding=50, dont_use = boxartist[0])
        # scale_plot(ax4, 0.5, 0.25)
        plt.draw()

    def add_vehicle_ax23_key_press_response(new_veh, curindex):
        LCind = generate_LCind(meas[new_veh], lane)
        x = meas[new_veh][:, 1]
        y2 = meas[new_veh][:, 3]
        energy = wt(y2, scale)

        ax2_artist_list = []
        ax3_artist_list = []

        ax2shift = 10*-(curindex - all_veh_list.index(selected_veh))
        ax3shift = 1000*-(curindex - all_veh_list.index(selected_veh))

        curvis2 = visbool[ax2]
        curvis3 = visbool[ax3]

        for i in range(len(LCind) - 1): #why is this plotting not use plot_ax3??
            kwargs = {}
            show2 = True
            show3 = True
            if meas[new_veh][LCind[i], 7] != lane:
                kwargs = {'linestyle': '--', 'alpha': .4}
                show2 = curvis2
                show3 = curvis3
            ax2_artist = ax2.plot(x[LCind[i]:LCind[i + 1]], y2[LCind[i]:LCind[i + 1]]+ax2shift, 'C0', picker=5, **kwargs)
            ax2_artist[0].set_visible(show2)
            ax2_artist_list.append(ax2_artist)
            ax3_artist = ax3.plot(x[LCind[i]:LCind[i + 1]], energy[LCind[i]:LCind[i + 1]]+ax3shift, 'C0', picker=5, **kwargs)
            ax3_artist[0].set_visible(show3)
            ax3_artist_list.append(ax3_artist)
        ax23_artist_dict[new_veh] = [None] * 2
        ax23_artist_dict[new_veh][0] = ax2_artist_list
        ax23_artist_dict[new_veh][1] = ax3_artist_list
        scale_plot(ax2)
        scale_plot(ax3, y_axis_padding=500)
        plt.draw()

    def remove_vehicle_ax23_key_press_response(veh_tbr):
        ax2_artist_list = ax23_artist_dict[veh_tbr][0]
        for ax2_artist in ax2_artist_list:
            for art in ax2_artist:
                art.remove()
                del art
        ax3_artist_list = ax23_artist_dict[veh_tbr][1]
        for ax3_artist in ax3_artist_list:
            for art in ax3_artist:
                art.remove()
                del art

        scale_plot(ax2)
        scale_plot(ax3, y_axis_padding=500)
        plt.draw()

    def key_press(event):
        nonlocal artist_dict
        nonlocal veh_list
        nonlocal left_window_index
        nonlocal right_window_index
        nonlocal ax23_artist_dict
        nonlocal current_ax23_veh_left_index
        nonlocal current_ax23_veh_right_index
        nonlocal visbool

        if event.key in ['H', 'h']:
            # print("Key press instructions (case insensitive):")
            print("Press T to add/remove out of lane trajectories;")
            print("Press V to print the list of current vehicles;")
            print("Press A to add the previous vehicle into plot 1&4;")
            print("Press D to add the next vehicle into plot 1&4;")
            print("Press C to remove the first vehicle from plot 1&4;")
            print("Press Z to remove the last vehicle from plot 1&4;\n")

            print("Press W to add the previous vehicle into plot 2&3;")
            print("Press E to add the next vehicle into plot 2&3;")
            print("Press U to remove the first vehicle from plot 2&3;")
            print("Press Y to remove the last vehicle from plot 2&3;")
            print("Press N to switch to the next study area.")

        if event.key in ['T', 't']:  # toggles all opaque lines
            ax = event.inaxes
            visibility = not visbool[ax]
            visbool[ax] = visibility
            for i in ax.lines:
                if i.get_alpha() != None and i is not boxartist[0]:
                    i.set_visible(visibility)
            fig.canvas.draw()

        if event.key in ['V', 'v']:  # print current vehicle list
            print('current vehicles shown are ' + str(veh_list))
        if event.key in ['A', 'a']:  # add a vehicle before in ax1 and ax4
            if left_window_index > 0:
                left_window_index += -1
                new_veh = all_veh_list[left_window_index]
                veh_list.insert(0, new_veh)

                add_vehicle_ax14_key_press_response(new_veh)

        if event.key in ['D', 'd']:  # add a vehicle after in ax1 and ax4
            if right_window_index < len(all_veh_list) - 1:
                right_window_index += 1
                new_veh = all_veh_list[right_window_index]
                veh_list.append(new_veh)

                add_vehicle_ax14_key_press_response(new_veh)

        if event.key in ['C', 'c']:  # remove a vehicle before in ax1 and ax4
            if right_window_index > left_window_index:
                veh_tbr = all_veh_list[left_window_index]
                left_window_index += 1
                veh_list.pop(0)

                remove_vehicle_ax14_key_press_response(veh_tbr)

        if event.key in ['Z', 'z']:  # remove a vehicle after in ax1 and ax4
            if right_window_index > left_window_index:
                veh_tbr = all_veh_list[right_window_index]
                right_window_index += -1
                veh_list.pop()

                remove_vehicle_ax14_key_press_response(veh_tbr)

        if event.key in ['W', 'w']:  # Add a vehicle before in ax2 and ax3
            if current_ax23_veh_left_index > 0:
                new_veh = all_veh_list[current_ax23_veh_left_index - 1]
                current_ax23_veh_left_index += -1

                add_vehicle_ax23_key_press_response(new_veh, current_ax23_veh_left_index)

        if event.key in ['E', 'e']:  # Add a vehicle after in ax2 and ax3
            if current_ax23_veh_right_index < len(all_veh_list) - 1:
                new_veh = all_veh_list[current_ax23_veh_right_index + 1]
                current_ax23_veh_right_index += 1

                add_vehicle_ax23_key_press_response(new_veh, current_ax23_veh_right_index)

        if event.key in ['U', 'u']:  # Remove a vehicle before in ax2 and ax3
            if current_ax23_veh_left_index <= current_ax23_veh_right_index:
                veh_tbr = all_veh_list[current_ax23_veh_left_index]
                current_ax23_veh_left_index += 1

                remove_vehicle_ax23_key_press_response(veh_tbr)

        if event.key in ['Y', 'y']:  # Remove a vehicle after in ax2 and ax3
            if current_ax23_veh_left_index <= current_ax23_veh_right_index:
                veh_tbr = all_veh_list[current_ax23_veh_right_index]
                current_ax23_veh_right_index += -1

                remove_vehicle_ax23_key_press_response(veh_tbr)

        if event.key in ['N', 'n']:  # next study area
            plt.close()
            print('have selected vehicle '+str(out[-1]))
            selectvehID(meas, platooninfo, lane, all_veh_list, vertlist, platoonlist, ind=ind + 1, out = out)

        return

    fig.canvas.callbacks.connect('pick_event', on_pick)
    plt.connect('key_press_event', key_press)
    return

# Scale given plot by going through all xy_data and compute min and max respectively
def scale_plot(axis, x_axis_padding=20, y_axis_padding=5, dont_use = None):
    min_x_value = 1e10
    max_x_value = -1
    min_y_value = 1e10
    max_y_value = -1
    for line in axis.lines:
        if line is dont_use:
            continue
        if len(line.get_xdata()) > 0:
            min_x_value = min(min(line.get_xdata()), min_x_value)
            max_x_value = max(max(line.get_xdata()), max_x_value)
        if len(line.get_ydata()) > 0:
            min_y_value = min(min(line.get_ydata()), min_y_value)
            max_y_value = max(max(line.get_ydata()), max_y_value)
    if len(axis.lines) > 0:
        axis.set_xlim(min_x_value - x_axis_padding, max_x_value + x_axis_padding)
        axis.set_ylim(min_y_value - y_axis_padding, max_y_value + y_axis_padding)