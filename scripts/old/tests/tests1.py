
"""
@author: rlk268@cornell.edu

test helper functions 
"""

from havsim.calibration.helper import makeleadfolinfo, boundaryspeeds, interpolate

path_reconngsim = 'C:/Users/rlk268/OneDrive - Cornell University/important misc/pickle files/meng/reconngsim.pkl' 

# reconstructed ngsim data
with open(path_reconngsim, 'rb') as f:
    data = pickle.load(f)[0]
meas, platooninfo, = makeplatoonlist(data,1,False)

platoon = [875.0, 903.0, 908.0]  # dataset only with these vehicles
toymeas = {}
for i in platoon:
    toymeas[i] = meas[i].copy()
toymeas[875][:, 4] = 0
toymeas[908][:, 5] = 0


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

def test_leadfolinfo():
    leadinfo, folinfo, rinfo = makeleadfolinfo(platoon,platooninfo,meas)
    assert leadinfo == [[[886.0, 2555, 2557], [889.0, 2558, 3059]],[[875.0, 2574, 3082]],[[903.0, 2621, 3104]]]
    assert folinfo == [[[903.0, 2574, 3082]], [[908.0, 2621, 3104]], []]
    assert rinfo == [[[2558, 6.591962153199944]], [], []]
    
test_leadfolinfo()


