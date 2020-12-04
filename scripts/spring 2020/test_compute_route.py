"""
Test that the compute_route() function in the havsim.simulation.road package works correctly.
"""
from havsim.simulation.road import Road
from havsim.simulation.road import compute_route


def test1():
    road1 = Road(num_lanes=2, length=4, name='road1')
    road2 = Road(num_lanes=1, length=4, name='road2')
    road3 = Road(num_lanes=1, length=3, name='road3')
    road4 = Road(num_lanes=1, length=2, name='road4')
    road5 = Road(num_lanes=1, length=24, name='road5')

    road1.connect(road2, [1], [0])
    road1.connect(road5, [0], [0])
    road2.connect(road3)
    road3.connect(road4)
    road4.merge(road5, 0, 0, (1, 2), (23, 24))
    road5.connect('exit', is_exit=True)
    assert compute_route(road1, 0, 'exit') == ['road2', 'road3', 'road4', 'road5', 'exit']


def test2():
    road1 = Road(num_lanes=2, length=4, name='road1')
    road2 = Road(num_lanes=1, length=4, name='road2')
    road3 = Road(num_lanes=1, length=3, name='road3')
    road4 = Road(num_lanes=1, length=2, name='road4')
    road5 = Road(num_lanes=1, length=8, name='road5')

    road1.connect(road2, [1], [0])
    road1.connect(road5, [0], [0])
    road2.connect(road3)
    road3.connect(road4)
    road4.merge(road5, 0, 0, (1, 2), (7, 8))
    road5.connect('exit', is_exit=True)
    assert compute_route(road1, 0, 'exit') == ['road5', 'exit']


def test_all():
    test1()
    test2()

test_all()
