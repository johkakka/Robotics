from Robot import *
from World import World
from Agent import Agent
from Map import *
from MCL import *
from Kalman import *


def main():
    time_interval = 0.1  ###draw_mcl7###
    world = World(60, time_interval, debug=False)

    m = Map()
    for ln in [(-4, 2), (2, -3), (3, 3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    initial_pose = np.array([0, 0, 0]).T
    # estimator = MCL(m, initial_pose, 100)
    estimator = KalmanFilter(m, initial_pose)
    circling = EstimationAgent(time_interval, 0.2, 10.0 / 180 * math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=circling, color="red")
    world.append(r)

    world.draw()


if __name__ == '__main__':
    main()