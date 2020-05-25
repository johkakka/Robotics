from Robot import *
from World import World
from Agent import Agent
from Map import *
from MCL import *


def main():
    world = World(30, 0.05)

    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    initial_pose = np.array([2, 2, math.pi/6]).T
    estimator = MCL(initial_pose, 100)
    agent = EstimationAgent(0.2, 10.0/180*math.pi, estimator)
    r = Robot(initial_pose, sensor=Camera(m), agent=agent)
    world.append(r)

    world.draw()

if __name__ == '__main__':
    main()