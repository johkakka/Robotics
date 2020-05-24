from Robot import *
from World import World
from Agent import Agent
from Map import *


def main():
    world = World(30, 0.05)

    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    agent1 = Agent(1, 0.0)
    agent2 = Agent(1, 30.0/180*math.pi)

    robot1 = Robot(np.array([2, 3, math.pi/5*6]).T, sensor=Camera(m), agent=agent1)
    robot2 = Robot(np.array([1, 0, math.pi/5*6]).T, agent=agent2, sensor=Camera(m), color='red')

    world.append(robot1)
    world.append(robot2)

    world.draw()

if __name__ == '__main__':
    main()