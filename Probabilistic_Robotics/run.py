from Robot import *
from World import World
from Agent import Agent
from Map import *


def main():
    world = World()

    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    agent1 = Agent(0.2, 0.0)
    agent2 = Agent(0.2, 10.0/180*math.pi)

    robot1 = Robot(np.array([2, 3, math.pi/5*6]).T, sensor=Camera(m), agent=agent1)
    robot2 = Robot(np.array([1, 0, math.pi/5*6]).T, agent=agent2, sensor=Camera(m), color='red')
    robot3 = Robot(np.array([0, 0, 0]).T, color='blue')

    world.append(robot1)
    world.append(robot2)
    world.append(robot3)

    world.draw()

if __name__ == '__main__':
    main()