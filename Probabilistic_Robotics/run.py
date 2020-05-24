import numpy as np
import math

from Robot import Robot
from World import World
from Agent import Agent
from Map import Map
from Map import Landmark

world = World()

m = Map()
m.append_landmark(Landmark(2, -2))
m.append_landmark(Landmark(-1, -3))
m.append_landmark(Landmark(3, 3))
world.append(m)

agent1 = Agent(0.2, 0.0)
agent2 = Agent(0.2, 10.0/180*math.pi)

robot1 = Robot(np.array([2, 3, math.pi/5*6]).T, agent=agent1)
robot2 = Robot(np.array([1, 0, math.pi/5*6]).T, agent=agent2, color='red')
robot3 = Robot(np.array([0, 0, 0]).T, color='blue')

world.append(robot1)
world.append(robot2)
world.append(robot3)

world.draw()