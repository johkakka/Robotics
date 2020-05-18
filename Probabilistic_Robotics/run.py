import numpy as np
import math

from Robot import Robot
from World import World

world = World()

robot1 = Robot(np.array([2, 3, math.pi/5*6]).T)
robot2 = Robot(np.array([1, 0, math.pi/5*6]).T, color='red')

world.append(robot1)
world.append(robot2)

world.draw()