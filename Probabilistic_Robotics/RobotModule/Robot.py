import math
import matplotlib.patches as patches
import numpy as np


class Robot:
    def __init__(self, pose, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)

        elems = elems + ax.plot([x, xn], [y, yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu*math.cos(t0),
                                    nu*math.sin(t0),
                                    omega])*time
        else:
            return pose + np.array([nu / omega * (math.sin(t0 + omega*time) - math.sin(t0)),
                                    nu / omega * (-math.cos(t0 + omega*time) + math.cos(t0)),
                                    omega]) * time
