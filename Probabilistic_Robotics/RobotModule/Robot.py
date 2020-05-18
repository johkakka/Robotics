import math
import matplotlib.patches as patches


class Robot:
    def __init__(self, pose, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color

    def draw(self, ax):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)

        ax.plot([x, xn], [y, yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        ax.add_patch(c)
