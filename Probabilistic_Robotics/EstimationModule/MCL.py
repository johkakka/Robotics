from Agent import *
import math


class EstimationAgent(Agent):
    def __init__(self, nu, omega, estimater):
        super().__init__(nu, omega)
        self.estimater = estimater

    def draw(self, ax, elems):
        self.estimater.deaw(ax, elems)


class Particle:
    def __init__(self, pose):
        self.pose = pose


class MCL:
    def __init__(self, pose, num):
        self.particles = [Particle(pose) for i in range(num)]

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.5))