from Agent import *
from Robot import *
import math
import numpy as np
from scipy.stats import multivariate_normal


class EstimationAgent(Agent):
    def __init__(self, interval, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.interval = interval
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self, observation=None):
        self.estimator.update(self.prev_nu, self.prev_omega, self.interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)


class Particle:
    def __init__(self, pose):
        self.pose = pose

    def update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns[0] * math.sqrt(abs(nu) / time) + ns[1] * math.sqrt(abs(omega) / time)
        noised_omega = omega + ns[2] * math.sqrt(abs(nu) / time) + ns[3] * math.sqrt(abs(omega) / time)
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)


class MCL:
    def __init__(self, pose, num, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}):
        self.particles = [Particle(pose) for i in range(num)]

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.3))

    def update(self, nu, omega, time):
        for p in self.particles:
            p.update(nu, omega, time, self.motion_noise_rate_pdf)
