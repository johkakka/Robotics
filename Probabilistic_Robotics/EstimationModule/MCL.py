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
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)


class Particle:
    def __init__(self, pose, weight):
        self.pose = pose
        self.weight = weight

    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            ###パーティクルの位置と地図からランドマークの距離と方角を算出###
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.observation_function(self.pose, pos_on_map)

            ###尤度の計算###
            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev ** 2, direction_dev ** 2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)

    def update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        noised_nu = nu + ns[0] * math.sqrt(abs(nu) / time) + ns[1] * math.sqrt(abs(omega) / time)
        noised_omega = omega + ns[2] * math.sqrt(abs(nu) / time) + ns[3] * math.sqrt(abs(omega) / time)
        self.pose = IdealRobot.state_transition(noised_nu, noised_omega, time, self.pose)


class MCL:
    def __init__(self, envmap, init_pose, num, motion_noise_stds=None,
                 distance_dev_rate=0.14, direction_dev=0.05):
        if motion_noise_stds is None:
            motion_noise_stds = {"nn": 0.19, "no": 0.001, "on": 0.13, "oo": 0.2}
        self.particles = [Particle(init_pose, 1.0 / num) for i in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"] ** 2, v["no"] ** 2, v["on"] ** 2, v["oo"] ** 2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)

    def update(self, nu, omega, time):
        for p in self.particles: p.update(nu, omega, time, self.motion_noise_rate_pdf)

    def observation_update(self, observation):
        for p in self.particles: p.observation_update(observation, self.map, self.distance_dev_rate,
                                                      self.direction_dev)

    def draw(self, ax, elems):  # 次のように変更
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2]) * p.weight * len(self.particles) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys,
                               angles='xy', scale_units='xy', scale=1.5, color="blue", alpha=0.3))
