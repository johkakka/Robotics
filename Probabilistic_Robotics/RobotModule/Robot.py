import math
import matplotlib.patches as patches
import numpy as np
from scipy.stats import expon, norm, uniform


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)

        elems += ax.plot([x, xn], [y, yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor is not None and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent is not None and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    def step(self, interval):
        if self.agent is None:
            return
        obs = self.sensor.data(self.pose) if self.sensor is not None else None
        nu, omege = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omege, interval, self.pose)

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu * math.cos(t0),
                                    nu * math.sin(t0),
                                    omega]) * time
        else:
            return pose + np.array([nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)),
                                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                                    omega * time])


class Robot(IdealRobot):
    def __init__(self, pose, agent=None, sensor=None, color='black',
                 noise_per_meter=5, noise_std=math.pi / 60,
                 bias_rate_stds=(0.1,0.1),
                 expected_stuck_time = 1e100, expected_escape_time = 1e-100,
                 expected_kidnap_time=1e100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)):
        super().__init__(pose, agent, sensor, color)
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)

        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False

        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        rx, ry = kidnap_range_x, kidnap_range_y
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1] - rx[0], ry[1] - ry[0], 2 * math.pi))


    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()

        return pose

    def bias(self, nu, omega):  # 追加
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega

    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck)

    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose

    def step(self, interval):
        if self.agent is None:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, interval)
        self.pose = self.state_transition(nu, omega, interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, interval)
        self.pose = self.kidnap(self.pose, interval)


class IdealCamera:
    def __init__(self, env_map,
                 distance_range=(0.5, 6.0),
                 direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []
        self.distance_range = distance_range
        self.direction_range = direction_range

    def is_visible(self, polarpos):
        if polarpos is None:
            return False
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] and \
               self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            if self.is_visible(p):
                observed.append((p, lm.id))

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi > np.pi:
            phi -= 2 * np.pi
        while phi < - np.pi:
            phi += 2 * np.pi
        return np.array([np.hypot(*diff), phi]).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')


class Camera(IdealCamera):
    def __init__(self, env_map,
                 distance_range=(0.5, 6.0),
                 direction_range=(-math.pi / 3, math.pi / 3),
                 distance_noise_rate=0.1, direction_noise=math.pi / 90,
                 distance_bias_rate_stddev=0.1, direction_bias_stddev=math.pi/90,
                 phantom_prob=0.0, phantom_range_x=(-5.0, 5.0), phantom_range_y=(-5.0, 5.0),
                 oversight_prob=0.1, occlusion_prob=0.0):
        super().__init__(env_map, distance_range, direction_range)

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

        self.distance_bias_rate_std = norm.rvs(scale=distance_bias_rate_stddev)
        self.direction_bias = norm.rvs(scale=direction_bias_stddev)

        rx, ry = phantom_range_x, phantom_range_y
        self.phantom_dist = uniform(loc=(rx[0], ry[0]), scale=(rx[1] - rx[0], ry[1] - ry[0]))
        self.phantom_prob = phantom_prob

        self.oversight_prob = oversight_prob
        self.occlusion_prob = occlusion_prob

    def noise(self, relpos):
        ell = norm.rvs(loc=relpos[0], scale=relpos[0] * self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T

    def bias(self, relpos):
        return relpos + np.array([relpos[0] * self.distance_bias_rate_std,
                                  self.direction_bias]).T

    def phantom(self, cam_pose, relpos):
        if uniform.rvs() < self.phantom_prob:
            pos = np.array(self.phantom_dist.rvs()).T
            return self.observation_function(cam_pose, pos)
        else:
            return relpos

    def oversight(self, relpos):  # 追加
        if uniform.rvs() < self.oversight_prob:
            return None
        else:
            return relpos

    def occlusion(self, relpos):
        if uniform.rvs() < self.occlusion_prob:
            ell = relpos[0] + uniform.rvs() * (self.distance_range[1] - relpos[0])
            return np.array([ell, relpos[1]]).T
        else:
            return relpos

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.oversight(z)
            z = self.occlusion(z)
            if self.is_visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed