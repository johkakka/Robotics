import math
import matplotlib.patches as patches
import numpy as np


class Robot:
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
                                    omega]) * time


class Camera:
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
               self.distance_range[0] <= polarpos[1] <= self.direction_range[1]

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
