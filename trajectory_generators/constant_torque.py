import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class ConstantTorque(TrajectoryGenerator):
    def __init__(self, c):
        self.c = c

    def generate(self, t):
        q = np.zeros_like(self.c)
        q_dot = np.zeros_like(self.c)
        q_ddot = self.c
        return q, q_dot, q_ddot
