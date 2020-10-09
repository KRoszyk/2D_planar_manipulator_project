import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        e = q_d - q
        e_dot = q_d_dot - q_dot
        u = self.kp * e + self.kd * e_dot
        return u
