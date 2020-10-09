import numpy as np
from .controller import Controller


class ADRController(Controller):
    def __init__(self, b, kp, kd):
        self.b = b
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_d, q_d_dot, q_d_ddot, eso_estimates):
        q_est = eso_estimates[0]
        q_dot_est = eso_estimates[1]
        f = eso_estimates[2]
        v = q_d_ddot + self.kd * (q_d_dot - q_dot_est) + self.kp * (q - q_est)
        u = (v - f) / self.b
        return u
