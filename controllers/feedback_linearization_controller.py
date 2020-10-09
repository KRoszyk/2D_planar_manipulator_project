import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller



class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kd = 0.3
        self.Kp = 0.6

    def calculate_control(self, x, v, q_d, q_d_dot, q_d_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        M = self.model.M(x)
        C = self.model.C(x)
        q = np.array([[q1], [q2]])
        q_dot = np.array([[q1_dot], [q2_dot]])
        v = q_d_ddot + self.Kd*(q_dot-q_d_dot)+self.Kp*(q-q_d)
        tau = M.dot(v)+C.dot(q_dot)

        return tau
