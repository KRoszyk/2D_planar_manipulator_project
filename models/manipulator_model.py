import numpy as np
from manipulators.planar_2dof import PlanarManipulator2DOF

class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.planar = PlanarManipulator2DOF(Tp)
        self.l1 = 0.5
        self.r1 = 0.5 * self.l1
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.5 * self.l2
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.5
        self.r3 = 0.01
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2


    def M(self, x):
        q1, q2, q1_dot, q2_dot = x
        # M matrix including m3 mass
        M1 = self.I_1 + self.I_2 + self.m1 * (self.l1 / 2) ** 2 + self.m2 * (self.l1 ** 2 + (self.l2 / 2) ** 2) + self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2) + 2*(self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.cos(q2)
        M2 = self.I_2 + self.m2 * (self.l2 / 2) ** 2 + self.I_3 + self.m3 * self.l2 ** 2 + (self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.cos(q2)
        M3 = M2
        M4 = self.I_2 + self.m2 * (self.l2 / 2) ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        M = np.array([[M1, M2], [M3, M4]])
        return M

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x
        # C matrix including m3 mass
        C1 = (-1)*(self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.sin(q2) * q2_dot
        C2 = (-1)*(self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.sin(q2) * (q1_dot + q2_dot)
        C3 = (self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.sin(q2) * q1_dot
        C4 = 0
        C = np.array([[C1, C2], [C3, C4]])
        return C

class ManipulatorModelParams:
    def __init__(self, Tp, m3, r3):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.5 * self.l1
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.5 * self.l2
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def M(self, x):
        q1, q2, q1_dot, q2_dot = x
        M1 = self.I_1 + self.I_2 + self.m1 * (self.l1 / 2) ** 2 + self.m2 * (
                    self.l1 ** 2 + (self.l2 / 2) ** 2) + self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2) + 2 * (
                         self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.cos(q2)
        M2 = self.I_2 + self.m2 * (self.l2 / 2) ** 2 + self.I_3 + self.m3 * self.l2 ** 2 + (
                    self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.cos(q2)
        M3 = M2
        M4 = self.I_2 + self.m2 * (self.l2 / 2) ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        M = np.array([[M1, M2], [M3, M4]])
        return M

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x
        C1 = (-1) * (self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.sin(q2) * q2_dot
        C2 = (-1) * (self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.sin(q2) * (q1_dot + q2_dot)
        C3 = (self.m2 * self.l1 * self.l2 / 2 + self.m3 * self.l1 * self.l2) * np.sin(q2) * q1_dot
        C4 = 0
        C = np.array([[C1, C2], [C3, C4]])
        return C

