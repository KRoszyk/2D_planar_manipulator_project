from random import random

import numpy as np
import matplotlib.pyplot as plt
from manipulators.planar_2dof import PlanarManipulator2DOF


class MMPlanarManipulator2DOF(PlanarManipulator2DOF):
    """
    https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-manipdyn_v1_2.pdf
    https://shodhganga.inflibnet.ac.in/bitstream/10603/26579/9/09_chapter4.pdf
    """
    def __init__(self, Tp):
        super(MMPlanarManipulator2DOF, self).__init__(Tp)
        self.i = 0
        self.mos = [0.1, 0.01, 1.]
        self.ros = [0.05, 0.01, 0.3]
        assert len(self.mos) == len(self.ros)

    def update_handled_object(self):
        self.m3 = self.mos[self.i]
        self.r3 = self.ros[self.i]
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def x_dot(self, x, u):
        invM = np.linalg.inv(self.M(x))
        zeros = np.zeros((2, 2), dtype=np.float32)
        A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
        b = np.concatenate([zeros, invM], 0)
        if random() < 0.05:
            self.i = (self.i + 1) % len(self.mos)
            self.update_handled_object()
        return A @ x[:, np.newaxis] + b @ u
