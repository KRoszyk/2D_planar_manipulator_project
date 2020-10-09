import numpy as np


class ESO:
    def __init__(self, A, B, L):
        self.A = A
        self.B = B
        self.L = L

    def compute_dot(self, eso_estimates, q, u):
        e = q - eso_estimates[0]
        # print(eso_estimates)
        z_tophat = eso_estimates[:, np.newaxis]
        z_dot = self.A @ z_tophat + self.B * u + self.L * e
        return z_dot
