import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PlanarManipulator2DOF:
    """
    https://www.cds.caltech.edu/~murray/books/MLS/pdf/mls94-manipdyn_v1_2.pdf
    https://shodhganga.inflibnet.ac.in/bitstream/10603/26579/9/09_chapter4.pdf
    """
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.5 * self.l1
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.5 * self.l2
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.495
        self.r3 = 0.01
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def plot(self, x):
        fig, ax = plt.subplots()
        ln, = plt.plot([], [])

        def init():
            plt.xlim(-1.0, 1.0)
            plt.ylim(-1.0, 1.0)
            return ln,

        def update(i):
            ln.set_data(p[0, i], p[1, i])
            return ln,

        q1, q2, q1_dot, q2_dot = np.split(x, 4, axis=-1)
        p0 = np.array([np.zeros_like(q1), np.zeros_like(q1)])
        p1 = np.array([self.l1 * np.cos(q1), self.l1 * np.sin(q1)])
        p2 = p1 + np.array([self.l2 * np.cos(q1 + q2), self.l2 * np.sin(q1 + q2)])
        p = np.concatenate([p0, p1, p2], -1)
        ani = FuncAnimation(fig, update, frames=range(x.shape[0]),
                            init_func=init, blit=True, interval=int(self.Tp * 1000), repeat=False)
        plt.show()

    """
    Don't read the contents below unless you finished the first two classes.


























    Whether you really want to cheat?








    It will not give you anything!








    STAHP!












    PLEASE STOP!







    We can't go deeper!!!!















    There is nothing interesting below.











    Cheating is not the way you want to pass that subject.



    Believe me




















    Are you really still scrolling down? Are you mad?















    Please don't!



















    Congratulations! You failed!
    """

    def M(self, x):
        q1, q2, q1_dot, q2_dot = x
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        alpha = self.I_1 + self.I_2 + self.m1 * d1 ** 2 + self.m2 * (self.l1 ** 2 + d2 ** 2) + \
                self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        delta = self.I_2 + self.m2 * d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        m_11 = alpha + 2 * beta * np.cos(q2)
        m_12 = delta + beta * np.cos(q2)
        m_21 = m_12
        m_22 = delta
        return np.array([[m_11, m_12], [m_21, m_22]])

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x
        d2 = self.l2 / 2
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        c_11 = -beta * np.sin(q2) * q2_dot
        c_12 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        c_21 = beta * np.sin(q2) * q1_dot
        c_22 = 0
        return np.array([[c_11, c_12], [c_21, c_22]])

    def x_dot(self, x, u):
        invM = np.linalg.inv(self.M(x))
        zeros = np.zeros((2, 2), dtype=np.float32)
        A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
        b = np.concatenate([zeros, invM], 0)
        return A @ x[:, np.newaxis] + b @ u
