import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from controllers.pd_controller import PDDecentralizedController
from manipulators.planar_2dof import PlanarManipulator2DOF
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3

Tp = 0.01
start = 0
end = 3
t = np.linspace(start, end, int((end - start) / Tp))
manipulator = PlanarManipulator2DOF(Tp)

kp1 = 3
kp2 = 0.8
kd1 = 7.5
kd2 = 10
fl_controller = PDDecentralizedController(kp1, kd1)
sl_controller = PDDecentralizedController(kp2, kd2)
traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)


ctrl = []
T = []
Q_d = []


def system(x, t):
    T.append(t)
    q_d, q_d_dot, q_d_ddot = traj_gen.generate(t)
    Q_d.append(q_d)
    print(q_d_ddot)
    u1 = fl_controller.calculate_control(x[0], x[2], q_d[0], q_d_dot[0], q_d_ddot[0])
    u2 = sl_controller.calculate_control(x[1], x[3], q_d[1], q_d_dot[1], q_d_ddot[1])
    control = np.stack([u1, u2])[:, np.newaxis]
    ctrl.append(control)
    x_dot = manipulator.x_dot(x, control)
    return x_dot[:, 0]


q_d, q_d_dot, q_d_ddot = traj_gen.generate(0.)
x = odeint(system, np.concatenate([q_d, q_d_dot], 0), t)
manipulator.plot(x)

"""
You can add here some plots of the state 'x' (consists of q and q_dot), controls 'ctrl', desired trajectory 'Q_d'
with respect to time 'T' to analyze what is going on in the system
"""
plt.plot(t, x[:, 1], 'r')
plt.plot(T, np.stack(Q_d, 0)[:, 1], 'b')
plt.show()
