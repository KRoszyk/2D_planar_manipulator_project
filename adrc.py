import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.adrc_controller import ADRController
from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from manipulators.planar_2dof import PlanarManipulator2DOF
from observers.eso import ESO
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3

Tp = 0.001
start = 0
end = 3
t = np.linspace(start, end, int((end - start) / Tp))
manipulator = PlanarManipulator2DOF(Tp)

b_est_1 = 10.
b_est_2 = 10.
kp_1 = 10.
kp_2 = 12
kd_1 = 2
kd_2 = 2
fl_controller = ADRController(b_est_1, kp_1, kd_1)
sl_controller = ADRController(b_est_2, kp_2, kd_2)

p = 200
l1 = 3*p
l2 = 3*p**2.
l3 = p**3
A = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
B1 = np.array([0., b_est_1, 0.])[:, np.newaxis]
B2 = np.array([0., b_est_2, 0.])[:, np.newaxis]
L1 = np.array([l1, l2, l3])[:, np.newaxis]
L2 = np.array([l1, l2, l3])[:, np.newaxis]
first_link_state_estimator = ESO(A, B1, L1)
second_link_state_estimator = ESO(A, B2, L2)

traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)


ctrl = []
T = []
Q_d = []


def system(x_and_eso, t):
    x = x_and_eso[:4]
    fl_estimates = x_and_eso[4:7]
    sl_estimates = x_and_eso[7:]
    T.append(t)
    q_d, q_d_dot, q_d_ddot = traj_gen.generate(t)
    Q_d.append(q_d)
    u1 = fl_controller.calculate_control(x[0], q_d[0], q_d_dot[0], q_d_ddot[0], fl_estimates)
    u2 = sl_controller.calculate_control(x[1], q_d[1], q_d_dot[1], q_d_ddot[1], sl_estimates)
    control = np.stack([u1, u2])[:, np.newaxis]
    ctrl.append(control)
    fl_eso_dot = first_link_state_estimator.compute_dot(fl_estimates, x[0], u1)
    sl_eso_dot = second_link_state_estimator.compute_dot(sl_estimates, x[1], u2)
    x_dot = manipulator.x_dot(x, control)
    x_and_eso_dot = np.concatenate([x_dot, fl_eso_dot, sl_eso_dot])
    return x_and_eso_dot[:, 0]


q_d, q_d_dot, q_d_ddot = traj_gen.generate(0.)
x = odeint(system, [*q_d, *q_d_dot, 0., 0., 0., 0., 0., 0.], t, hmax=1e-3)
manipulator.plot(x[:, :4])

"""
You can add here some plots of the state 'x' (consists of q and q_dot), controls 'ctrl', desired trajectory 'Q_d'
with respect to time 'T' to analyze what is going on in the system
"""
plt.plot(t, x[:, 1], 'r')
plt.plot(T, np.stack(Q_d, 0)[:, 1], 'b')
plt.show()
