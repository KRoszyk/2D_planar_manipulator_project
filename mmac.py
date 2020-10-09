import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from controllers.mma_controller import MMAController
from manipulators.mm_planar_2dof import MMPlanarManipulator2DOF
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3


Tp = 0.01
start = 0
end = 3
t = np.linspace(start, end, int((end - start) / Tp))
manipulator = MMPlanarManipulator2DOF(Tp)


# Choose the controller
controller = MMAController(Tp)
#controller = DummyController(Tp)

"""
Here you have some trajectory generators. You can use them to check your implementations.
"""
# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
#traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)


ctrl = []
T = []
Q_d = []


def system(x, t):
    T.append(t)
    q_d, q_d_dot, q_d_ddot = traj_gen.generate(t)
    Q_d.append(q_d)
    print(q_d_ddot)
    control = controller.calculate_control(x, q_d, q_d_dot, q_d_ddot[:, np.newaxis])
    ctrl.append(control)
    x_dot = manipulator.x_dot(x, control)
    controller.choose_model(x, control, x_dot)
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
