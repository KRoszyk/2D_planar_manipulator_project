import numpy as np
from .controller import Controller
from models import manipulator_model


class MMAController(Controller):
    def __init__(self, Tp):
        self.Tp = Tp
        model1 = manipulator_model.ManipulatorModelParams(Tp, 0.1, 0.05)
        model2 = manipulator_model.ManipulatorModelParams(Tp, 0.01, 0.01)
        model3 = manipulator_model.ManipulatorModelParams(Tp, 1.0, 0.3)
        self.models = [model1, model2, model3]
        self.i = 0
        self.Kd = 0.15
        self.Kp = 0.6

    def calculate_error(self, x_dot_new, x_dot):
        error = x_dot - x_dot_new
        return error

    def calculate_x_dot(self, x, u, model):
        M_inv = np.linalg.inv(model.M(x))
        A = np.concatenate([np.concatenate([np.zeros((2, 2), dtype=np.float32), np.eye(2)], 1), np.concatenate([np.zeros((2, 2), dtype=np.float32), -M_inv @ model.C(x)], 1)], 0)
        b = np.concatenate([np.zeros((2, 2), dtype=np.float32), M_inv], 0)
        x_dots_new = A @ x[:, np.newaxis] + b @ u
        return x_dots_new

    def choose_model(self, x, u, x_dot):
        x_dot_new = []
        error_list = []
        for i in range(0, 3):
            x_dot_new.append(self.calculate_x_dot(x, u, self.models[i]))
            error_list.append((self.calculate_error(x_dot_new[i], x_dot)**2))
        minimum = error_list[0]
        #print(error_list)
        for index in range(0, 3):
            if (sum(error_list[index]))**(1/2) <= (sum(minimum))**(1/2):
                # print("----------------------------")
                # print(error_list[index])
                # print(sum(error_list[index]))
                # print("----------------------------")
                minimum = error_list[index]
                self.i = index
        print("I chose the type of model: ", self.i)


    def calculate_control(self, x, q_r, q_r_d, desired_q_ddot):
        q1, q2, q1_dot, q2_dot = x
        q = np.array([[q1], [q2]])
        q_dot = np.array([[q1_dot], [q2_dot]])

        q_r = q_r[:, np.newaxis]
        q_r_d = q_r_d[:, np.newaxis]
        M = self.models[self.i].M(x)
        v = desired_q_ddot + self.Kd * (q_dot-q_r_d) + self.Kp * (q - q_r)
        return M @ (v + np.linalg.inv(M) @ self.models[self.i].C(x) @ q_dot)