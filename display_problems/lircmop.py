from pymoo.core.problem import Problem
import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


class LIRCMOP(Problem):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=n_ieq_constr, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        pass

    def g1(self, X):
        # return a vector, 1-dim
        g1 = np.sum((X[:, 2::2] - np.sin(0.5 * np.pi * X[:, 0:1])) ** 2, axis=1)
        return g1

    def g2(self, X):
        g2 = np.sum((X[:, 1::2] - np.cos(0.5 * np.pi * X[:, 0:1])) ** 2, axis=1)
        return g2


class LIRCMOP1(LIRCMOP):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def constraints(self, X):
        a = 0.51
        b = 0.5
        c = np.zeros((X.shape[0], self.n_ieq_constr))
        c[:, 0] = (a - self.g1(X)) * (self.g1(X) - b)
        c[:, 1] = (a - self.g2(X)) * (self.g2(X) - b)
        return -1 * c

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + self.g1(x)
        f2 = 1 - x[:, 0] ** 2 + self.g2(x)
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x)

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 100)
        f2 = 1 - f1 ** 2
        pareto_front = np.column_stack([f1, f2]) + 0.5
        return pareto_front



class LIRCMOP2(LIRCMOP1):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + self.g1(x)
        f2 = 1 - np.sqrt(x[:, 0]) + self.g2(x)
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x)


class LIRCMOP3(LIRCMOP1):

    def __init__(self, n_var=30, n_ieq_constr=3):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def constraints(self, X):
        a = 0.51
        b = 0.5
        c = 20
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        cv[:, 0] = (a - self.g1(X)) * (self.g1(X) - b)
        cv[:, 1] = (a - self.g2(X)) * (self.g2(X) - b)
        cv[:, 2] = np.sin(c * np.pi * X[:, 0]) - 0.5
        return -1 * cv

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 1000)
        f2 = 1 - f1**2
        f_objs = np.column_stack([f1, f2])
        dis = np.sin(20 * np.pi * f1) < 0.5
        pareto_front = f_objs[~dis]
        pareto_front += 0.5
        return pareto_front

    def get_pf_region(self):
        pf_region = self._calc_pareto_front()
        return pf_region


class LIRCMOP4(LIRCMOP3):

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + self.g1(x)
        f2 = 1 - np.sqrt(x[:, 0]) + self.g2(x)
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x)

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 1000)
        f2 = 1 - np.sqrt(f1)
        f_objs = np.column_stack([f1, f2])
        dis = np.sin(20 * np.pi * f1) < 0.5
        pareto_front = f_objs[~dis]
        pareto_front += 0.5
        return pareto_front

    def get_pf_region(self):
        pf_region = self._calc_pareto_front()
        return pf_region


class LIRCMOP5(LIRCMOP):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def constraints(self, X, f1, f2):
        p_k = np.array([1.6, 2.5])
        q_k = np.array([1.6, 2.5])
        a_k = np.array([2, 2])
        b_k = np.array([4, 8])
        r = 0.1
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))

        cv[:, 0] = (((f1 - p_k[0]) * np.cos(theta) - (f2 - q_k[0]) * np.sin(theta)) ** 2 / (a_k[0] ** 2)) + \
                   (((f1 - p_k[0]) * np.sin(theta) + (f2 - q_k[0]) * np.cos(theta)) ** 2 / (b_k[0] ** 2)) - r
        cv[:, 1] = (((f1 - p_k[1]) * np.cos(theta) - (f2 - q_k[1]) * np.sin(theta)) ** 2 / (a_k[1] ** 2)) + \
                   (((f1 - p_k[1]) * np.sin(theta) + (f2 - q_k[1]) * np.cos(theta)) ** 2 / (b_k[1] ** 2)) - r

        return -1 * cv

    def g1(self, X):
        i = np.array([np.arange(1, self.n_var + 1)[2::2] for k in range(len(X))])
        g1 = np.sum((X[:, 2::2] - np.sin(0.5 * i * np.pi * X[:, 0:1] / self.n_var)) ** 2, axis=1)
        return g1

    def g2(self, X):
        j = np.array([np.arange(1, self.n_var + 1)[1::2] for k in range(len(X))])
        g2 = np.sum((X[:, 1::2] - np.cos(0.5 * j * np.pi * X[:, 0:1] / self.n_var)) ** 2, axis=1)
        return g2

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + 10 * self.g1(x) + 0.7057
        f2 = 1 - np.sqrt(x[:, 0]) + 10 * self.g2(x) + 0.7057

        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x, f1, f2)

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0.7057, 5, sample_num), np.linspace(0.7057, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        t = np.sqrt(x - 0.7057) + y >= 1.7057
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 1000)
        f2 = 1 - np.sqrt(f1)
        f_objs = np.column_stack([f1, f2])
        f_objs += 0.7057
        cv = self.constraints(f1, f_objs[:, 0], f_objs[:, 1])
        feasible = np.all(cv <= 0, axis=1)
        pareto_front = f_objs[feasible]
        return pareto_front


class LIRCMOP6(LIRCMOP5):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + 10 * self.g1(x) + 0.7057
        f2 = 1 - x[:, 0] ** 2 + 10 * self.g2(x) + 0.7057
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p_k = np.array([1.8, 2.8])
        q_k = np.array([1.8, 2.8])
        a_k = np.array([2, 2])
        b_k = np.array([8, 8])
        r = 0.1
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        cv[:, 0] = (((f1 - p_k[0]) * np.cos(theta) - (f2 - q_k[0]) * np.sin(theta)) ** 2 / (a_k[0] ** 2)) + \
                   (((f1 - p_k[0]) * np.sin(theta) + (f2 - q_k[0]) * np.cos(theta)) ** 2 / (b_k[0] ** 2)) - r
        cv[:, 1] = (((f1 - p_k[1]) * np.cos(theta) - (f2 - q_k[1]) * np.sin(theta)) ** 2 / (a_k[1] ** 2)) + \
                   (((f1 - p_k[1]) * np.sin(theta) + (f2 - q_k[1]) * np.cos(theta)) ** 2 / (b_k[1] ** 2)) - r

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0.7057, 5, sample_num), np.linspace(0.7057, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        # (x-0.7057).^2+y>=1.7057
        t = (x - 0.7057)**2 + y >= 1.7057
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 1000)
        f2 = 1 - f1**2
        f_objs = np.column_stack([f1, f2])
        f_objs += 0.7057
        cv = self.constraints(f1, f_objs[:, 0], f_objs[:, 1])
        feasible = np.all(cv <= 0, axis=1)
        pareto_front = f_objs[feasible]
        return pareto_front


class LIRCMOP7(LIRCMOP6):

    def __init__(self, n_var=30, n_ieq_constr=3):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + 10 * self.g1(x) + 0.7057
        f2 = 1 - np.sqrt(x[:, 0]) + 10 * self.g2(x) + 0.7057
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p_k = [1.2, 2.25, 3.5]
        q_k = [1.2, 2.25, 3.5]
        a_k = [2, 2.5, 2.5]
        b_k = [6, 12, 10]
        r = 0.1
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], 3))
        cv[:, 0] = (((f1 - p_k[0]) * np.cos(theta) - (f2 - q_k[0]) * np.sin(theta)) ** 2 / (a_k[0] ** 2)) + \
                   (((f1 - p_k[0]) * np.sin(theta) + (f2 - q_k[0]) * np.cos(theta)) ** 2 / (b_k[0] ** 2)) - r
        cv[:, 1] = (((f1 - p_k[1]) * np.cos(theta) - (f2 - q_k[1]) * np.sin(theta)) ** 2 / (a_k[1] ** 2)) + \
                   (((f1 - p_k[1]) * np.sin(theta) + (f2 - q_k[1]) * np.cos(theta)) ** 2 / (b_k[1] ** 2)) - r
        cv[:, 2] = (((f1 - p_k[2]) * np.cos(theta) - (f2 - q_k[2]) * np.sin(theta)) ** 2 / (a_k[2] ** 2)) + \
                   (((f1 - p_k[2]) * np.sin(theta) + (f2 - q_k[2]) * np.cos(theta)) ** 2 / (b_k[2] ** 2)) - r

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0.7057, 5, sample_num), np.linspace(0.7057, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        t = np.sqrt(x - 0.7057) + y >= 1.7057
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        R = np.zeros((1000, 2))
        R[:, 0] = np.linspace(0, 1, 1000)
        R[:, 1] = 1 - np.sqrt(R[:, 0])
        R = R + 0.7057
        theta = -0.25 * np.pi
        c1 = 0.1 - (((R[:, 0] - 1.2) * np.cos(theta) - (R[:, 1] - 1.2) * np.sin(theta)) ** 2 / (2 ** 2)) \
             - (((R[:, 0] - 1.2) * np.sin(theta) + (R[:, 1] - 1.2) * np.cos(theta)) ** 2 / (6 ** 2))

        invalid = c1 > 0
        while np.any(invalid):
            R[invalid] = (R[invalid] - 0.7057) * 1.001 + 0.7057
            c1 = 0.1 - (((R[:, 0] - 1.2) * np.cos(theta) - (R[:, 1] - 1.2) * np.sin(theta)) ** 2 / (2 ** 2)) \
                 - (((R[:, 0] - 1.2) * np.sin(theta) + (R[:, 1] - 1.2) * np.cos(theta)) ** 2 / (6 ** 2))
            invalid = c1 > 0

        return R


class LIRCMOP8(LIRCMOP7):

    def __init__(self, n_var=30, n_ieq_constr=3):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[:, 0] + 10 * self.g1(x) + 0.7057
        f2 = 1 - x[:, 0] ** 2 + 10 * self.g2(x) + 0.7057
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p_k = [1.2, 2.25, 3.5]
        q_k = [1.2, 2.25, 3.5]
        a_k = [2, 2.5, 2.5]
        b_k = [6, 12, 10]
        r = 0.1
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        for i in range(self.n_ieq_constr):
            cv[:, i] = ((f1 - p_k[i]) * np.cos(theta) - (f2 - q_k[i]) * np.sin(theta)) ** 2 / a_k[i] ** 2 + \
                       ((f1 - p_k[i]) * np.sin(theta) + (f2 - q_k[i]) * np.cos(theta)) ** 2 / b_k[i] ** 2 - r

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0.7057, 5, sample_num), np.linspace(0.7057, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        t = np.sqrt(x - 0.7057) + y >= 1.7057
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z


class LIRCMOP9(LIRCMOP5):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1.7057 * x[:, 0] * (10 * self.g1(x) + 1)
        f2 = 1.7057 * (1 - x[:, 0]**2) * (10 * self.g2(x) + 1)
        out['F'] = np.column_stack([f1, f2])
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p1 = 1.4
        q1 = 1.4
        a1 = 1.5
        b1 = 6.0
        r = 0.1
        alpha = 0.25 * np.pi
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        cv[:, 0] = ((f1 - p1) * np.cos(theta) - (f2 - q1) * np.sin(theta)) ** 2 / a1 ** 2 + \
                   ((f1 - p1) * np.sin(theta) + (f2 - q1) * np.cos(theta)) ** 2 / b1 ** 2 - r

        cv[:, 1] = f1 * np.sin(alpha) + f2 * np.cos(alpha) - \
                   np.sin(4 * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha))) - 2

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0, 5, sample_num), np.linspace(0, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        # (x/1.7057).^2+y/1.7057>=1
        t = (x/1.7057)**2 + y/1.7057 >= 1
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 1000)
        f2 = 1 - f1**2
        f_objs = np.column_stack([f1, f2])
        f_objs = f_objs * 1.7057
        cv = self.constraints(f_objs[:, 0], f_objs[:, 0], f_objs[:, 1])
        f_fea = f_objs[np.all(cv <= 0, axis=1)]
        t = np.array([[0.0, 2.182], [1.856, 0.0]])
        pareto_front = np.row_stack([f_fea, t])
        return pareto_front


class LIRCMOP10(LIRCMOP5):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1.7057 * x[:, 0] * (10 * self.g1(x) + 1)
        f2 = 1.7057 * (1 - np.sqrt(x[:, 0])) * (10 * self.g2(x) + 1)
        out['F'] = np.column_stack((f1, f2))
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p1 = 1.1
        q1 = 1.2
        a1 = 2.0
        b1 = 4.0
        r = 0.1
        alpha = 0.25 * np.pi
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        cv[:, 0] = ((f1 - p1) * np.cos(theta) - (f2 - q1) * np.sin(theta)) ** 2 / a1 ** 2 + \
                   ((f1 - p1) * np.sin(theta) + (f2 - q1) * np.cos(theta)) ** 2 / b1 ** 2 - r

        cv[:, 1] = f1 * np.sin(alpha) + f2 * np.cos(alpha) - \
                   np.sin(4 * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha))) - 1

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0, 5, sample_num), np.linspace(0, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        t = np.sqrt(x / 1.7057) + y/1.7057 >= 1
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        f1 = np.linspace(0, 1, 1000)
        f2 = 1 - np.sqrt(f1)
        f_objs = np.column_stack([f1, f2])
        f_objs = f_objs * 1.7057
        cv = self.constraints(f_objs[:, 0], f_objs[:, 0], f_objs[:, 1])
        fea = np.all(cv <= 0, axis=1)
        f_objs_fea = f_objs[fea]
        t = np.array([[1.747, 0.0]])
        pareto_front = np.row_stack([f_objs_fea, t])
        return pareto_front


class LIRCMOP11(LIRCMOP5):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1.7057 * x[:, 0] * (10 * self.g1(x) + 1)
        f2 = 1.7057 * (1 - np.sqrt(x[:, 0])) * (10 * self.g2(x) + 1)
        out['F'] = np.column_stack((f1, f2))
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p1 = 1.2
        q1 = 1.2
        a1 = 1.5
        b1 = 5.0
        r = 0.1
        alpha = 0.25 * np.pi
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        cv[:, 0] = ((f1 - p1) * np.cos(theta) - (f2 - q1) * np.sin(theta)) ** 2 / a1 ** 2 + \
                   ((f1 - p1) * np.sin(theta) + (f2 - q1) * np.cos(theta)) ** 2 / b1 ** 2 - r

        cv[:, 1] = f1 * np.sin(alpha) + f2 * np.cos(alpha) - \
                   np.sin(4 * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha))) - 2.1

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0, 5, sample_num), np.linspace(0, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        t = np.sqrt(x / 1.7057) + y/1.7057 >= 1
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        pareto_front = np.array([[1.3965, 0.1591], [1.0430, 0.5127], [0.6894, 0.8662], [0.3359, 1.2198],
                                 [0.0106, 1.6016], [0, 2.1910], [1.8730, 0]])
        return pareto_front


class LIRCMOP12(LIRCMOP10):

    def __init__(self, n_var=30, n_ieq_constr=2):
        super().__init__(n_var=n_var, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 1.7057 * x[:, 0] * (10 * self.g1(x) + 1)
        f2 = 1.7057 * (1 - x[:, 0] ** 2) * (10 * self.g2(x) + 1)
        out['F'] = np.column_stack((f1, f2))
        out['G'] = self.constraints(x, f1, f2)

    def constraints(self, X, f1, f2):
        p1 = 1.6
        q1 = 1.6
        a1 = 1.5
        b1 = 6.0
        r = 0.1
        alpha = 0.25 * np.pi
        theta = -0.25 * np.pi
        cv = np.zeros((X.shape[0], self.n_ieq_constr))
        cv[:, 0] = ((f1 - p1) * np.cos(theta) - (f2 - q1) * np.sin(theta)) ** 2 / a1 ** 2 + \
                   ((f1 - p1) * np.sin(theta) + (f2 - q1) * np.cos(theta)) ** 2 / b1 ** 2 - r

        cv[:, 1] = f1 * np.sin(alpha) + f2 * np.cos(alpha) - \
                   np.sin(4 * np.pi * (f1 * np.cos(alpha) - f2 * np.sin(alpha))) - 2.5

        return -1.0 * cv

    def get_pf_region(self):
        sample_num = 1000
        x, y = np.linspace(0, 5, sample_num), np.linspace(0, 5, sample_num)
        x, y = np.meshgrid(x, y)
        f1 = x.reshape(sample_num * sample_num)
        f2 = y.reshape(sample_num * sample_num)
        cv = self.constraints(f1, f1, f2)
        feas = np.all(cv <= 0, axis=1)
        z = np.ones_like(x)
        t = np.power(x / 1.7057, 2) + y/1.7057 >= 1
        z[feas.reshape((sample_num, sample_num)) & t] = 0
        return x, y, z

    def _calc_pareto_front(self, *args, **kwargs):
        pareto_front = np.array([[1.6794, 0.4419], [1.3258, 0.7955], [0.9723, 1.1490], [2.0320, 0.0990],
                                 [0.6187, 1.5026], [0.2652, 1.8562], [0, 2.2580], [2.5690, 0]])
        return pareto_front


class LIRCMOP13(Problem):

    def __init__(self, n_var=30, n_obj=3, n_ieq_constr=2, xl=0.0, xu=1.0):
        super(LIRCMOP13, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        g1 = self.g1(x)
        f1 = (1.7057 + g1) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
        f2 = (1.7057 + g1) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
        f3 = (1.7057 + g1) * np.sin(0.5 * np.pi * x[:, 0])
        out['F'] = np.column_stack((f1, f2, f3))
        out['G'] = self.constrains(x, f1, f2, f3)

    def g1(self, X):
        return 10 * np.sum((X[:, 2:] - 0.5) ** 2, axis=1)

    def constrains(self, x, f1, f2, f3):
        cv = np.zeros((x.shape[0], self.n_ieq_constr))
        g = f1 ** 2 + f2 ** 2 + f3 ** 2
        cv[:, 0] = (g - 9) * (g - 4)
        cv[:, 1] = (g - 3.61) * (g - 3.24)
        return -1.0 * cv

    def _calc_pareto_front(self, *args, **kwargs):
        f_objs = get_reference_directions("uniform", n_dim=3, n_points=300)
        t = np.sqrt(np.sum(f_objs**2, axis=1))[:, None]
        pareto_front = 1.7057 * f_objs / t
        return pareto_front

    def get_pf_region(self):
        a = np.linspace(0, np.pi/2, 10)[:, None]
        x = np.sin(a) @ np.cos(a.T) * 1.7057
        y = np.sin(a) @ np.sin(a.T) * 1.7057
        z = np.cos(a) @ np.ones_like(a.T) * 1.7057

        return x, y, z


class LIRCMOP14(LIRCMOP13):

    def __init__(self, n_var=30, n_obj=3, n_ieq_constr=3, xl=0.0, xu=1.0):
        super(LIRCMOP14, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        g1 = self.g1(x)
        f1 = (1.7057 + g1) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
        f2 = (1.7057 + g1) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
        f3 = (1.7057 + g1) * np.sin(0.5 * np.pi * x[:, 0])
        out['F'] = np.column_stack((f1, f2, f3))
        out['G'] = self.constrains(x, f1, f2, f3)

    def constrains(self, x, f1, f2, f3):
        cv = np.zeros((x.shape[0], self.n_ieq_constr))
        g = f1 ** 2 + f2 ** 2 + f3 ** 2
        cv[:, 0] = (g - 9) * (g - 4)
        cv[:, 1] = (g - 3.61) * (g - 3.24)
        cv[:, 2] = (g - 3.0625) * (g - 2.56)
        return -1.0 * cv

    def _calc_pareto_front(self, *args, **kwargs):
        f_objs = get_reference_directions("uniform", n_dim=3, n_points=300)
        t = np.sqrt(np.sum(f_objs**2, axis=1))[:, None]
        pareto_front = np.sqrt(3.0625) * f_objs / t
        return pareto_front

