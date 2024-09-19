from pymoo.core.population import Population
from sklearn.preprocessing import StandardScaler
import numpy as np
from abc import ABC, abstractmethod


class Scaler(ABC):

    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def inverse_transform(self, X):
        pass


class BoundedScaler(Scaler):
    '''
    Scale data to [0, 1] according to bounds
    '''
    def __init__(self, bounds):
        self.bounds = bounds

    def transform(self, X):
        # return np.clip((X - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), 0, 1)
        return (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def inverse_transform(self, X):
        # return np.clip(X, 0, 1) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]


class DataBase(object):

    def __init__(self, n_obj, n_ieq_constr, xl, xu, problem):
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr
        self.obj_transform = StandardScaler()
        self.x_transform = BoundedScaler([xl, xu])
        self.constr_transform = StandardScaler()
        self.xl = xl
        self.xu = xu
        self.problem = problem

    def create_data(self, arch: Population) -> dict[str, np.ndarray]:
        pop_X = arch.get('X')
        pop_F = arch.get('F')
        self.x_transform.fit(pop_X)
        self.obj_transform.fit(pop_F)
        X_norm = self.x_transform.transform(pop_X)
        F_norm = self.obj_transform.transform(pop_F)
        # data = {'X': X_norm, 'F': F_norm, 'G': None}
        data = {'X': X_norm, 'F': F_norm, 'G': None}
        if self.n_ieq_constr > 0:
            pop_G = arch.get('G')
            self.constr_transform.fit(pop_G)
            G_norm = self.constr_transform.transform(pop_G)
            data['G'] = G_norm

        return data

    def create_obj_con_data(self, obj_arch, con_arch):
        data = {'objective': {'X': None, 'F': None}, 'constraint': {'X': [], 'con_value': []}, 'G_norm_0': None}
        pop_obj_X = obj_arch.get('X')
        pop_obj_F = obj_arch.get('F')
        pop_con_G_list = []
        for i in range(len(con_arch)):
            pop_con_G_list.append(con_arch[i].get('G')[:, i][:, None])
        pop_con_G = np.hstack(pop_con_G_list)
        self.x_transform.fit(pop_obj_X)
        self.obj_transform.fit(pop_obj_F)
        self.constr_transform.fit(pop_con_G)
        X_norm = self.x_transform.transform(pop_obj_X)
        F_norm = self.obj_transform.transform(pop_obj_F)
        G_norm = self.constr_transform.transform(pop_con_G)
        data['objective']['X'] = X_norm
        data['objective']['F'] = F_norm
        G_i_norm_0 = self.constr_transform.transform(np.zeros([1, self.n_ieq_constr]))[0, :]
        data['G_norm_0'] = G_i_norm_0

        for con_index in range(self.n_ieq_constr):
            arch_con_index = con_arch[con_index]
            arch_con_index_X = arch_con_index.get('X')
            G_i_norm = G_norm[:, con_index]
            X_norm = self.x_transform.transform(arch_con_index_X)
            data['constraint']['X'].append(X_norm)
            data['constraint']['con_value'].append(G_i_norm)

        return data
