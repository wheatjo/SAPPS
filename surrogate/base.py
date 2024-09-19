from pymoo.core.problem import Problem
from abc import abstractmethod
import numpy as np
"""
Surrogate model that predicts the performance of given design variables
"""


class SurrogateModel(object):

    def __init__(self, n_var, n_obj, n_ieq_constr):
        super(SurrogateModel, self).__init__()
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr


    @abstractmethod
    def fit(self, X: np.ndarray, F: np.ndarray, G):
        """
        Fit the surrogate model from data (X, F, G)
        :param X:
        :param F:
        :param G:
        :return: None
        """

        pass


    def evaluate(self, X: np.ndarray, std=False, calc_G=True, calc_gradient=False, calc_hessian=False) -> np.ndarray:
        """
        Predict the performance given set of design variables X
        Input:
            std / calc_gradient / calc_hessian : whether to calculate std / gradient / hessian of prediction
        Output:
            val['F']: mean, shape (N, n_obj)
            val['dF']: gradient of mean, shape (N, n_obj, n_var)
            val['hF']: hessian of mean, shape (N, n_obj, n_var, n_var)
            val['FS']: std, shape (N, n_obj)
            val['dFS']: gradient of std, shape (N, n_obj, n_var)
            val['hFS']: hessian of std, shape (N, n_obj, n_var, n_var)

            val['G']: mean, shape (N, n_ieq_constr)
            val['dG']: gradient of mean, shape (N, n_obj, n_var)
            val['hG']: hessian of mean, shape (N, n_obj, n_var, n_var)
            val['SG']: std, shape (N, n_obj)
            val['dSG']: gradient of std, shape (N, n_obj, n_var)
            val['hSG']: hessian of std, shape (N, n_obj, n_var, n_var)
        """
        pass
