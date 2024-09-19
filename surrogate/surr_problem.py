from pymoo.core.problem import Problem
from surrogate.base import SurrogateModel
import numpy as np


class SurrogateProblem(Problem):
    
    def __init__(self, n_var, n_obj, n_ieq_constr, surr_model: SurrogateModel, G_norm_0: np.ndarray,
                 lambda_F_lcb: float, lambda_cv_ucb: float, norm_z, xl: float, xu: float, epsilon_cv: float):
        super(SurrogateProblem, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.surr_model = surr_model
        self.G_norm_0 = G_norm_0
        self.lambda_F_lcb = lambda_F_lcb
        self.lambda_cv_ucb = lambda_cv_ucb
        self.norm_z = norm_z
        self.epsilon_cv = epsilon_cv

    def _evaluate(self, x, out, *args, **kwargs):
        pred_res = self.surr_model.evaluate(x, std=True)
        # out['F'] = np.maximum(pred_res['F'], self.norm_z)
        out['F'] = pred_res['F']
        out['G'] = pred_res['G'] - self.G_norm_0 - self.epsilon_cv


class SurrogateProblemNoCV(Problem):

    def __init__(self, n_var, n_obj, n_ieq_constr, surr_model: SurrogateModel, G_norm_0: np.ndarray,
                 lambda_F_lcb: float, lambda_cv_ucb: float, xl: float, xu: float):
        super(SurrogateProblemNoCV, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.surr_model = surr_model
        self.G_norm_0 = G_norm_0
        self.lambda_F_lcb = lambda_F_lcb
        self.lambda_cv_ucb = lambda_cv_ucb
        self.fix_flag = False
        self.z = None

    def _evaluate(self, x, out, *args, **kwargs):
        pred_res = self.surr_model.evaluate(x, std=True, calc_G=False)
        if self.fix_flag:
            out['F'] = np.maximum(pred_res['F'] - self.lambda_F_lcb * pred_res['FS'], self.z)
        else:
            out['F'] = pred_res['F'] - self.lambda_F_lcb * pred_res['FS']

    def set_surrogate(self, surrogate_model):
        self.surr_model = surrogate_model

    def set_fix_flag(self, flag: bool, z_min: np.ndarray):
        self.fix_flag = flag
        self.z = z_min


class SurrogateProblemEpsilonCV(Problem):

    def __init__(self, n_var, n_obj, n_ieq_constr, surr_model: SurrogateModel, G_norm_0: np.ndarray,
                 lambda_F_lcb: float, lambda_cv_ucb: float, epsilon_cv: float, xl: float, xu: float):
        super(SurrogateProblemEpsilonCV, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.surr_model = surr_model
        self.G_norm_0 = G_norm_0
        self.lambda_F_lcb = lambda_F_lcb
        self.lambda_cv_ucb = lambda_cv_ucb
        self.epsilon_cv = epsilon_cv

    def _evaluate(self, x, out, *args, **kwargs):
        pred_res = self.surr_model.evaluate(x, std=True, calc_G=False)
        out['F'] = pred_res['F'] - self.lambda_F_lcb * pred_res['FS']

    def set_surrogate(self, surrogate_model):
        self.surr_model = surrogate_model




