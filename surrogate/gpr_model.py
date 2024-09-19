import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from surrogate.base import SurrogateModel
from utilsmodule.tools import safe_divide


def constrained_optimization(obj_func, initial_theta, bounds):
    opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
    '''
    NOTE: Temporarily disable the checking below because this error sometimes occurs:
        ConvergenceWarning: lbfgs failed to converge (status=2):
        ABNORMAL_TERMINATION_IN_LNSRCH
        , though we already optimized enough number of iterations and scaled the data.
        Still don't know the exact reason of this yet.
    '''
    # _check_optimize_result("lbfgs", opt_res)
    return opt_res.x, opt_res.fun


def get_gp_kernel(nu, n_var):
    if nu > 0:
        main_kernel = Matern(length_scale=np.ones(n_var), length_scale_bounds=(1e-5, 1e5),
                             nu=0.5 * nu)
    else:
        main_kernel = RBF(length_scale=np.ones(n_var), length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)))

    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(np.sqrt(1e-10), np.sqrt(1e10))) * \
             main_kernel + ConstantKernel(constant_value=1e-2, constant_value_bounds=(np.exp(-6), np.exp(0)))

    return kernel


def gp_predict(gp: GaussianProcessRegressor, X, nu, std=False, calc_gradient=False, calc_hessian=False):

    res = {}

    K = gp.kernel_(X, gp.X_train_)  # K: shape (N, N_train)
    f_mean = K.dot(gp.alpha_)
    res['mean'] = f_mean
    K_inv = None
    if std:
        L_inv = solve_triangular(gp.L_.T,
                                 np.eye(gp.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)

        y_var = gp.kernel_.diag(X)
        y_var -= np.einsum("ij,ij->i",
                           np.dot(K, K_inv), K)

        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            y_var[y_var_negative] = 0.0

        f_std = np.sqrt(y_var)
        res['std'] = np.where(f_std < 1e-8, 1e-8, f_std)

    if not (calc_gradient or calc_hessian):
        return res

    ell = np.exp(gp.kernel_.theta[1:-1])  # ell: shape (n_var,)
    sf2 = np.exp(gp.kernel_.theta[0])  # sf2: shape (1,)
    d = np.expand_dims(cdist(X / ell, gp.X_train_ / ell), 2)  # d: shape (N, N_train, 1)
    X_, X_train_ = np.expand_dims(X, 1), np.expand_dims(gp.X_train_, 0)
    dd_N = X_ - X_train_  # numerator
    dd_D = d * ell ** 2  # denominator
    dd = safe_divide(dd_N, dd_D)  # dd: shape (N, N_train, n_var)

    dK = None
    if calc_gradient or calc_hessian:
        if nu == 1:
            dK = -sf2 * np.exp(-d) * dd

        elif nu == 3:
            dK = -3 * sf2 * np.exp(-np.sqrt(3) * d) * d * dd

        elif nu == 5:
            dK = -5. / 3 * sf2 * np.exp(-np.sqrt(5) * d) * (1 + np.sqrt(5) * d) * d * dd

        else:  # RBF
            dK = -sf2 * np.exp(-0.5 * d ** 2) * d * dd

    dK_T = dK.transpose(0, 2, 1)  # dK: shape (N, N_train, n_var), dK_T: shape (N, n_var, N_train)

    if calc_gradient:
        dy_mean = dK_T @ gp.alpha_  # gp.alpha_: shape (N_train,)
        res['d_mean'] = dy_mean  # dy_mean: shape (N, n_var)

        # TODO: check
        if std:
            K = np.expand_dims(K, 1)  # K: shape (N, 1, N_train)
            K_Ki = K @ K_inv  # gp._K_inv: shape (N_train, N_train), K_Ki: shape (N, 1, N_train)
            dK_Ki = dK_T @ K_inv  # dK_Ki: shape (N, n_var, N_train)

            dy_var = -np.sum(dK_Ki * K + K_Ki * dK_T, axis=2)  # dy_var: shape (N, n_var)
            # print(dy_var.shape)
            # print(np.expand_dims(y_std,1).shape)
            # print(np.expand_dims(y_std,1).shape)
            # dy_std = 0.5 * safe_divide(dy_var, y_std) # dy_std: shape (N, n_var)
            if np.min(f_std) != 0:
                dy_std = 0.5 * dy_var / np.expand_dims(f_std, 1)  # dy_std: shape (N, n_var)
            else:
                dy_std = np.zeros(dy_var.shape)
            res['d_std'] = dy_std

    if calc_hessian:
        d = np.expand_dims(d, 3)  # d: shape (N, N_train, 1, 1)
        dd = np.expand_dims(dd, 2)  # dd: shape (N, N_train, 1, n_var)
        hd_N = d * np.expand_dims(np.eye(len(ell)), (0, 1)) - np.expand_dims(X_ - X_train_, 3) * dd  # numerator
        hd_D = d ** 2 * np.expand_dims(ell ** 2, (0, 1, 3))  # denominator
        hd = safe_divide(hd_N, hd_D)  # hd: shape (N, N_train, n_var, n_var)

        if nu == 1:
            hK = -sf2 * np.exp(-d) * (hd - dd ** 2)

        elif nu == 3:
            hK = -3 * sf2 * np.exp(-np.sqrt(3) * d) * (d * hd + (1 - np.sqrt(3) * d) * dd ** 2)

        elif nu == 5:
            hK = -5. / 3 * sf2 * np.exp(-np.sqrt(5) * d) * (
                    -5 * d ** 2 * dd ** 2 + (1 + np.sqrt(5) * d) * (dd ** 2 + d * hd))

        else:  # RBF
            hK = -sf2 * np.exp(-0.5 * d ** 2) * ((1 - d ** 2) * dd ** 2 + d * hd)

        hK_T = hK.transpose(0, 2, 3,
                            1)  # hK: shape (N, N_train, n_var, n_var), hK_T: shape (N, n_var, n_var, N_train)

        hy_mean = hK_T @ gp.alpha_  # hy_mean: shape (N, n_var, n_var)
        res['h_mean'] = hy_mean

        # TODO: check
        if std:
            K = np.expand_dims(K, 2)  # K: shape (N, 1, 1, N_train)
            dK = np.expand_dims(dK_T, 2)  # dK: shape (N, n_var, 1, N_train)
            dK_Ki = np.expand_dims(dK_Ki, 2)  # dK_Ki: shape (N, n_var, 1, N_train)
            hK_Ki = hK_T @ K_inv  # hK_Ki: shape (N, n_var, n_var, N_train)

            hy_var = -np.sum(hK_Ki * K + 2 * dK_Ki * dK + K_Ki * hK_T,
                             axis=3)  # hy_var: shape (N, n_var, n_var)
            hy_std = 0.5 * safe_divide(hy_var * f_std - dy_var * dy_std,
                                       y_var)  # hy_std: shape (N, n_var, n_var)
            res['h_std'] = hy_std

    return res


class GaussianProcess(SurrogateModel):

    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn

    '''
    Gaussian process
    '''

    def __init__(self, n_var, n_obj, n_ieq_constr, nu, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr)

        self.nu = nu
        self.eps = 1e-8
        self.F_gps = []
        self.G_gps = []
        self.X_transformer = None
        for _ in range(self.n_obj):
            gp = GaussianProcessRegressor(kernel=get_gp_kernel(self.nu, self.n_var), optimizer=constrained_optimization)
            self.F_gps.append(gp)

        for _ in range(self.n_ieq_constr):
            gp = GaussianProcessRegressor(kernel=get_gp_kernel(self.nu, self.n_var), optimizer=constrained_optimization)
            self.G_gps.append(gp)

    def fit(self, X, F, G=None):

        for i, gp in enumerate(self.F_gps):
            gp.fit(X, F[:, i])

        if G is not None:
            for i, gp in enumerate(self.G_gps):
                gp.fit(X, G[:, i])

    def fit_obj_con_split(self, fit_data):
        fit_obj_F = fit_data['objective']['F']
        fit_obj_X = fit_data['objective']['X']
        for i, gp in enumerate(self.F_gps):
            gp.fit(fit_obj_X, fit_obj_F[:, i])

        for i, gp in enumerate(self.G_gps):
            fit_G_i_X = fit_data['constraint']['X'][i]
            fit_G_i_value = fit_data['constraint']['con_value'][i]
            gp.fit(fit_G_i_X, fit_G_i_value)

    def evaluate(self, X, std=True, calc_G=True, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], []  # mean
        FS, dFS, hFS = [], [], []  # std
        G, dG, hG = [], [], []
        GS, dGS, hGS = [], [], []

        for gp in self.F_gps:
            res = gp_predict(gp, X, self.nu, std, calc_gradient, calc_hessian)
            F.append(res['mean'])
            if std:
               FS.append(res['std'])

            if calc_gradient:
                dF.append(res['d_mean'])

            if calc_gradient and std:
                dFS.append(res['d_std'])

            if calc_hessian:
                hF.append(res['h_mean'])

            if calc_hessian and std:
                hFS.append(res['h_std'])

        if self.n_ieq_constr > 0 and calc_G:
            for gp in self.G_gps:
                res = gp_predict(gp, X, self.nu, std, calc_gradient, calc_hessian)
                G.append(res['mean'])
                if std:
                   GS.append(res['std'])

                if calc_gradient:
                    dG.append(res['d_mean'])

                if calc_gradient and std:
                    dGS.append(res['d_std'])

                if calc_hessian:
                    hG.append(res['h_mean'])

                if calc_hessian and std:
                    hGS.append(res['h_std'])

        F = np.stack(F, axis=1)
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None

        S = np.stack(FS, axis=1) if std else None
        dS = np.stack(dFS, axis=1) if std and calc_gradient else None
        hS = np.stack(hFS, axis=1) if std and calc_hessian else None

        if self.n_ieq_constr > 0 and calc_G:
            G = np.stack(G, axis=1)
            dG = np.stack(dG, axis=1) if calc_gradient else None
            hG = np.stack(hG, axis=1) if calc_hessian else None

            GS = np.stack(GS, axis=1) if std else None
            dGS = np.stack(dGS, axis=1) if std and calc_gradient else None
            hGS = np.stack(hGS, axis=1) if std and calc_hessian else None

            out = {'F': F, 'dF': dF, 'hF': hF, 'FS': S, 'dFS': dS, 'hFS': hS, 'G': G, 'dG': dG, 'hG': hG,
                   'GS': GS, 'dGS': dGS, 'hGS': hGS}

        else:
            out = {'F': F, 'dF': dF, 'hF': hF, 'FS': S, 'dFS': dS, 'hFS': hS}

        return out
