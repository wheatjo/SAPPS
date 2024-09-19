import numpy as np
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.population import Population
from pymoo.core.duplicate import DefaultDuplicateElimination
from sklearn.gaussian_process import GaussianProcessClassifier
from scipy.linalg import solve
from pymoo.util.misc import at_least_2d_array
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import solve_triangular, cholesky
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival


def assisted_pop_select(assist_pop: Population, num_select: int, test_pro) -> Population:

    assist_pop_inverse = Population.new(X=assist_pop.get('X'), F=-1.0 * assist_pop.get('F'),
                                        origin_F=assist_pop.get('F'), origin_G=assist_pop.get('G'))
    assist_sel = RankAndCrowdingSurvival().do(test_pro, assist_pop_inverse, n_survive=num_select)
    pop_assist_new = Population.new(X=assist_sel.get('X'), F=assist_sel.get('origin_F'), G=assist_sel.get('origin_G'))
    return pop_assist_new


class BiCoSelection(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        F = pop.get('F')
        CV = pop.get('CV')
        F_CV = np.column_stack([F, CV])
        fcv_nd_pop_index = NonDominatedSorting().do(F_CV, only_non_dominated_front=True)
        fcv_nd_pop = pop[fcv_nd_pop_index]
        assist_pop = fcv_nd_pop[~fcv_nd_pop.get('feas')]

        num_assist_pop = kwargs['num_assist_pop']
        num_main_pop = kwargs['num_main_pop']
        if len(assist_pop) > 0:
            ass_pop_sel = assisted_pop_select(assist_pop, num_assist_pop, problem)
        else:
            ass_pop_sel = Population.new()

        pop_remain = DefaultDuplicateElimination().do(pop, ass_pop_sel)
        if len(ass_pop_sel) > 0:
            main_pop_sel = RankAndCrowdingSurvival().do(problem, pop_remain,
                                                        n_survive=num_main_pop + num_assist_pop - len(ass_pop_sel))
        else:
            main_pop_sel = RankAndCrowdingSurvival().do(problem, pop_remain, n_survive=num_main_pop+num_assist_pop)
        return Population.merge(main_pop_sel, ass_pop_sel)


def active_learning_selection(pre_sel_pop: Population, pop: Population, num_select: int) -> Population:

    remain_pop = DefaultDuplicateElimination().do(pop, pre_sel_pop)
    if len(remain_pop) > 0:
        fea_label = np.zeros(len(pre_sel_pop))
        fea_label[pre_sel_pop.get('feas')] = 1
        pop_x = pre_sel_pop.get('X')
        kernel = ConstantKernel() * RBF(1.0) + ConstantKernel()
        gpc = GaussianProcessClassifier(kernel)
        gpc.fit(pop_x, fea_label)

        remain_x = remain_pop.get('X')
        remain_x = at_least_2d_array(remain_x)
        remain_pop_prob = gpc.predict_proba(remain_x)

        K_star = gpc.base_estimator_.kernel_(pop_x, remain_x)  # K_star =k(x_star)
        f_star = K_star.T.dot(gpc.base_estimator_.y_train_ - gpc.base_estimator_.pi_)  # Line 4
        v = solve(gpc.base_estimator_.L_, gpc.base_estimator_.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = gpc.base_estimator_.kernel_.diag(remain_x) - np.einsum("ij,ij->j", v, v)

        H = -1.0 * remain_pop_prob[:, 1] * np.log(remain_pop_prob[:, 1]) - \
            (1 - remain_pop_prob[:, 1]) * np.log(1 - remain_pop_prob[:, 1])

        C = np.sqrt(np.pi * np.log(2.0) / 2.0)
        EC = C / np.sqrt(var_f_star + C ** 2) * np.exp(-1.0 * f_star ** 2 / (2 * (var_f_star + C ** 2)))

        Sq = H - EC
        sel_Sq_index = np.argsort(-1.0 * Sq)
        active_sel_pop = remain_pop[sel_Sq_index[:num_select]]

    else:
        active_sel_pop = Population.new()
    return active_sel_pop


def active_learning_single_constraint_sel(pre_sel_pop: Population, pop: Population, num_select: int, con_index: int) -> Population:

    remain_pop = DefaultDuplicateElimination().do(pop, pre_sel_pop)
    if len(remain_pop) > 0:
        fea_label = np.zeros(len(pre_sel_pop))
        fea_label[pre_sel_pop.get('G')[:, con_index] <= 0] = 1
        pop_x = pre_sel_pop.get('X')
        kernel = ConstantKernel() * RBF(1.0) + ConstantKernel()
        gpc = GaussianProcessClassifier(kernel)
        gpc.fit(pop_x, fea_label)

        remain_x = remain_pop.get('X')
        remain_x = at_least_2d_array(remain_x)
        # print(remain_x)
        remain_pop_prob = gpc.predict_proba(remain_x)

        K_star = gpc.base_estimator_.kernel_(pop_x, remain_x)  # K_star =k(x_star)
        f_star = K_star.T.dot(gpc.base_estimator_.y_train_ - gpc.base_estimator_.pi_)  # Line 4
        v = solve(gpc.base_estimator_.L_, gpc.base_estimator_.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        var_f_star = gpc.base_estimator_.kernel_.diag(remain_x) - np.einsum("ij,ij->j", v, v)

        H = -1.0 * remain_pop_prob[:, 1] * np.log(remain_pop_prob[:, 1]) - \
            (1 - remain_pop_prob[:, 1]) * np.log(1 - remain_pop_prob[:, 1])

        C = np.sqrt(np.pi * np.log(2.0) / 2.0)
        EC = C / np.sqrt(var_f_star + C ** 2) * np.exp(-1.0 * f_star ** 2 / (2 * (var_f_star + C ** 2)))

        Sq = H - EC

        sel_Sq_index = np.argsort(-1.0 * Sq)
        active_sel_pop = remain_pop[sel_Sq_index[:num_select]]

    else:
        active_sel_pop = Population.new()
    return active_sel_pop


def merge_bico_active_sel(problem, pop, n_bico_base_select, n_bico_active_pre_select, n_active_sel, n_select):

    bico_base_select = BiCoSelection().do(problem, pop, n_survive=n_bico_base_select,
                                            num_assist_pop=int(n_bico_base_select/2),
                                            num_main_pop=int(n_bico_base_select/2))

    bico_pop_feas = bico_base_select[bico_base_select.get('feas')]
    bico_pop_infeas = bico_base_select[~bico_base_select.get('feas')]

    if len(bico_pop_infeas) > 0 and len(bico_pop_feas) > 0:
        al_pop_set = DefaultDuplicateElimination().do(pop, bico_base_select)
        al_pre_pop = BiCoSelection().do(problem, al_pop_set, n_survive=n_bico_active_pre_select,
                                        num_assist_pop=int(n_bico_active_pre_select/2),
                                        num_main_pop=int(n_bico_active_pre_select/2))

        al_pop = active_learning_selection(bico_base_select, al_pre_pop, n_active_sel)
        all_fit_pop = Population.merge(bico_base_select, al_pop)
    else:
        all_fit_pop = RankAndCrowdingSurvival().do(problem, pop, n_survive=n_select)

    return all_fit_pop


def get_single_constraint_active_sel(pop_pre, pop_bico_dataset, con_index, n_active_sel):
    con_i_value = pop_pre.get('G')[:, con_index]
    num_con_i_fea = np.sum(con_i_value <= 0)
    num_con_i_infea = np.sum(con_i_value > 0)
    if num_con_i_fea > 0 and num_con_i_infea > 0:
        al_pop = active_learning_single_constraint_sel(pop_pre, pop_bico_dataset, n_active_sel, con_index)
        fit_pop = Population.merge(pop_pre, al_pop)
    else:
        al_pre_pop_i_con_value = pop_bico_dataset.get('G')[:, con_index]
        cv_sort_pop = pop_bico_dataset[al_pre_pop_i_con_value.argsort()[:n_active_sel]]
        fit_pop = Population.merge(pop_pre, cv_sort_pop)

    fit_pop = DefaultDuplicateElimination().do(fit_pop)
    return fit_pop


def get_batch_bald_sel(problem, pop_pre, pop_bico_dataset, i_index, n_active_sel, flag):
    main_kernel = Matern()
    gpr = GaussianProcessRegressor(kernel=main_kernel)

    if flag == 'obj':
        X = pop_pre.get('X')
        f = pop_pre.get('F')[:, i_index]
    else:
        X = pop_pre.get('X')
        f = pop_pre.get('G')[:, i_index]

    gpr.fit(X, f)
    pop_sel = Population.empty()
    X_condition = pop_pre.get('X')
    X_predict = pop_bico_dataset.get('X')

    if len(X_predict) <= n_active_sel:
        return pop_pre

    for i in range(n_active_sel):

        Kxx = gpr.kernel_(X_condition, X_condition)
        Kxx = Kxx + 1e-8 * np.eye(Kxx.shape[0])
        Kzx = gpr.kernel_(X_predict, X_condition)
        L = cholesky(Kxx, lower=True, check_finite=False)
        L_inv = solve_triangular(L, np.eye(L.shape[0]))
        Kxx_inv = L_inv.dot(L_inv.T)
        y_var = gpr.kernel_.diag(X_predict)
        y_var -= np.einsum("ij,ij->i", np.dot(Kzx, Kxx_inv), Kzx)
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            y_var[y_var_negative] = 0.0
        sel_index = np.argmax(y_var)
        pop_sel = Population.merge(pop_sel, pop_bico_dataset[sel_index])
        pop_bico_dataset = np.delete(pop_bico_dataset, sel_index)
        X_condition = np.vstack((X_condition, X_predict[sel_index]))
        X_predict = np.delete(X_predict, sel_index, axis=0)

    return pop_sel


def bico_active_object_constraint_select(problem, pop, objective_select, n_bico_base_select, n_bico_active_pre_select,
                                         n_active_sel):
    """
    objective_selection_pop -> (X, F)
    pop_G_data -> (X, G)
    """
    if len(pop) > objective_select:
        bico = BiCoSelection()
        objective_selection_pop = bico.do(problem, pop, n_survive=objective_select, num_assist_pop=int(objective_select/2),
                                          num_main_pop=int(objective_select/2))

        constraint_base_select = BiCoSelection().do(problem, pop, n_survive=n_bico_base_select,
                                                num_assist_pop=int(n_bico_base_select/2),
                                                num_main_pop=int(n_bico_base_select/2))

        constraint_active_pre_select = BiCoSelection().do(problem, pop, n_survive=n_bico_base_select,
                                                num_assist_pop=int(n_bico_active_pre_select/2),
                                                num_main_pop=int(n_bico_active_pre_select/2))

        constraint_active_pre_select = DefaultDuplicateElimination().do(constraint_active_pre_select,
                                                                        constraint_base_select)
        pop_G_data = []
        for i in range(problem.n_ieq_constr):
            con_i_act_pop = get_single_constraint_active_sel(constraint_base_select, constraint_active_pre_select, i,
                                                             n_active_sel)
            pop_G_data.append(con_i_act_pop)
    else:
        objective_selection_pop = pop
        pop_G_data = []
        for i in range(problem.n_ieq_constr):
            pop_G_data.append(pop)

    return objective_selection_pop, pop_G_data


def batch_active_object_constraint_select(problem, archive: Population, num_base_pop: int, num_pre_select: int,
                                          num_batch_blad_select: int) -> Population:
    """
    archive : true evaluation population
    """
    if len(archive) > num_base_pop + num_batch_blad_select:
        bico = BiCoSelection()
        base_pop = bico.do(problem, archive, n_survive=num_base_pop, num_assist_pop=int(num_base_pop/2),
                           num_main_pop=int(num_base_pop/2))
        num_pre_select = num_pre_select + num_base_pop
        bald_select_pool = bico.do(problem, archive, n_survive=num_base_pop, num_assist_pop=int(num_pre_select/2),
                                   num_main_pop=int(num_pre_select/2))
        bald_select_pool = DefaultDuplicateElimination().do(bald_select_pool, base_pop)
        pop_train = base_pop
        # if len(bald_select_pool) > num_batch_blad_select:
        print(f"batch selection: num_pool:{len(bald_select_pool)}, num_bald_select: {num_batch_blad_select}")
        for i in range(problem.n_obj):
            bald_sel_pop_i = get_batch_bald_sel(problem, base_pop, bald_select_pool, i, num_batch_blad_select, 'obj')
            pop_train = Population.merge(pop_train, bald_sel_pop_i)

        for i in range(problem.n_ieq_constr):
            bald_sel_pop_i = get_batch_bald_sel(problem, base_pop, bald_select_pool, i, num_batch_blad_select, 'con')
            pop_train = Population.merge(pop_train, bald_sel_pop_i)

        pop_train_X = pop_train.get('X')
        _,  pop_train_unique_index = np.unique(pop_train_X, axis=0, return_index=True)
        pop_train_unique = pop_train[pop_train_unique_index]

    else:
        pop_train_unique = archive

    return pop_train_unique

