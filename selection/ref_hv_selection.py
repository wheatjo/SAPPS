import numpy as np
from pymoo.indicators.hv import HV
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
from utilsmodule.database import DataBase
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.core.duplicate import DefaultDuplicateElimination
from selection.hvi import BatchSelection
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival


def calc_ref_assign(F, ref_dirs):
    ref_dirs_norm = np.linalg.norm(ref_dirs, axis=1)
    d1s = F.dot(ref_dirs.T) / ref_dirs_norm
    F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
    d2s = np.sqrt(F_norm**2 - d1s**2)
    ref_index = np.argmin(d2s, axis=1)
    return ref_index


def HVI_calc(F_cand_fea, F_arch_fea, nadir_point):
    F_all = np.row_stack((F_cand_fea, F_arch_fea))
    z_ideal = np.min(F_all, axis=0)
    nadir = np.max(F_all, axis=0)
    F_cand_fea = (F_cand_fea - z_ideal) / (nadir - z_ideal)
    F_arch_fea = (F_arch_fea - z_ideal) / (nadir - z_ideal)
    hv = HV(ref_point=nadir_point)
    hvi_cand = []
    hv_value_arch = hv.do(F_arch_fea)
    for i in range(len(F_cand_fea)):
        hv_cand = hv.do(np.row_stack([F_arch_fea, F_cand_fea[i]]))
        hv_improvement = hv_cand - hv_value_arch
        hvi_cand.append(hv_improvement)

    return hvi_cand


def HVI_greed(F_cand_fea, F_arch_fea, nadir_point):
    hvi_greed_sort_index = []
    hvi_greed_sort_value = []
    F_all = np.column_stack((F_cand_fea, F_arch_fea))
    z_ideal = np.min(F_all, axis=0)
    nadir = np.max(F_all, axis=0)
    F_cand_fea = (F_cand_fea - z_ideal) / (nadir - z_ideal)
    F_arch_fea = (F_arch_fea - z_ideal) / (nadir - z_ideal)
    for i in range(len(F_cand_fea)):
        hvi_cand = HVI_calc(F_cand_fea, F_arch_fea, nadir_point)
        max_hvi_index = np.argmax(hvi_cand)
        hvi_greed_sort_index.append(max_hvi_index)
        hvi_greed_sort_value.append(hvi_cand[max_hvi_index])
        F_arch_fea = np.vstack([F_arch_fea, F_cand_fea[max_hvi_index]])

    return np.array(hvi_greed_sort_index), np.array(hvi_greed_sort_value)


class RefHvi(BatchSelection):

    def __init__(self, opt_problem):
        super(RefHvi, self).__init__()
        self.opt_problem = opt_problem

    def select(self, pop_arch: Population, pop_cand: Population, n_select: int, transformer: DataBase, **kwargs) \
            -> Population:

        ref_dir = UniformReferenceDirectionFactory(n_dim=transformer.n_obj, n_points=n_select).do()
        F_arch = pop_arch.get('F')
        F_cand = pop_cand.get('F')
        F_all = np.row_stack((F_arch, F_cand))
        z = np.min(F_all, axis=0)
        nadir = np.max(F_all, axis=0)
        ref_dir_norm = np.linalg.norm(ref_dir, axis=1)
        cand_d1s = (F_cand - z).dot(ref_dir.T) / ref_dir_norm
        F_cand_norm = np.linalg.norm(F_cand - z, axis=1).reshape(-1, 1)
        cand_d2s = np.sqrt(F_cand_norm**2 - cand_d1s**2)
        cand_ref_index = np.argmin(cand_d2s, axis=1)
        arch_fea = pop_arch[pop_arch.get('feas')]
        cand_fea = pop_cand[pop_cand.get('feas')]
        pop_sel = Population.new()
        arch_fea_F = arch_fea.get('F')
        if len(arch_fea_F) > 0:
            for i in range(n_select):
                hvi_cand_value = HVI_calc(arch_fea_F, cand_fea.get('F'), nadir)
                cand_fea.set(hvi=hvi_cand_value)
                cand_ref_i = pop_cand[cand_ref_index == i]
                cand_fea = cand_ref_i.get('feas')

                if len(cand_fea) == 0:
                    cand_cv = cand_ref_i.get('cv')
                    cand_sel = cand_ref_i[np.argmin(cand_cv)]
                else:
                    cand_sel_index = np.argmax(cand_fea.get('hvi'))
                    cand_sel = cand_fea[cand_sel_index]

                pop_sel = Population.merge(pop_sel, cand_sel)
                arch_fea_F = np.row_stack([arch_fea_F, cand_sel.get('F')])

        else:
            survival_sel = RankAndCrowdingSurvival()
            for i in range(n_select):
                cand_ref_i = pop_cand[cand_ref_index == i]
                cand_sel = survival_sel.do(self.opt_problem, cand_ref_i, n_survive=1)
                pop_sel = Population.merge(pop_sel, cand_sel)


        pop_sel_X_norm = pop_sel.get('X')
        pop_sel_X = transformer.x_transform.inverse_transform(pop_sel_X_norm)
        return Population.new(X=pop_sel_X)


class RefHviPush(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer: DataBase, **kwargs) -> Population:

        ref_dir = UniformReferenceDirectionFactory(n_dim=transformer.n_obj, n_points=n_select).do()
        F_cand = pop_cand.get('F')
        F_arch = pop_arch.get('F')
        F_all = np.row_stack((F_arch, F_cand))
        z = np.min(F_all, axis=0)
        nadir = np.max(F_all, axis=0)
        cand_ref_index = calc_ref_assign(F_cand - z, ref_dir)
        arch_ref_index = calc_ref_assign(F_arch - z, ref_dir)
        pop_sel = Population.new()

        cand_nd2arch = pop_cand[find_non_dominated(F_cand, F_arch)]
        cand_nd = cand_nd2arch[NonDominatedSorting().do(cand_nd2arch.get('F'), only_non_dominated_front=True)]
        hvi_nd_index, hvi_nd_value = HVI_greed(cand_nd.get('F'), F_arch, nadir)
        cand_nd_sort_hvi = cand_nd[hvi_nd_index]
        cand_nd_ref_index = calc_ref_assign(cand_nd_sort_hvi.get('F') - z, ref_dir)

        for ref_i in range(n_select):
            cand_sel = Population.new()
            pop_ref_i = pop_cand[cand_ref_index == ref_i]
            cand_nd_ref_i = cand_nd_sort_hvi[cand_nd_ref_index == ref_i]
            if len(cand_nd_ref_i) > 0:
                pop_sel = cand_nd_ref_i[np.argmax(hvi_nd_value[cand_nd_ref_index == ref_i])]

            elif len(pop_ref_i) > 0:
                pop_ref_nd = pop_ref_i[find_non_dominated(pop_ref_i.get('F'), F_arch)]

                if len(pop_ref_nd) > 0:
                    pop_hvi = HVI_calc(pop_ref_nd.get('F'), F_arch, nadir)
                    cand_sel = pop_ref_nd[np.argmax(pop_hvi)]

                else:
                    arch_ref_i = pop_arch[arch_ref_index == ref_i]
                    if len(arch_ref_i) > 0:
                        cand_local_nd = pop_ref_i[find_non_dominated(pop_ref_i.get('F'), arch_ref_i.get('F'))]

                        if len(cand_local_nd) > 0:
                            pop_hvi = HVI_calc(pop_ref_i.get('F') - z, arch_ref_i.get('F') - z, nadir)
                            cand_sel = pop_ref_i[np.argmax(pop_hvi)]

                        else:
                            ref_dir_i = ref_dir[ref_i]
                            pop_ref_i_tch = np.max((pop_ref_i.get('F') - z) * ref_dir_i, axis=1)\
                                            + 0.01 * np.sum((pop_ref_i.get('F') - z) * ref_dir_i, axis=1)

                            arch_ref_i_tch_best = np.min(np.max((arch_ref_i.get('F') - z) * ref_dir_i, axis=1)
                                                         + 0.01 * np.sum((arch_ref_i.get('F') - z) * ref_dir_i, axis=1))

                            tch_improve_temp = arch_ref_i_tch_best - pop_ref_i_tch
                            cand_sel = pop_ref_i[np.argmax(tch_improve_temp)]

            pop_sel = Population.merge(pop_sel, cand_sel)

        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)


class HviPush(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer: DataBase, **kwargs) -> Population:
        F_cand = pop_cand.get('F')
        F_arch = pop_arch.get('F')
        F_all = np.row_stack((F_arch, F_cand))
        z = np.min(F_all, axis=0)
        nadir = np.max(F_all, axis=0)
        pop_sel = Population.new()

        cand_nd2arch = pop_cand[find_non_dominated(F_cand, F_arch)]
        cand_nd = cand_nd2arch[NonDominatedSorting().do(cand_nd2arch.get('F'), only_non_dominated_front=True)]
        hvi_nd_index, hvi_nd_value = HVI_greed(cand_nd.get('F') - z, F_arch - z, nadir)
        cand_nd_sort_hvi = cand_nd[hvi_nd_index]
        if len(cand_nd) > n_select:
            pop_sel = Population.merge(pop_sel, cand_nd_sort_hvi[:n_select])
        else:
            pop_sel = Population.merge(pop_sel, cand_nd_sort_hvi)
        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)


class RefGreedHviPush(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        ref_dir = UniformReferenceDirectionFactory(n_dim=transformer.n_obj, n_points=n_select).do()
        F_cand = pop_cand.get('F')
        F_arch = pop_arch.get('F')
        F_all = np.row_stack((F_arch, F_cand))
        z = np.min(F_all, axis=0)
        nadir = np.max(F_all, axis=0)
        pop_sel = Population.new()
        cand_nd2arch = pop_cand[find_non_dominated(F_cand, F_arch)]
        cand_nd = cand_nd2arch[NonDominatedSorting().do(cand_nd2arch.get('F'), only_non_dominated_front=True)]
        hvi_nd_index, hvi_nd_value = HVI_greed(cand_nd.get('F'), F_arch, nadir)
        cand_nd_sort_hvi = cand_nd[hvi_nd_index]

        if len(cand_nd) > n_select:
            pop_sel = Population.merge(pop_sel, cand_nd_sort_hvi[:n_select])
            X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
            return Population.new(X=X_sel)
        else:
            pop_sel = Population.merge(pop_sel, cand_nd_sort_hvi)

        remain_select = n_select - len(pop_sel)
        remain_cand = DefaultDuplicateElimination().do(pop_cand, pop_sel)
        pop_arch = Population.merge(pop_arch, pop_sel)
        F_arch = pop_arch.get('F')
        arch_ref_index = calc_ref_assign(F_arch - z, ref_dir)
        cand_ref_index = calc_ref_assign(remain_cand.get('F') - z, ref_dir)

        if len(pop_sel) > 0:
            pop_sel_ref_index = calc_ref_assign(pop_sel.get('F') - z, ref_dir)
            ref_index = np.arange(n_select)
            ref_flag = []
            for ref_id in ref_index:
                if ref_id in pop_sel_ref_index:
                    ref_flag.append(False)
                else:
                    ref_flag.append(True)
            remain_ref_id = ref_index[np.array(ref_flag)]
            np.random.shuffle(remain_ref_id)

            for i in range(remain_select):
                ref_id_use = remain_ref_id[i]
                arch_ref_i = pop_arch[arch_ref_index == ref_id_use]
                cand_ref_i = remain_cand[cand_ref_index == ref_id_use]
                cand_sel = Population.new()
                if len(cand_ref_i) > 0:

                    if len(arch_ref_i) > 0:
                        cand_local_nd = cand_ref_i[find_non_dominated(cand_ref_i.get('F'), arch_ref_i.get('F'))]

                        if len(cand_local_nd) > 0:
                            hvi_cand_local = HVI_calc(cand_local_nd.get('F'), arch_ref_i.get('F'), nadir)
                            cand_sel = cand_local_nd[np.argmax(hvi_cand_local)]
                        else:
                            cand_ref_i_x = cand_ref_i.get('X')
                            surr_pro = kwargs['surr_pro']
                            pred_res = surr_pro.evaluate(cand_ref_i_x, std=True, calc_G=False)
                            cand_ref_i_uncertainty = pred_res['FS']
                            cand_sel = cand_ref_i[np.argmax(np.max(cand_ref_i_uncertainty, axis=1))]

                pop_sel = Population.merge(pop_sel, cand_sel)

        if len(pop_sel) < n_select:
            last_select_num = n_select - len(pop_sel)
            pop_last_cand = DefaultDuplicateElimination().do(pop_cand, pop_sel)
            surr_pro = kwargs['surr_pro']
            pop_last_cand_x = pop_last_cand.get('X')
            pred_res = surr_pro.evaluate(pop_last_cand_x, std=True, calc_G=False)
            uncertainly = np.max(pred_res['FS'], axis=1)
            sel_index = np.argsort(-1.0 * uncertainly)[:last_select_num]
            cand_sel = pop_last_cand[sel_index]
            pop_sel = Population.merge(pop_sel, cand_sel)

        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)

