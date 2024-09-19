import numpy as np
from pymoo.indicators.hv import HV
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utilsmodule.database import DataBase


class BatchSelection(object):

    def __init__(self):
        pass

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        pass


class HVI(BatchSelection):

    def select(self, pop_arch_norm: Population, pop_cand: Population, n_select: int,
               transformer: DataBase, **kwargs) -> Population:

        F_arch = pop_arch_norm.get('F')
        F_cand = pop_cand.get('F')
        nds = NonDominatedSorting()
        F_arch_nds = F_arch[nds.do(F_arch, only_non_dominated_front=True)]
        F_cand_nd_index = nds.do(F_cand, only_non_dominated_front=True)
        ref_point = np.max(np.vstack([F_arch_nds, F_cand]), axis=0) * 1.1
        select_mask = np.full(len(pop_cand), False)
        surr_pro = kwargs['surr_pro']
        FS = surr_pro.evaluate(pop_cand.get('X'), std=True, calc_G=False)['FS']
        F_cand_id_list = np.arange(len(F_cand))

        for i in range(n_select):
            hv = HV(ref_point=ref_point)
            max_hvi = 0.
            best_sub_set = None
            current_hv = hv.do(F_arch_nds)
            for nd_index in F_cand_nd_index:
                F_j_nd = F_cand[nd_index]
                temp = np.vstack([F_arch_nds, F_j_nd])
                hv_value_j = hv.do(temp) - current_hv
                if hv_value_j > max_hvi:
                    max_hvi = hv_value_j
                    best_sub_set = nd_index

            if best_sub_set is not None:
                F_arch_nds = np.vstack([F_arch_nds, F_cand[best_sub_set]])
                select_mask[best_sub_set] = True

            else:
                remain_FS = FS[~select_mask]
                max_FS_index = np.argmax(np.mean(remain_FS, axis=1))
                best_sub_set = F_cand_id_list[~select_mask][max_FS_index]
                select_mask[best_sub_set] = True

        pop_sel = pop_cand[select_mask]
        pop_sel_X = pop_sel.get('X')
        return Population.new(X=pop_sel_X)
