import numpy as np
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
from pymoo.core.survival import split_by_feasibility
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.core.duplicate import DefaultDuplicateElimination
from selection.hvi import BatchSelection
from selection.ref_hv_selection import HVI_greed, HVI_calc, calc_ref_assign
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from selection.ModfiySPEA2 import ModifySPEA2Survival


class CDPSelection(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        pop_cand = DefaultDuplicateElimination().do(pop_cand, pop_arch)
        pop_sel = RankAndCrowdingSurvival().do(kwargs['surr_opt_pro'], pop_cand, n_survive=n_select)
        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)


class RefCVHvi(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        ref_dir = UniformReferenceDirectionFactory(n_dim=transformer.n_obj, n_points=n_select).do()
        feasible, infeasible = split_by_feasibility(pop_arch)
        pop_arch_feasible = pop_arch[feasible]
        # pop_arch_infeasible = pop_arch[infeasible]
        # think if pop_arch have no feasible ind
        pop_cand = DefaultDuplicateElimination().do(pop_cand, pop_arch)
        feasible, infeasible = split_by_feasibility(pop_cand)
        pop_cand_feasible = pop_cand[feasible]
        F_arch_fea = pop_arch_feasible.get('F')
        F_cand_fea = pop_cand_feasible.get('F')
        F_arch = pop_arch.get('F')
        F_cand = pop_cand.get('F')

        if len(F_cand_fea) > 0 and len(F_arch_fea) > 0:
            F_all_fea = np.vstack([F_cand_fea, F_arch_fea])
        elif len(F_cand_fea) > 0:
            F_all_fea = F_cand_fea
        elif len(F_arch_fea) > 0:
            F_all_fea = F_arch_fea
        else:
            F_all_fea = np.array([])

        if len(F_cand_fea) > 0 or len(F_arch_fea) > 0:
            nadir_fea = np.max(F_all_fea, axis=0)
        else:
            nadir_fea = np.array([])

        F_all = np.vstack([F_arch, F_cand])
        z = np.min(F_all, axis=0)
        arch_ref_index = calc_ref_assign(F_arch - z, ref_dir)
        cand_ref_index = calc_ref_assign(F_cand - z, ref_dir)

        cand_nd_sort_hvi = Population.new()
        cand_hvi_ref_index = np.array([])
        hvi_nd_value = np.array([])
        pop_sel = Population.new()

        if len(pop_arch_feasible) > 0 and len(pop_cand_feasible) > 0:
            cand_nd2arch = pop_cand[find_non_dominated(pop_cand_feasible.get('F'), pop_arch_feasible.get('F'))]
            if len(cand_nd2arch) < 0:
                cand_nd = cand_nd2arch[NonDominatedSorting().do(cand_nd2arch.get('F'), only_non_dominated_front=True)]
            else:
                cand_nd = Population.new()
            if len(cand_nd) > 0:
                cand_nd = RankAndCrowdingSurvival().do(kwargs['surr_opt_pro'], cand_nd, n_survive=100)
                hvi_nd_index, hvi_nd_value = HVI_greed(cand_nd.get('F'), pop_arch_feasible.get('F'), nadir_fea)
                cand_nd_sort_hvi = cand_nd[hvi_nd_index]


        if len(cand_nd_sort_hvi) > 0:
            cand_hvi_ref_index = calc_ref_assign(cand_nd_sort_hvi.get('F') - z, ref_dir)

        ref_index = np.arange(n_select)
        np.random.shuffle(ref_index)

        for ref_i in ref_index:

            if ref_i in cand_hvi_ref_index:
                ref_hvi_ind = cand_nd_sort_hvi[cand_hvi_ref_index == ref_i]
                ref_i_hv_value = hvi_nd_value[cand_hvi_ref_index == ref_i]
                cand_sel = ref_hvi_ind[np.argmax(ref_i_hv_value)]
            else:
                cand_ref_i = pop_cand[cand_ref_index == ref_i]
                arch_ref_i = pop_arch[arch_ref_index == ref_i]

                if len(cand_ref_i) > 0:

                    cand_ref_i_fea = cand_ref_i[cand_ref_i.get('feas')]
                    if len(arch_ref_i) > 0:
                        arch_ref_i_fea = arch_ref_i[arch_ref_i.get('feas')]
                    else:
                        arch_ref_i_fea = np.array([])

                    if len(cand_ref_i_fea) > 0:

                        if len(arch_ref_i_fea) > 0:
                            local_nadir_point = np.max(np.vstack([cand_ref_i_fea.get('F'), arch_ref_i_fea.get('F')]),
                                                       axis=0)
                            hvi_cand_local = HVI_calc(cand_ref_i_fea.get('F'), arch_ref_i_fea.get('F'),
                                                      local_nadir_point)

                            max_hvi_local = np.max(hvi_cand_local)
                            if max_hvi_local > 0:
                                cand_sel = cand_ref_i_fea[np.argmax(max_hvi_local)]
                            else:
                                surr_pro = kwargs['surr_pro']
                                cand_ref_i_fea_x = cand_ref_i_fea.get('X')
                                pred_res = surr_pro.evaluate(cand_ref_i_fea_x, std=True, calc_G=False)
                                cand_ref_i_uncertainty = pred_res['FS']
                                cand_sel = cand_ref_i[np.argmax(np.max(cand_ref_i_uncertainty, axis=1))]

                        else:
                            ref_dir_i = ref_dir[ref_i]
                            F_cand_ref_i = cand_ref_i_fea.get('F')
                            tch_value = np.max(F_cand_ref_i * ref_dir_i, axis=1)
                            cand_sel = cand_ref_i[np.argmin(tch_value)]

                    else:
                        ref_dir_i = ref_dir[ref_i]
                        cv = cand_ref_i.get('cv')
                        F = cand_ref_i.get('F')
                        tch_cv_value = np.max(F * ref_dir_i, axis=1) + cv
                        cand_sel = cand_ref_i[np.argmin(tch_cv_value)]

                else:
                    ref_dir_i = ref_dir[ref_i][np.newaxis, :]
                    cand_use_ref_i = DefaultDuplicateElimination().do(pop_cand, pop_sel)
                    if cand_use_ref_i is None:
                        F_cand_use_ref_i = cand_use_ref_i.get('F') - z
                        ref_dir_i_norm = np.linalg.norm(ref_dir_i, axis=1)
                        d1s = F_cand_use_ref_i.dot(ref_dir_i.T) / ref_dir_i_norm
                        F_cand_use_ref_i_norm = np.linalg.norm(F_cand_use_ref_i, axis=1)
                        d2s = np.sqrt(F_cand_use_ref_i_norm**2 - d1s[:, 0]**2)
                        cand_sel = cand_use_ref_i[np.argmin(d2s)]
                    else:
                        cand_sel = Population.new()

            pop_sel = Population.merge(pop_sel, cand_sel)

        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)


class RefCVHviSel(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        ref_dir = UniformReferenceDirectionFactory(n_dim=transformer.n_obj, n_points=n_select).do()
        feasible, infeasible = split_by_feasibility(pop_arch)
        # pop_arch_feasible = pop_arch[feasible]
        # pop_arch_infeasible = pop_arch[infeasible]
        # think if pop_arch have no feasible ind
        pop_cand = DefaultDuplicateElimination().do(pop_cand, pop_arch)
        # feasible, infeasible = split_by_feasibility(pop_cand)
        # pop_cand_feasible = pop_cand[feasible]
        # F_arch_fea = pop_arch_feasible.get('F')
        # F_cand_fea = pop_cand_feasible.get('F')
        F_arch = pop_arch.get('F')
        F_cand = pop_cand.get('F')
        F_all = np.vstack([F_arch, F_cand])
        z = np.min(F_all, axis=0)
        arch_ref_index = calc_ref_assign(F_arch - z, ref_dir)
        cand_ref_index = calc_ref_assign(F_cand - z, ref_dir)

        ref_index = np.arange(n_select)
        np.random.shuffle(ref_index)
        pop_sel = Population.new()

        for ref_i in ref_index:
            cand_ref_i = pop_cand[cand_ref_index == ref_i]
            arch_ref_i = pop_arch[arch_ref_index == ref_i]
            cand_sel = Population.new()
            if len(arch_ref_i) > 0 and len(cand_ref_i) > 0:
                arch_ref_i_fea = arch_ref_i[arch_ref_i.get('feas')]
                cand_ref_i_fea = cand_ref_i[cand_ref_i.get('feas')]

                if len(arch_ref_i_fea) > 0:

                    if len(cand_ref_i_fea) > 0:
                        local_nadir_point = np.max(np.vstack([cand_ref_i_fea.get('F'), arch_ref_i_fea.get('F')]),
                                                   axis=0)
                        hvi_cand_local = HVI_calc(cand_ref_i_fea.get('F'), arch_ref_i_fea.get('F'),
                                                  local_nadir_point)
                        max_hvi_local = np.max(hvi_cand_local)
                        if max_hvi_local > 0:
                            cand_sel = cand_ref_i_fea[np.argmax(max_hvi_local)]
                        else:
                            # think again, need feasible, use GS?
                            surr_pro = kwargs['surr_pro']
                            cand_ref_i_fea_x = cand_ref_i_fea.get('X')
                            pred_res = surr_pro.evaluate(cand_ref_i_fea_x, std=True, calc_G=False)
                            cand_ref_i_uncertainty = pred_res['FS']
                            cand_sel = cand_ref_i[np.argmax(np.max(cand_ref_i_uncertainty, axis=1))]

                    else:
                        cand_ref_i_fea_index = cand_ref_i.get('feas')
                        cand_ref_i_fea = cand_ref_i[cand_ref_i_fea_index]
                        cand_ref_i_infea = cand_ref_i[~cand_ref_i_fea_index]

                        if len(cand_ref_i_fea) > 0:
                            G_fea = cand_ref_i_fea.get('G')
                            fea_cv_value = np.sum(G_fea, axis=1)
                            cand_ref_i_fea.set(G_value=fea_cv_value)

                        if len(cand_ref_i_infea) > 0:
                            G_infea = cand_ref_i_infea.get('cv')
                            cand_ref_i_infea.set(G_value=G_infea)

                        cand_ref_i_G_value = cand_ref_i.get('G_value')
                        cand_sel = cand_ref_i[np.argmin(cand_ref_i_G_value)]

                else:
                    cand_ref_i_fea_index = cand_ref_i.get('feas')
                    cand_ref_i_fea = cand_ref_i[cand_ref_i_fea_index]
                    cand_ref_i_infea = cand_ref_i[~cand_ref_i_fea_index]

                    if len(cand_ref_i_fea) > 0:
                        G_fea = cand_ref_i_fea.get('G')
                        fea_cv_value = np.sum(G_fea, axis=1)
                        cand_ref_i_fea.set(G_value=fea_cv_value)

                    if len(cand_ref_i_infea) > 0:
                        G_infea = cand_ref_i_infea.get('cv')
                        cand_ref_i_infea.set(G_value=G_infea)

                    cand_ref_i_G_value = cand_ref_i.get('G_value')
                    cand_sel = cand_ref_i[np.argmin(cand_ref_i_G_value)]

            elif len(arch_ref_i) == 0 and len(cand_ref_i) > 0:
                cand_ref_i_fea_index = cand_ref_i.get('feas')
                cand_ref_i_fea = cand_ref_i[cand_ref_i_fea_index]
                cand_ref_i_infea = cand_ref_i[~cand_ref_i_fea_index]

                if len(cand_ref_i_fea) > 0:
                    G_fea = cand_ref_i_fea.get('G')
                    fea_cv_value = np.sum(G_fea, axis=1)
                    cand_ref_i_fea.set(G_value=fea_cv_value)

                if len(cand_ref_i_infea) > 0:
                    G_infea = cand_ref_i_infea.get('cv')
                    cand_ref_i_infea.set(G_value=G_infea)

                cand_ref_i_G_value = cand_ref_i.get('G_value')
                cand_sel = cand_ref_i[np.argmin(cand_ref_i_G_value)]

            else:
                ref_dir_i = ref_dir[ref_i]
                cv = pop_cand.get('cv')
                F = pop_cand.get('F')
                tch_cv_value = np.max(F * ref_dir_i, axis=1) + 10.0 * cv
                cand_sel = pop_cand[np.argmin(tch_cv_value)]


            pop_sel = Population.merge(pop_sel, cand_sel)

        X_sel = pop_sel.get('X')
        return Population.new(X=X_sel)



"""
need to think according to arch !!!
~have cand:
    1. have arch:
        - have feasible cand
        - not have feasi
    2. not have arch: 

have none cand:
    choose minimum distance ~
"""


class RefCVHviSelImprove(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        ref_dir = kwargs['ref_dirs']
        # pop_arch_infeasible = pop_arch[infeasible]
        # think if pop_arch have no feasible ind
        pop_cand = DefaultDuplicateElimination().do(pop_cand, pop_arch)
        F_arch = pop_arch.get('F')
        F_cand = pop_cand.get('F')
        F_all = np.vstack([F_arch, F_cand])
        z = np.min(F_all, axis=0)
        nadir = np.max(F_all, axis=0)
        arch_ref_index = calc_ref_assign((F_arch - z) / (nadir - z), ref_dir)
        cand_ref_index = calc_ref_assign((F_cand - z) / (nadir - z), ref_dir)

        ref_index = np.arange(n_select)
        np.random.shuffle(ref_index)
        pop_sel = Population.new()

        for ref_i in ref_index:
            cand_ref_i = pop_cand[cand_ref_index == ref_i]
            arch_ref_i = pop_arch[arch_ref_index == ref_i]

            if len(arch_ref_i) > 0 and len(cand_ref_i) > 0:
                arch_ref_i_fea = arch_ref_i[arch_ref_i.get('feas')]
                cand_ref_i_fea = cand_ref_i[cand_ref_i.get('feas')]

                if len(arch_ref_i_fea) > 0:
                    # have feasible arch and have feasible cand
                    if len(cand_ref_i_fea) > 0:
                        local_nadir_point = np.max(np.vstack([cand_ref_i_fea.get('F'), arch_ref_i_fea.get('F')]),
                                                   axis=0)
                        hvi_cand_local = HVI_calc(cand_ref_i_fea.get('F'), arch_ref_i_fea.get('F'),
                                                  local_nadir_point)
                        max_hvi_local = np.max(hvi_cand_local)
                        if max_hvi_local > 0:
                            cand_sel = cand_ref_i_fea[np.argmax(max_hvi_local)]
                        else:
                            # think again, need feasible, use GS?
                            surr_pro = kwargs['surr_pro']
                            cand_ref_i_fea_x = cand_ref_i_fea.get('X')
                            pred_res = surr_pro.evaluate(cand_ref_i_fea_x, std=True, calc_G=False)
                            cand_ref_i_uncertainty = pred_res['FS']
                            cand_sel = cand_ref_i[np.argmax(np.mean(cand_ref_i_uncertainty, axis=1))]

                    else:
                        cand_sel = select_by_feasible_indicator(cand_ref_i)

                else:
                    cand_sel = select_by_feasible_indicator(cand_ref_i)

            elif len(arch_ref_i) == 0 and len(cand_ref_i) > 0:
                cand_sel = select_by_feasible_indicator(cand_ref_i)

            else:
                ref_dir_i = ref_dir[ref_i][None, :]
                ref_dirs_norm = np.linalg.norm(ref_dir_i, axis=1)
                F = pop_cand.get('F')
                F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
                d1s = F.dot(ref_dir_i.T) / ref_dirs_norm
                d2s = np.sqrt(F_norm**2 - d1s**2)
                cand_sel = pop_cand[np.argmin(d2s)]

            pop_sel = Population.merge(pop_sel, cand_sel)

        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)


class RefSPEA2Sel(BatchSelection):

    def select(self, pop_arch, pop_cand, n_select, transformer, **kwargs):
        spea2_select = ModifySPEA2Survival()
        ref_dir_origin = kwargs['ref_dirs']
        # ref_dir = kwargs['ref_dirs']
        # pop_arch_infeasible = pop_arch[infeasible]
        # think if pop_arch have no feasible ind
        pop_cand_dup = DefaultDuplicateElimination().do(pop_cand, pop_arch)
        if len(pop_cand_dup) > 0:
            pop_cand = pop_cand_dup

        F_arch = pop_arch.get('F')
        F_cand = pop_cand.get('F')
        F_all = np.vstack([F_arch, F_cand])
        z = np.min(F_all, axis=0)
        nadir = np.max(F_all, axis=0)
        try:
            a = (ref_dir_origin * (nadir - z)) / np.linalg.norm(ref_dir_origin * (nadir - z), axis=1)[:, None]
        except ValueError:
            print(f"ref_dir_origin: {ref_dir_origin} | nadir: {nadir} | {z}")
        ref_dir = (ref_dir_origin * (nadir - z)) / np.linalg.norm(ref_dir_origin * (nadir - z), axis=1)[:, None]
        arch_ref_index = calc_ref_assign(F_arch, ref_dir)
        cand_ref_index = calc_ref_assign(F_cand, ref_dir)

        ref_index = np.arange(n_select)
        np.random.shuffle(ref_index)
        pop_sel = Population.new()
        arch_feas = pop_arch[pop_arch.get('feas')]
        if len(arch_feas) > 0:
            arch_nd = arch_feas[NonDominatedSorting().do(arch_feas.get('F'))[0]]
        else:
            arch_nd = pop_arch[np.argmin(pop_arch.get('CV')[:, 0])]

        pop_cand = spea2_select.do(kwargs['problem'], pop_cand, n_survive=len(pop_cand),
                                   pop_nd_arch=arch_nd)

        for ref_i in ref_index:
            cand_ref_i = pop_cand[cand_ref_index == ref_i]
            arch_ref_i = pop_arch[arch_ref_index == ref_i]

            if len(arch_ref_i) > 0 and len(cand_ref_i) > 0:
                arch_ref_i_fea = arch_ref_i[arch_ref_i.get('feas')]
                cand_ref_i_fea = cand_ref_i[cand_ref_i.get('feas')]

                if len(arch_ref_i_fea) > 0:
                    # have feasible arch and have feasible cand
                    if len(cand_ref_i_fea) > 0:
                        spea2_value = cand_ref_i_fea.get("SPEA_F")
                        cand_sel = cand_ref_i_fea[np.argmin(spea2_value)]
                    else:
                        cand_sel = select_by_feasible_indicator(cand_ref_i)

                else:
                    cand_sel = select_by_feasible_indicator(cand_ref_i)

            elif len(arch_ref_i) == 0 and len(cand_ref_i) > 0:
                cand_sel = select_by_feasible_indicator(cand_ref_i)

            else:
                ref_dir_i = ref_dir[ref_i][None, :]
                ref_dirs_norm = np.linalg.norm(ref_dir_i, axis=1)
                F = pop_cand.get('F')
                F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
                d1s = F.dot(ref_dir_i.T) / ref_dirs_norm
                d2s = np.sqrt(F_norm**2 - d1s**2)
                cand_sel = pop_cand[np.argmin(d2s)]

            pop_sel = Population.merge(pop_sel, cand_sel)

        X_sel = transformer.x_transform.inverse_transform(pop_sel.get('X'))
        return Population.new(X=X_sel)


def select_by_feasible_indicator(cand_ref_i_temp):

    cand_ref_i_fea_index = cand_ref_i_temp.get('feas')
    cand_ref_i_fea = cand_ref_i_temp[cand_ref_i_fea_index]
    cand_ref_i_infea = cand_ref_i_temp[~cand_ref_i_fea_index]

    if len(cand_ref_i_fea) > 0:
        G_fea = cand_ref_i_fea.get('G')
        fea_cv_value = np.sum(G_fea, axis=1)
        cand_ref_i_fea.set(G_value=fea_cv_value)

    if len(cand_ref_i_infea) > 0:
        G_infea = cand_ref_i_infea.get('cv')
        cand_ref_i_infea.set(G_value=G_infea)

    cand_ref_i_G_value = cand_ref_i_temp.get('G_value')
    cand_sel = cand_ref_i_temp[np.argmin(cand_ref_i_G_value)]
    return cand_sel




# if __name__ == '__main__':
#     ref_dir = np.random.rand(3, 2)
#     ref_i = 0
#     ref_dir_i = ref_dir[ref_i][None, :]
#     ref_dirs_norm = np.linalg.norm(ref_dir_i, axis=1)
#     F = np.random.rand(2, 2)
#     F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
#     d1s = F.dot(ref_dir_i.T) / ref_dirs_norm
#     d2s = np.sqrt(F_norm ** 2 - d1s ** 2)
#     print(d2s)
