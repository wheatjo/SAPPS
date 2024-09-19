import numpy as np
from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization
from pymoo.core.survival import Survival
from pymoo.util.dominator import Dominator
from pymoo.util.misc import vectorized_cdist


class ModifySPEA2Survival(Survival):

    def __init__(self, normalize=False, filter_infeasible=True):
        super().__init__(filter_infeasible)

        # whether the survival should considered normalized distance or just raw
        self.normalize = normalize

        # an object keeping track of normalization points
        self.norm = None

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        pop_nd_arch = kwargs['pop_nd_arch']

        F = pop.get("F").astype(float, copy=False)

        # the domination matrix to see the relation for each solution to another
        M = Dominator().calc_domination_matrix(F)

        # the number of solutions each individual dominates
        S = (M == 1).sum(axis=0)

        # the raw fitness of each solution - strength of its dominators
        R = ((M == -1) * S).sum(axis=1)

        # determine what k-th nearest neighbor to consider
        k = int(np.sqrt(len(pop)))
        if k >= len(pop):
            k = len(pop) - 1

        # if normalization is enabled keep track of ideal and nadir
        if self.normalize:

            # initialize the first time and then always update the boundary points
            if self.norm is None:
                self.norm = HyperplaneNormalization(F.shape[1])
            self.norm.update(F)

            ideal, nadir = self.norm.ideal_point, self.norm.nadir_point

            _F = (F - ideal) / (nadir - ideal)
            dists = vectorized_cdist(_F, _F, fill_diag_with_inf=True)

        # if no normalize is required simply use the F values from the population
        else:
            dists = vectorized_cdist(F, pop_nd_arch.get('F'), fill_diag_with_inf=True)

        # the distances sorted for each individual
        sdists = np.sort(dists, axis=1)

        # inverse distance as part of the fitness
        D = 1 / (sdists[:, 0] + 2)

        # the actual fitness value used to determine who survives
        SPEA_F = R + D

        # set all the attributes to the population
        pop.set(SPEA_F=SPEA_F, SPEA_R=R, SPEA_D=D)

        # get all the non-dominated solutions
        survivors = list(np.where(np.all(M >= 0, axis=1))[0])

        # if we normalize give boundary points most importance - give the boundary points in the nds set the lowest fit.
        if self.normalize:
            I = vectorized_cdist(self.norm.extreme_points, F[survivors]).argmin(axis=1)
            pop[survivors][I].set("SPEA_F", -1.0)

        # identify the remaining individuals to choose from
        H = set(survivors)
        rem = np.array([k for k in range(len(pop)) if k not in H])

        # if not enough solutions, will up by F
        if len(survivors) < n_survive:

            # sort them by the fitness values (lower is better) and append them
            rem_by_F = rem[SPEA_F[rem].argsort()]
            survivors.extend(rem_by_F[:n_survive - len(survivors)])

        # if too many, delete based on distances
        elif len(survivors) > n_survive:

            # remove one individual per loop, until we hit n_survive
            while len(survivors) > n_survive:
                i = dists[survivors][:, survivors].min(axis=1).argmin()
                survivors = [survivors[j] for j in range(len(survivors)) if j != i]

        return pop[survivors]

