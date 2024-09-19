from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, calc_crowding_distance, binary_tournament
from pymoo.core.mating import Mating
import copy
from pymoo.algorithms.soo.nonconvex.de import Variant
from pymoo.operators.control import NoParameterControl
from pymoo.util.optimum import filter_optimum
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.core.repair import Repair
from pymoo.util.display.multi import MultiObjectiveOutput
import numpy as np
from pymoo.core.population import Population


class BoundRepair(Repair):

    def _do(self, problem, X, **kwargs):
        X = np.where(X > problem.xu, problem.xu, X)
        X = np.where(X < problem.xl, problem.xl, X)
        return X


class RankAndCrowdingSurvivalIgnoreConstraint(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class CCMO(GeneticAlgorithm):
    # problem is surrogate model
    def __init__(self, pop_o_init, pop_size, n_offspring, epsilon_cv=0., **kwargs):
        super(CCMO, self).__init__(pop_size=pop_size, sampling=pop_o_init, n_offspring=n_offspring,
                                   output=MultiObjectiveOutput(), advance_after_initial_infill=True, **kwargs)

        self.pop_h = None
        self.survival = RankAndCrowdingSurvival()
        self.survival_help = RankAndCrowdingSurvivalIgnoreConstraint()
        self.mating_o = Mating(selection=TournamentSelection(func_comp=binary_tournament),
                               crossover=SBX(eta=15, prob=0.9),
                               mutation=PM(eta=20),
                               repair=BoundRepair(),
                               eliminate_duplicates=self.eliminate_duplicates,
                               n_max_iterations=100)
        # mating_h
        self.mating_h = Variant(selection='rand', n_diffs=1, crossover='bin', control=NoParameterControl)
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'
        self.epsilon_cv = epsilon_cv
        self.pop_sel = None
        self.cand = None

    def _infill(self):
        # print(self.pop_h.get('crowding'))
        off_o = self.mating_o.do(problem=self.problem, pop=self.pop, n_offsprings=self.n_offsprings, algorithm=self)
        off_h = self.mating_o.do(problem=self.problem, pop=self.pop_h, n_offsprings=self.n_offsprings, algorithm=self)
        self.evaluator.eval(self.problem, off_o)
        self.evaluator.eval(self.problem, off_h)
        pop_o = Population.merge(self.pop, off_o, off_h)
        pop_h = copy.deepcopy(Population.merge(self.pop_h, off_o, off_h))
        pop_G = self.pop.get('G')
        pop_epsilon_cv = np.sum(np.maximum(pop_G, 0.), axis=1)[:, None] - self.epsilon_cv
        self.pop.set(CV=pop_epsilon_cv)
        self.pop_h.set(CV=np.zeros([len(self.pop_h), 1]))
        self.pop = self.survival.do(problem=self.problem, pop=pop_o, n_survive=self.pop_size)
        self.pop_h = self.survival_help.do(problem=self.problem, pop=pop_h, n_survive=self.pop_size)
        self.cand = self.pop_h

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop_h = copy.deepcopy(infills)
        self.pop = self.survival.do(self.problem, infills, self.pop_size)
        self.pop_h = self.survival_help.do(self.problem, self.pop_h, self.pop_size)
        self.archive_all = filter_optimum(infills)

    def _advance(self, infills=None, **kwargs):
        temp = Population.merge(self.pop, self.pop_h)
        self.archive_all = Population.merge(self.archive_all, filter_optimum(temp))
        self.pop_o_h = Population.merge(self.pop, self.pop_h)
        # self.cand = self.archive_all
