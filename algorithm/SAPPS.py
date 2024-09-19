import copy

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from surrogate.gpr_model import GaussianProcess
from pymoo.core.sampling import Sampling
from utilsmodule.database import DataBase
from selection.hvi import BatchSelection
import numpy as np
from utilsmodule.tools import calc_hv_change, get_z
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from surrogate.surr_problem import SurrogateProblem, SurrogateProblemNoCV
from pymoo.optimize import minimize
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from collections import deque
from utilsmodule.tools import estimate_constraint_type
from selection.ArchSelection import batch_active_object_constraint_select
from algorithm.ccmo import CCMO
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.core.evaluator import Evaluator
from sklearn.cluster import KMeans


class SAPPS(GeneticAlgorithm):

    def __init__(self, surrogate_model: GaussianProcess,
                 sampling: Sampling, database: DataBase, push_survival: BatchSelection,
                 pull_survival: BatchSelection, num_cand: int, num_cand_survival: int,
                 init_num: int, max_gen: int, change_threshold: float,
                 problem_name: str, data_save_path: str, lambda_cv_ucb: float,
                 lambda_F_lcb: float, last_gen: int, surr_evo_gen: int, total_eval: int, show_flag, **kwargs):

        super(SAPPS, self).__init__(sampling=sampling, **kwargs)
        self.surr_model = surrogate_model
        self.num_cand = num_cand
        self.num_cand_survival = num_cand_survival
        self.push_survival = push_survival
        self.pull_survival = pull_survival
        self.database = database
        self.init_num = init_num
        self.max_gen = max_gen
        self.last_gen = last_gen
        self.change_threshold = change_threshold
        self.problem_name = problem_name
        self.data_save_path = data_save_path
        self.lambda_cv_ucb = lambda_cv_ucb
        self.lambda_F_lcb = lambda_F_lcb
        self.sa_evo_gen = surr_evo_gen
        self.filter_duplicate = DefaultDuplicateElimination()
        self.sampling = sampling
        self.pop_F_queue = deque(maxlen=last_gen+1)
        self.ideal_fix_flag = False
        self.hv_change = 1.0
        self.show_flag = show_flag
        self.pop_sel = None
        self.cand = None
        self.pred_pop = None
        self.arch_norm = None
        self.pop_surr = None
        self.surr_cand = None
        self.ideal_fix_gen = 10
        self.z_queue = deque(maxlen=self.ideal_fix_gen + 1)
        self.total_eval = total_eval

    def _setup(self, problem, **kwargs):
        if problem.n_obj == 2:
            self.ref_dirs = UniformReferenceDirectionFactory(self.problem.n_obj, n_points=5).do()
        else:
            ref_points = UniformReferenceDirectionFactory(n_dim=problem.n_obj, n_partitions=100).do()
            cluster_ref = KMeans(n_clusters=5, random_state=0).fit(ref_points)
            m_point = cluster_ref.cluster_centers_
            self.ref_dirs = m_point

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.ideal_list = []
        self.nadir_list = []
        self.ideal_list.append(np.min(self.pop.get('F'), axis=0))
        self.nadir_list.append(np.max(self.pop.get('F'), axis=0))
        self.push_flag = True
        self.share_parameters = True
        self.z = np.zeros(self.problem.n_obj)
        self.pop_F_queue.append(self.pop.get('F'))

    def _infill(self):

        if self.n_gen > self.last_gen and self.push_flag:
            F_last = self.pop_F_queue.popleft()
            F_now = self.pop_F_queue.pop()
            self.hv_change = calc_hv_change(F_now, F_last, self.change_threshold, self.push_flag)
            self.pop_F_queue.append(F_now)
            self.pop_F_queue.appendleft(F_last)

        if ((self.hv_change < self.change_threshold) or (self.evaluator.n_eval > self.total_eval / 2)) \
                and self.push_flag:
            self.push_flag = False

        if self.push_flag:
            cand_sel = self.ignore_constraint_search()

        else:
            estimate_type = estimate_constraint_type(self.pop)
            if estimate_type == 0:
                """ignore constraint search"""
                print("ignore constraint search")
                cand_sel = self.ignore_constraint_search()
            elif estimate_type == 1:
                """epsilon constraint search and ignore constraint"""
                print("epsilon constraint search and ignore constraint")
                rand_num = np.random.rand(1)[0]
                if rand_num < 0.5:
                    cand_sel = self.constraint_search()
                else:
                    cand_sel = self.constraint_search()
            else:
                "epsilon constraint search and constraint search"
                print("epsilon constraint search and constraint search")
                cand_sel = self.constraint_search()

        self.evaluator.eval(self.problem, cand_sel)

        self.pop = Population.merge(self.pop, cand_sel)
        self.pop_sel = cand_sel

        if self.push_flag:
            self.pop_F_queue.append(self.pop.get('F'))
            if not self.ideal_fix_flag:
                self.z_queue.append(np.min(self.pop.get('F'), axis=0))

    def _advance(self, infills=None, **kwargs):

        self.ideal_list.append(np.min(self.pop.get('F'), axis=0))
        self.nadir_list.append(np.max(self.pop.get('F'), axis=0))
        self.pop_F_queue.append(self.pop.get('F'))

    def ignore_constraint_search(self):

        opt_pro = SurrogateProblemNoCV(n_var=self.problem.n_var, n_obj=self.problem.n_obj, n_ieq_constr=0,
                                       surr_model=self.surr_model, G_norm_0=np.zeros(self.problem.n_ieq_constr),
                                       lambda_F_lcb=0.1, lambda_cv_ucb=self.lambda_cv_ucb,
                                       xl=self.problem.xl, xu=self.problem.xu)

        fit_pop = RankAndCrowdingSurvival().do(opt_pro, self.pop, n_survive=400)
        fit_pop = DefaultDuplicateElimination().do(fit_pop)
        train_data = self.database.create_data(fit_pop)

        pop_cov = RankAndCrowdingSurvival().do(opt_pro, self.pop, n_survive=150)
        pop_cov_init = norm_pop(pop_cov, self.database)
        self.surr_model.fit(train_data['X'], train_data['F'], G=None)
        opt_pro.set_surrogate(self.surr_model)
        evo_alg = NSGA2(sampling=pop_cov_init, pop_size=100, n_offspring=100)
        res = minimize(opt_pro, evo_alg, ('n_gen', 100))
        self.pop_surr = copy.deepcopy(res.pop)
        pop_cand_unique = DefaultDuplicateElimination().do(res.pop)
        arch_norm = norm_pop(pop_cov, self.database)
        cand_sel = self.push_survival.select(copy.deepcopy(arch_norm), copy.deepcopy(pop_cand_unique), n_select=5,
                                             transformer=self.database, surr_pro=self.surr_model,
                                             ref_dirs=self.ref_dirs)

        self.arch_norm = norm_pop(pop_cov, self.database)
        self.surr_cand = self.push_survival.select(copy.deepcopy(arch_norm), copy.deepcopy(pop_cand_unique), n_select=5,
                                                   transformer=self.database, surr_pro=self.surr_model,
                                                   ref_dirs=self.ref_dirs)
        Evaluator().eval(opt_pro, self.surr_cand)
        self.surr_cand.set("ooF", self.surr_cand.get('F'))
        return cand_sel

    def constraint_search(self):

        fit_pop = batch_active_object_constraint_select(self.problem, self.pop, num_base_pop=300, num_pre_select=200,
                                                        num_batch_blad_select=100)
        train_data = self.database.create_data(fit_pop)
        z_point = get_z(train_data['F'])
        self.surr_model.fit(X=train_data['X'], F=train_data['F'], G=train_data['G'])
        G_norm_0 = self.database.constr_transform.transform(np.zeros([1, self.surr_model.n_ieq_constr]))[0, :]
        pop_evo_init = RankAndCrowdingSurvival().do(self.problem, self.pop, n_survive=100, num_assist_pop=50,
                                                    num_main_pop=50)
        pop_evo_init_norm = norm_pop(pop_evo_init, self.database)
        evo_alg = CCMO(pop_o_init=pop_evo_init_norm, pop_size=100, n_offspring=100, epsilon_cv=0.)
        opt_pro = SurrogateProblem(n_var=self.problem.n_var, n_obj=self.problem.n_obj,
                                   n_ieq_constr=self.problem.n_ieq_constr, surr_model=self.surr_model,
                                   G_norm_0=G_norm_0, lambda_F_lcb=self.lambda_F_lcb,
                                   lambda_cv_ucb=self.lambda_cv_ucb, norm_z=z_point, xl=self.problem.xl,
                                   xu=self.problem.xu, epsilon_cv=0.)

        res = minimize(opt_pro, evo_alg, ('n_gen', 10))
        self.pop_surr = copy.deepcopy(res.algorithm.pop)
        pop_cand_unique = DefaultDuplicateElimination().do(res.pop)
        arch_pop_norm = norm_pop(pop_evo_init, self.database)
        cand_sel = self.pull_survival.select(copy.deepcopy(arch_pop_norm), copy.deepcopy(pop_cand_unique), n_select=5, transformer=self.database,
                                             surr_pro=self.surr_model, ref_dirs=self.ref_dirs, problem=opt_pro)
        self.arch_norm = norm_pop(pop_evo_init, self.database)
        self.surr_cand = self.pull_survival.select(copy.deepcopy(arch_pop_norm), copy.deepcopy(pop_cand_unique), n_select=5, transformer=self.database,
                                             surr_pro=self.surr_model, ref_dirs=self.ref_dirs, problem=opt_pro)
        Evaluator().eval(opt_pro, self.surr_cand)
        self.surr_cand.set("ooF", self.surr_cand.get('F'))
        return cand_sel


def norm_pop(pop, data_base):
    pop_X = pop.get('X')
    pop_F = pop.get('F')
    pop_G = pop.get('G')
    pop_X_norm = data_base.x_transform.transform(pop_X)
    pop_F_norm = data_base.obj_transform.transform(pop_F)
    pop_G_norm = data_base.constr_transform.transform(pop_G)
    G_norm_0 = data_base.constr_transform.transform(np.zeros([1, data_base.n_ieq_constr]))[0, :]
    pop_norm = Population.new(X=pop_X_norm, F=pop_F_norm, G=pop_G_norm - G_norm_0)
    return pop_norm
