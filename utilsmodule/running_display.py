import numpy as np
from pymoo.core.callback import Callback
from pymoo.util.display.output import pareto_front_if_possible
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from algorithm.SAPPS import SAPPS
import pandas as pd
import os
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival


class RunningDisplayAndDataSave(Callback):

    def __init__(self, max_eval: int, save_path: str, run_id: int, problem_name: str, output=None, progress=False, verbose=False):
        super().__init__()
        self.max_eval = max_eval
        self.output = output
        self.verbose = verbose
        self.data_title = ['alg_igd', 'alg_HV']
        self.data['pop_F_his'] = []
        self.data['net_F_his'] = []
        # self.progress_bar = tqdm(total=max_eval)
        self.pf = None
        self.pref_sample = None
        self.save_file_path = os.path.join(save_path, 'indicator.csv')
        self.run_id = run_id
        self.problem_name = problem_name
        self.survival_sel = RankAndCrowdingSurvival()

    def initialize(self, algorithm: SAPPS):
        problem = algorithm.problem
        self.pf = pareto_front_if_possible(problem)

    def update(self, algorithm: SAPPS):

        F, feas = algorithm.opt.get("F", "feas")
        F = F[feas]
        igd_value, hv_value = np.nan, np.nan
        gen = algorithm.n_gen
        if len(F) > 0:

            if self.pf is not None:

                if feas.sum() > 0:
                    igd_value = IGD(self.pf).do(F)
                    hv_value = Hypervolume(pf=self.pf, zero_to_one=True).do(F)

        if algorithm.push_flag:
            search_stage = "push"
        else:
            search_stage = "pull"

        problem = algorithm.problem

        print(f"\n{self.problem_name} run_id: {self.run_id} | evals: {algorithm.evaluator.n_eval}/{self.max_eval} | {search_stage}",
              f"| pop_igd_hv: {[round(igd_value, 4), round(hv_value, 4)]}")

        data = np.array([[igd_value, hv_value]])
        if gen == 1:
            df = pd.DataFrame(data, columns=self.data_title)
            df.to_csv(self.save_file_path, index=False)
        else:
            df = pd.DataFrame(data, columns=self.data_title)
            df.to_csv(self.save_file_path, mode='a', index=False, header=False)

        pop_nds_gen = self.survival_sel.do(problem, algorithm.pop, n_survive=50)
        self.data['pop_F_his'].append(pop_nds_gen.get('F'))


class RunningDisplay(Callback):

    def __init__(self, max_eval: int, save_path: str, run_id: int, problem_name: str, output=None, progress=False, verbose=False):
        super().__init__()
        self.max_eval = max_eval
        self.output = output
        self.verbose = verbose
        self.data_title = ['alg_igd', 'alg_HV']
        self.data['pop_F_his'] = []
        self.data['net_F_his'] = []
        # self.progress_bar = tqdm(total=max_eval)
        self.pf = None
        self.pref_sample = None
        self.save_file_path = os.path.join(save_path, 'indicator.csv')
        self.run_id = run_id
        self.problem_name = problem_name
        self.survival_sel = RankAndCrowdingSurvival()

    def initialize(self, algorithm: SAPPS):
        problem = algorithm.problem
        self.pf = pareto_front_if_possible(problem)

    def update(self, algorithm: SAPPS):

        F, feas = algorithm.opt.get("F", "feas")
        F = F[feas]
        igd_value, hv_value = np.nan, np.nan
        if len(F) > 0:

            if self.pf is not None:

                if feas.sum() > 0:
                    igd_value = IGD(self.pf, zero_to_one=True).do(F)
                    hv_value = Hypervolume(pf=self.pf).do(F)

        if algorithm.push_flag:
            search_stage = "push"
        else:
            search_stage = "pull"

        print(f"\n{self.problem_name} run_id: {self.run_id} | evals: {algorithm.evaluator.n_eval}/{self.max_eval} | {search_stage}",
              f"| pop_igd_hv: {[round(igd_value, 4), round(hv_value, 4)]}")

