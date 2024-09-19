import sys
sys.path.append('../')

from algorithm.SAPPS import SAPPS
from pymoo.optimize import minimize
from utilsmodule.running_display import RunningDisplayAndDataSave
import pickle
from datetime import datetime
import os
from surrogate.gpr_model import GaussianProcess
from utilsmodule.database import DataBase
from selection.hvi import HVI
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from multiprocessing import Pool
from selection.pullselection import RefSPEA2Sel


def run_algorithm_benchmark(problem, run_num, problem_name, root_path):
    # not use
    num_cand = 1000

    num_cand_select = 5
    pop_init_num = 105
    total_evals = 1000
    max_gen = int((total_evals - pop_init_num) / num_cand_select)
    ideal_nadir_threshold = 1e-3
    lambda_cv_ucb = 0.
    lambda_F_lcb = 0.0
    last_gen = 30
    surr_evo_gen = 100
    surr_model = GaussianProcess(problem.n_var, problem.n_obj, problem.n_ieq_constr, 5.)
    sample = LatinHypercubeSampling()
    database = DataBase(problem.n_obj, problem.n_ieq_constr, 0., 1., problem)
    push_survival = HVI()
    pull_survival = RefSPEA2Sel()

    data_save_path = f"{root_path}/{problem_name}/run_{run_num}"
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    my_callback = RunningDisplayAndDataSave(total_evals, data_save_path, run_num, problem_name)
    alg = SAPPS(surrogate_model=surr_model, sampling=sample, pop_size=105, database=database,
                                  push_survival=push_survival, pull_survival=pull_survival, num_cand=num_cand,
                                  num_cand_survival=num_cand_select, init_num=pop_init_num, max_gen=max_gen,
                                  change_threshold=ideal_nadir_threshold, problem_name=problem_name,
                                  data_save_path=data_save_path, lambda_cv_ucb=lambda_cv_ucb, lambda_F_lcb=lambda_F_lcb,
                                  last_gen=last_gen, surr_evo_gen=surr_evo_gen, total_eval=total_evals, show_flag=False)

    res = minimize(problem, algorithm=alg, termination=('n_eval', total_evals), callback=my_callback, verbose=False)
    time = str(time_now.year) + '-' + str(time_now.month) + '-' + str(time_now.day) + '-' + time_now.strftime(
        "%H_%M_%S")
    history_F = res.algorithm.callback.data['pop_F_his']
    with open(f"{data_save_path}/{problem_name}_pps_{time}.pickle", 'wb') as output_file:
        pickle.dump({'history': history_F, 'arch_F': res.pop.get('F'), 'pop': res.pop}, output_file)


if __name__ == '__main__':

    from display_problems.problem_list import MW_list

    pro_ser_name = "MW"
    n_var = 10
    pro_list = MW_list

    time_now = datetime.now()
    time = str(time_now.year) + '-' + str(time_now.month) + '-' + str(time_now.day) + '-' + time_now.strftime(
        "%H_%M_%S")
    root_save_path = f"../datafile/benchmark/{pro_ser_name}_n{n_var}/{time}/"

    # save gif ?
    save_gif_flag = True

    every_problem_run_total = 30

    with Pool(processes=5) as pool:

        for i in range(1, 14+1):
            test_problem_args = []

            for run_id in range(every_problem_run_total):
                test_problem_args.append((pro_list[f"{pro_ser_name}{i}"](n_var=n_var),
                                          run_id, f"{pro_ser_name}{i}", root_save_path))

            iters = pool.starmap(run_algorithm_benchmark, test_problem_args)
            for ret in iters:
                print(f"finish {pro_ser_name}{i}")
