import numpy as np
import datetime
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population


def safe_divide(x1, x2):
    """
    Divide x1 / x2, return 0 where x2 == 0
    """
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


def get_ideal_point_no_constr(z: np.ndarray, pop_F_norm: np.ndarray):
    ideal_point = np.min(np.row_stack((pop_F_norm - 0.1, z)), axis=0)
    return ideal_point


def get_z(pop_F_norm: np.ndarray):
    z = np.min(pop_F_norm, axis=0)
    return z


def calc_max_change(z_his, nadir_his, gen, last_gen):
    delta_value = 1e-6 * np.ones(z_his[0].shape[0])

    # delta_value = 1e-6
    z_now = z_his[gen-1]
    z_now2last = z_his[gen-1-last_gen]
    print(f"z_now: {z_now}, z_now2last: {z_now2last}")
    # print(f"z_his: {z_his}")
    # print(f"nadir_his: {nadir_his}")
    rz = np.max(np.abs(z_now - z_now2last) / np.where(z_now2last > delta_value, z_now2last, delta_value))
    print(f"rz: {rz}")

    return rz


def calc_hv_change(F_now, F_last, change_threshold: float, push_stage: bool):
    nd_now = F_now[NonDominatedSorting().do(F_now, only_non_dominated_front=True)]
    nd_last = F_last[NonDominatedSorting().do(F_last, only_non_dominated_front=True)]
    ref_point = np.max(np.vstack((nd_now, nd_last)), axis=0) * 1.2
    ideal_point = np.min(np.vstack((nd_now, nd_last)), axis=0)
    hv_calc = Hypervolume(ref_point=ref_point, ideal=ideal_point, nadir=ref_point)
    hv_now = hv_calc.do(nd_now)
    hv_last = hv_calc.do(nd_last)
    delta_value = 1e-6
    hv_change = np.max(np.abs(hv_now - hv_last) / np.where(hv_last > delta_value, hv_last, delta_value))
    print(f"hv change: {hv_change}")


    return hv_change


def push_pull_judge(gen: int, last_gen: int, total_gen: int, change_threshold: float,
                    z_his: list[np.ndarray], nadir_his: list[np.ndarray], push_stage: bool) -> bool:

    max_change = 1e6
    if gen > last_gen and push_stage:
        max_change = calc_max_change(z_his, nadir_his, gen, last_gen)
        # print("max_change", self.max_change_pop)

    if max_change <= change_threshold and push_stage:
        push_stage = False

    return push_stage


def push_z_judge(gen: int, last_gen: int, total_gen: int, change_threshold: float,
                    z_his: list[np.ndarray], nadir_his: list[np.ndarray], repair_flag: bool) -> bool:

    max_change = 1e6
    if gen > last_gen and repair_flag:
        max_change = calc_max_change(z_his, nadir_his, gen, last_gen)
        # print("max_change", self.max_change_pop)

    if max_change <= change_threshold:
        repair_f_flag = True

    return repair_flag


def calculate_time_diff(time_start: datetime.datetime, time_end: datetime.datetime):
    diff_sec = (time_end - time_start).seconds
    h, m, s = 0, 0, 0
    if diff_sec >= 3600:
        h = int(diff_sec / 3600)
        diff_sec = diff_sec % 3600

    if diff_sec >= 60:
        m = int(diff_sec / 60)
        diff_sec = diff_sec % 60

    s = diff_sec

    total_time = f"{h}h:{m}m:{s}s"
    return total_time


def estimate_constraint_type(pop: Population):

    pop_F = pop.get('F')
    pop_nd = pop[NonDominatedSorting().do(pop_F, only_non_dominated_front=True)]
    pop_nd_fea = pop_nd[pop_nd.get('feas')]
    feasible_ratio = len(pop_nd_fea) / len(pop_nd)
    if feasible_ratio >= 1:
        con_type = 0
    elif (feasible_ratio > 0) and (feasible_ratio < 1):
        con_type = 1
    else:
        con_type = 2

    return con_type


def calc_ref_assign(F, ref_dirs):
    ref_dirs_norm = np.linalg.norm(ref_dirs, axis=1)
    d1s = F.dot(ref_dirs.T) / ref_dirs_norm
    F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
    d2s = np.sqrt(F_norm**2 - d1s**2)
    ref_index = np.argmin(d2s, axis=1)
    return ref_index

