from ksz_solver import Solver, h_from_z
from numpy import newaxis
import numpy as np
from numpy.random import default_rng
from unittest import TestCase
import time


def show_violations(k, s, z, p, t):
    w = s[:,np.newaxis,np.newaxis] * p
    h = h_from_z(z)
    y = (k*h*w).sum(2)
    max_t_violation = max(np.abs(1-y.sum(1)/t))
    total_s_violation = np.maximum(-s, 0).sum()
    max_k_violation = (abs(k * w.sum(0) - np.flip(w,2).sum(0))).max()
    total_z_violation = np.maximum(-z, 0).sum() + np.maximum(z-1, 0).sum()
    print("total_s_violation:", total_s_violation)
    print("total_z_violation:", total_z_violation)
    print("max_t_violation:", max_t_violation)
    print("max_k_violation:", max_k_violation)


def uniform_random_case(rng, Nv, Nq):
    power = np.full((Nv,), Nq*20)
    preference = rng.integers(0, 10, (Nv, Nq, 2))
    preference -= np.amin(preference, 2)[:,:,np.newaxis]
    return (preference, power)


def full_report(loss, k, s, z, opt_result, preference, power, dt):
    w = s[:,newaxis,newaxis] * preference
    margins = np.diff(w.sum(0),axis=1)
    raw_margins = np.diff(preference.sum(0), axis=1)
    print(f"weighed margins: {margins}")
    print(f"raw margins: {raw_margins}")
    print(f"loss: {loss}")
    print(f"time: {dt} seconds")
    print('preference diff:', preference[:,:,1]-preference[:,:,0])
    print('k:', k)
    print('s:', s)
    print('z:', z)
    print('opt result:', opt_result)
    show_violations(k, s, z, preference, power)


class Test1(TestCase):
    def test_random(self):
        rng = default_rng(seed=1)
        preference, power = uniform_random_case(rng, 10, 5)
        start_time = time.perf_counter()
        solver = Solver(preference, power)
        (k,s,z),opt_result = solver.solve()
        dt = time.perf_counter() - start_time
        full_report(solver.packed_loss(opt_result.x), k, s, z, opt_result, preference, power, dt)
        k, s, z = solver.sequential_projection(k, s, z)
        print(z, s, solver.ksz_loss(k, s, z))

if __name__ == '__main__':
    import pdb, traceback
    try:
        Test1().test_random()
    except:
        traceback.print_exc()
        pdb.post_mortem()


