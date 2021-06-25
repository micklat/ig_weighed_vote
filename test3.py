from u_solver import Solver
from numpy import newaxis
import numpy as np
from numpy.random import default_rng
from unittest import TestCase
import time


def uniform_random_case(rng, Nv, Nq):
    power = np.full((Nv,), Nq*20)
    preference = rng.integers(0, 10, (Nv, Nq, 2))
    preference -= np.amin(preference, 2)[:,:,np.newaxis]
    return (preference, power)


def full_report(solver, s, opt_result, preference, power, dt):
    d = solver.calculate(opt_result.x)
    margins = np.diff(d['w'].sum(0),axis=1)
    raw_margins = np.diff(preference.sum(0), axis=1)
    print(f"weighed margins: {margins}")
    print(f"raw margins: {raw_margins}")
    print(f"loss: {opt_result.fun}")
    print(f"time: {dt} seconds")
    print('preference diff:', preference[:,:,1]-preference[:,:,0])
    print('s:', s)
    print('z:', d['z'])
    print('opt result:', opt_result)


class Test1(TestCase):
    def test_random(self):
        rng = default_rng(seed=6)
        preference, power = uniform_random_case(rng, 10, 100)
        start_time = time.perf_counter()
        solver = Solver(preference, power)
        s,opt_result = solver.solve()
        dt = time.perf_counter() - start_time
        full_report(solver, s, opt_result, preference, power, dt)


if __name__ == '__main__':
    import pdb, traceback
    try:
        Test1().test_random()
    except:
        traceback.print_exc()
        pdb.post_mortem()


