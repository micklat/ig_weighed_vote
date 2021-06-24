from ksz_solver import Solver
import numpy as np
from numpy.random import default_rng
from unittest import TestCase
import time

class Test1(TestCase):
    def test_random(self):
        rng = default_rng(seed=1)
        Nv = 10
        Nq = 5
        power = np.full((Nv,), Nq*20)
        preference = rng.integers(0, 10, (Nv, Nq, 2))
        preference -= np.amin(preference, 2)[:,:,np.newaxis]
        start_time = time.perf_counter()
        (k,s,z),result = Solver(preference, power).solve()
        dt = time.perf_counter() - start_time
        print(f"time: {dt} seconds")
        print('preference diff:', preference[:,:,1]-preference[:,:,0])
        print('k:', k)
        print('s:', s)
        print('z:', z)
        print('opt result:', result)


if __name__ == '__main__':
    import pdb, traceback
    try:
        Test1().test_random()
    except:
        traceback.print_exc()
        pdb.post_mortem()


