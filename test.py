from ksz_solver import Solver
import numpy as np
from unittest import TestCase

class Test1(TestCase):
    def test_random(self):
        Nv = 10
        Nq = 5
        power = np.full((Nv,), Nq*20)
        preference = np.random.randint(0, 10, (Nv, Nq, 2))
        (k,s,z),result = Solver(preference, power).solve()
        print('preference:', preference)
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


