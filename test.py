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
    raw_margins = np.diff(preference.sum(0), axis=1)[~solver.uncontested_q]
    print(f"weighed margins: {margins}")
    print(f"raw margins: {raw_margins}")
    print(f"loss: {opt_result.fun}")
    print(f"time: {dt} seconds")
    print('preference diff:', preference[:,:,1]-preference[:,:,0])
    print('s:', s)
    print('decisions:', solver.decisions(opt_result.x))
    print('opt result:', opt_result)
    return locals()


class Test1(TestCase):
    def test_random(self):
        rng = default_rng(seed=6)
        preference, power = uniform_random_case(rng, 10, 100)
        start_time = time.perf_counter()
        trace = []
        solver = Solver(preference, power, trace=trace)
        s,opt_result = solver.solve()
        dt = time.perf_counter() - start_time
        return full_report(solver, s, opt_result, preference, power, dt)


if __name__ == '__main__':
    import pdb, traceback
    try:
        d = Test1().test_random()
    except:
        traceback.print_exc()
        pdb.post_mortem()
    else:
        import pylab
        pylab.figure(1)
        maxx = d['raw_margins'].max()
        minx = d['raw_margins'].min()
        pylab.xlabel('unweighed margin')
        pylab.ylabel('weighed margin')
        pylab.axhline(y=0, color='k')
        pylab.axvline(x=0, color='k')
        pylab.plot([minx,maxx],[minx,maxx],'--')
        pylab.scatter(d['raw_margins'], d['margins'],s=20)
        pylab.show()

