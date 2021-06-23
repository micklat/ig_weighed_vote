import jax
from jax import numpy as jnp
import numpy as np
from numpy import newaxis
from scipy.optimize import minimize, NonlinearConstraint

# voters pick one of the choices per vote, and assign value to their choice.
# the system calculates the winning choice per vote
# the system distributes payoffs to voters according to their choices and values

# preference - the initial value for the calculation, provided by voters per vote
# initially, value[voter, vote, choice] = preference[voter, vote, choice]

# goals:
# - distribute benefits from votes fairly between voters
# - the stronger the opposition to a winning choice, weaken the winners' weight in other votes

# implementation:
# - decisions are made according to values; the winning choice is the one with the highest total value
# - values are adjusted according to payments

# after convergence, what we want:
# power[voter] == payment[voter].sum()
# winning_choice[vote] = argmax(value[:,vote,:].sum(over voters))
# payment[voter,vote, choice] == value[voter,vote, choice] * value[:,vote,losing_choice].sum() / value[:,vote,winning_choice].sum()
# value[voter] is proportional to preference[voter] 

# v - voter
# q - question (vote)
# c - choice

# inputs:
#
# p[v,q,c] - preference
# t[v] - power of voter v, (total)
#
# optimization variables:
#
# z[q] - soft decision. z[q]=1 means the decision is "yes", z[q]=0 means "no". 
# k[q,c] - weight ratio. Will be optimized towards sum(w[:,q,1-c])/sum(w[:,q,c]) when that ratio is defined.
# s[v] - voter preference scaler. s[v]>=0
#
# intermediate variables:
# 
# w[v,q,c] = s[v] * p[v,q,c]   # value (weight)
# h[q,c] = c*z[q] + (1-c)(1-z[q])   # soft assignment of winner
# y[v,q] = sum(h[k,c]*k[q,c]*w[v,q,c], over c)   # payment by voter v for winning on question q
#
# hard constraints:
#   0 == k[q,c] * w[:,q,c].sum() - w[:,q,1-c].sum()    # k equals weight ratio
#   0 <= s[v]
#   0 == t[v] - y[v].sum()    # power equals sum of payments
#   0 <= z[q] <= 1
# 
# loss function:
#   # we want if w[:,q,c] > w[:,q,1-c] then h[q,c] = 1
#   loss(z, k, s) = -sum(z * (sum(w[:,:,1], axis=voter) - sum(w[:,:,0], axis=voter)))
#
# h[q,c] - "soft winner"
# we want h[q,c] to be close to 0 or 1
# if h[q,0] = 1 we say that choice 0 was selected
# if h[q,1] = 1 we say that choice 1 was selected


class TensorPacker:
    __slots__ = ('shapes', 'length', 'np')

    def __init__(self, shapes, np=np):
        self.np = np
        self.shapes = shapes
        self.length = sum([np.product(shape) for shape in shapes])

    def pack(self, parts):
        return self.np.concatenate([part.ravel() for part in parts])

    def unpack(self, x):
        res = []
        start = 0
        for shape in self.shapes:
            l = np.product(shape)
            res.append(x[start : start+l].reshape(shape))
            start += l
        return res
        

class Solver:  # solve for s,z,k given p,t
    __slots__ = ('preference', 'power', 'Nv', 'Nq',
            '_ksz_packer', '_kstz_constraint_packer')

    def __init__(self, preference, power):
        self.preference = preference
        self.power = power
        Nv, Nq, two = preference.shape
        assert two==2
        self.Nv = Nv
        self.Nq = Nq
        assert self.power.shape == (Nv, )
        self._ksz_packer = TensorPacker(((Nq, 2), (Nv,), (Nq,)))
        self._kstz_constraint_packer = TensorPacker(((Nq,2), (Nv,), (Nv,), (Nq,)))

    def _constraint_func(self, x):
        k,s,z = self._ksz_packer.unpack(x)
        w = s[:,newaxis,newaxis] * self.preference
        w_sums = w.sum(axis=0)
        h = jnp.hstack(((1-z),z))
        y = (k*h*w).sum(axis=2)
        
        return self._kstz_constraint_packer.pack((
            k * w_sums - w_sums.flip(1), # k equals the ratio of w_sums
            s, # s is non-negative
            t - y.sum(1), # power equals the sum of payments
            z, # 0<=z<=1
            ))

    def _loss(self, x):
        k,s,z = self._ksz_packer.unpack(x)
        assert s.shape == (self.Nv,)
        assert k.shape == (self.Nq,2)
        assert z.shape == (self.Nq,)
        w = s[:,newaxis,newaxis] * self.preference
        assert w.shape == (self.Nv, self.Nq, 2)
        loss = z.dot(w[:,:,0].sum(0) - w[:,:,1].sum(0))
        assert not isinstance(loss, np.ndarray) or np.isscalar(loss)
        return loss

    def solve(self):
        Nv, Nq = self.Nv, self.Nq
        Zk = np.zeros((Nq, 2)) 
        Zs = Zt = np.zeros((Nv,))
        Zz = np.zeros((Nq,))
        lower_bounds = self._kstz_constraint_packer.pack((Zk, Zs, Zt, Zz))
        upper_bounds = self._kstz_constraint_packer.pack((Zk, np.full_like(Zs, np.inf), Zt, np.full_like(Zz,1)))
        constraints_jacobian = jax.jacobian(self._constraint_func)
        constraints = NonlinearConstraint(self._constraint_func, lower_bounds, upper_bounds, constraints_jacobian)
        x0 = self._ksz_packer.pack((np.full_like(Zk, 0.5), np.ones((Nv,)), np.full_like(Zz, 0.5)))
        result = minimize(self._loss, x0, method='trust-constr', jac=jax.grad(self._loss), constraints = constraints)
        k,s,z = self._ksz_packer.unpack(result.x)
        return (k,s,z), result 


