import jax
from jax import numpy as jnp
import numpy as np
from numpy import newaxis
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from tensor_packer import TensorPacker


def trivial_questions(preference):
    """A question is trivial if either choice has 0 support"""
    p_sum = preference.sum(0)
    unsupported_choices = (p_sum==0)
    any_unsupported = np.any(unsupported_choices, axis=1)
    return any_unsupported


def k_from_w(w, np=np):
    w_sums = w.sum(axis=0)
    return np.flip(w_sums,1) / w_sums


def h_from_z(z, np=np):
    return np.vstack(((1-z),z)).T

def w_from_s(s, preference, np=np):
    return s[:,np.newaxis,np.newaxis] * preference

class Solver:
    __slots__ = ('preference', 'power', 'Nv', 'Nq', '_sz_packer')

    def __init__(self, preference, power):
        trivial = trivial_questions(preference)
        if np.any(trivial):
            raise ValueError(f"Votes {np.where(trivial)[0]} are trivial. Trivial votes must be removed first.")
        self.preference = preference
        self.power = power
        Nv, Nq, two = preference.shape
        assert two==2
        self.Nv = Nv
        self.Nq = Nq
        assert self.power.shape == (Nv, )
        self._sz_packer = TensorPacker(((Nv,), (Nq,)))
    
    def _power_constraint(self, x):
        s,z = self._sz_packer.unpack(x)
        w = w_from_s(s, self.preference, jnp)
        w_sums = w.sum(axis=0)
        k = k_from_w(w, jnp)
        h = h_from_z(z, jnp)
        y = (k*h*w).sum(axis=2)
        return self.power - y.sum(1) # should be 0

    def packed_loss(self, x):
        s,z = self._sz_packer.unpack(x)
        return self.sz_loss(s, z)

    def sz_loss(self, s, z):
        assert s.shape == (self.Nv,)
        assert z.shape == (self.Nq,)
        w = w_from_s(s, self.preference, jnp)
        assert w.shape == (self.Nv, self.Nq, 2)
        loss = z.dot(w[:,:,0].sum(0) - w[:,:,1].sum(0))
        assert not isinstance(loss, np.ndarray) or np.isscalar(loss)
        return loss

    def solve(self):
        Nv, Nq = self.Nv, self.Nq
        Zs = Zt = np.zeros((Nv,))
        Zz = np.zeros((Nq,))
        lower_sz = np.zeros((Nv+Nq,))
        upper_sz = np.concatenate((np.full_like(Zs, np.inf), np.ones((Nq,))))
        bounds = Bounds(lower_sz, upper_sz)
        power_jac = jax.jit(jax.jacobian(self._power_constraint))
        power_constraint = NonlinearConstraint(self._power_constraint, Zt, Zt, power_jac)
        x0 = self._sz_packer.pack((np.ones((Nv,)), np.full_like(Zz, 0.5)), np)
        loss_grad = jax.jit(jax.grad(self.packed_loss))
        result = minimize(self.packed_loss, x0, jac=loss_grad, bounds=bounds, constraints=power_constraint)
        s,z = self._sz_packer.unpack(result.x)
        return (s,z), result 

    def wrap_projections(self, x, projections):
        s,z = self._sz_packer.unpack(x)
        for projection in projections:
            s,z = projection(s,z)
        return self._sz_packer.pack((s,z), np)

    def project_z(self, s, z):
        z = (z>0.5).astype(float)
        return s, z

    def project_s(self, s, z):
        w = w_from_s(s, self.preference)
        h = h_from_z(z)
        k = k_from_w(w)
        weighed_preference = (k*h*self.preference).sum(2).sum(1)
        positive = (weighed_preference > 0)
        s[positive] = self.power[positive] / weighed_preference[positive]
        return s, z

    def sequential_projection(self, s, z):
        for proj in (self.project_z, self.project_s):
            s, z = proj(s, z)
        return s, z
    
