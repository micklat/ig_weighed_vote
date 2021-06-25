from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import numpy as np
import scipy
from numpy import newaxis
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from tensor_packer import TensorPacker


def uncontested(preference):
    """A question is trivial if either choice has 0 support"""
    p_sum = preference.sum(0)
    unsupported_choices = (p_sum==0)
    any_unsupported = np.any(unsupported_choices, axis=1)
    uncontested_choices = (np.diff(p_sum).squeeze(-1) > 0)[any_unsupported]
    return any_unsupported, uncontested_choices

def sum_of_squares(x):
    return x.dot(x)

# array axes:
# v - voter
# q - question (vote)
# c - choice
#
# inputs:
#
# p[v,q,c] # preference
# t[v] # total power of voter v
#
# hyperparameters:
#
# r # decision sharpness coefficient (starts small and increases)
#
# optimization variables:
#
# l[v] # pre-scaler of preferences
# 
# intermediate variables
#
# s[v] = log(110+exp(l[v]))  # preference scaler
# w[v,q,c] = s[v] * p[v,q,c]  # weighed vote
# ws[q,c] = w.sum(0)  # total weighed vote for each choice
# m[q] = ws[q].diff()  # weighed votes margin
# z[q] = expit(r * m[q])   # soft decision
# h[q,c] = c*z[q] + (1-c)*(1-z[q]) = z[q]*(2c-1) + 1-c  # soft decision as choice mass
# k[q] = ws[q].dot(1-h[q]) / ws[q].dot(h[q])  # soft ratio of losing weights to winning weights
# y[v,q] = k[q] * w[v,q].dot(h[q])  # payment for winning some of the votes
#
# objective:
#   loss = square(t - y.sum(1)).sum()


class ScipyAPIs:
    np = np
    scipy = scipy


class JaxAPIs:
    np = jax.numpy
    scipy = jax.scipy


class Solver:
    __slots__ = ('preference', 'power', 'Nv', 'Nq', 'expit_sharpness', 'original_preference', 'uncontested_q', 'uncontested_c',
            'trace')

    def __init__(self, preference, power, sharpness=1, trace=None):
        self.trace = trace
        self.uncontested_q, self.uncontested_c = uncontested(preference)
        self.original_preference = preference
        self.preference = preference = preference[:, ~self.uncontested_q]
        self.power = power
        self.expit_sharpness = sharpness
        Nv, Nq, two = preference.shape
        assert two==2
        self.Nv = Nv
        self.Nq = Nq
        assert self.power.shape == (Nv, )
    
    def s_from_l(self, l, apis=ScipyAPIs):
        np = apis.np
        return np.log(1+np.exp(l))

    def calculate(self, l, apis=ScipyAPIs):
        assert l.shape == (self.Nv,)
        np = apis.np
        s = self.s_from_l(l, apis)  # [v]
        w = s[:,newaxis,newaxis] * self.preference
        ws = w.sum(0) # [q,c], sum of weighed votes per choice
        z = apis.scipy.special.expit(self.expit_sharpness * np.diff(ws).squeeze(-1))  # soft decision
        h = np.vstack((1-z,z)).T  # [q,c], choice-selection factor
        assert h.shape == (self.Nq, 2)
        hws = (h*w).sum(2)  # [v,q], weighed preference for the winning choice
        winner_support = hws.sum(0)  # [q], soft-winner support
        total_support = ws.sum(1)   # [q], total weighed votes per question
        loser_support = total_support - winner_support
        k = loser_support / winner_support   # cost factor
        loss = sum_of_squares(self.power - hws.dot(k))
        results = locals()
        if self.trace is not None:
            self.trace.append(results)
        return results

    def loss(self, l, apis=ScipyAPIs):
        return self.calculate(l, apis)['loss']

    def no_contest(self):
        return np.all(self.uncontested_q)

    def solve(self):
        Nv, Nq = self.Nv, self.Nq
        l0 = np.zeros(Nv)
        loss_grad = jax.jit(jax.grad(lambda x: self.loss(x, JaxAPIs)))
        result = minimize(self.loss, l0, jac=loss_grad)
        return self.s_from_l(result.x), result 

    def decisions(self, l):
        d = self.calculate(l)
        decisions = np.zeros_like(self.uncontested_q)
        decisions[self.uncontested_q] = self.uncontested_c
        decisions[~self.uncontested_q] = (d['z']>0.5)
        return decisions


