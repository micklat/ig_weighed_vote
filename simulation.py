from jax import numpy as jnp
from jax.numpy import newaxis

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
# y[v,q,c] = h[k,c]*k[q,c]*w[v,q,c]   # payment by voter v for choice c in question q
#
# hard constraints:
#   0 <= z[q] <= 1
#   0 <= s[v]
#   t[v] == y[v,:,:].sum()    # power equals sum of payments
#   k[q,c] * w[:,q,c].sum() = w[:,q,1-c].sum()    # k equals weight ratio
# 
# loss function:
#   # we want if w[:,q,c] > w[:,q,1-c] then h[q,c] = 1
#   loss(z, k, s) = -sum(z * (sum(w[:,:,1], axis=voter) - sum(w[:,:,0], axis=voter)))
#
# h[q,c] - "soft winner"
# we want h[q,c] to be close to 0 or 1
# if h[q,0] = 1 we say that choice 0 was selected
# if h[q,1] = 1 we say that choice 1 was selected


def loss(s, z, k, preference, power):
    """
    let there by Nv voters and Nq questions.
    input shapes:
        s: (Nv,)
        z: (Nq,)
        k: (Nq,2)
        preference: (Nv, Nq, 2)
        power: (Nv,)
    """
    w = s[:,newaxis,newaxis] * preference
    


class VotersState:
    __slots__ = ['power', 'preference']
	
