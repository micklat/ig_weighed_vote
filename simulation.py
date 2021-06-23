import numpy as np

class VotersState:
    __slots__ = ['power', 'value']

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

    # until convergence:
        # after a decision is made:
        #   for each winner:
        #       payment[voter,vote, choice] = value[voter,vote, choice] * value[:,vote,losing_choice].sum() / value[:,vote,winning_choice].sum()
        #   for each loser:
        #       payment[voter,vote] = 0
        # value[voter,:] = value[voter,:] * power[voter] / payment[voter].sum()
        # -- look out: payment[voter].sum() may be 0, for now.

    # after convergence, what we want:
    # power[voter] == payment[voter].sum()
    # winning_choice[vote] = argmax(value[:,vote,:].sum(over voters))
    # payment[voter,vote, choice] == value[voter,vote, choice] * value[:,vote,losing_choice].sum() / value[:,vote,winning_choice].sum()
    # value[voter] is proportional to preference[voter] 

    # v - voter
    # q - question (vote)
    # c - choice
    # p[v,q,c] - preference
    # w[v,q,c] - value (weight)
    # y[v,q,c] - payment by voter v for choice c in question q
    # s[v] - scale of preferences of voter v
    # t[v] - power of voter v, (total)
    #
    # w[v,q,c] = s[v] * p[v,q,c]
    # t[v] should equal y[v,:,:].sum()
    # 
    # k - key
    # k[q,c] = w[:,q,1-c].sum() / w[:,q,c].sum()
    # 
    # h[q,c] - "soft winner"
    # we want h[q,c] to be close to 0 or 1
    # if h[q,0] = 1 we say that choice 0 was selected
    # if h[q,1] = 1 we say that choice 1 was selected
    # we want h[q].sum() == 1
    # we want if w[:,q,c] > w[:,q,1-c] then h[q,c] = 1

    # loss(h,p,t) is high when constraints are not met
    # s[v] = t[v] / sum(h*k*p[v])
    # y[v,q,c] = h[v,q,c] * k[q,c] * s[v] * p[v,q,c]

    # we reduce loss via gradient descent over h
    


    def __init__(self, n_voters, initial_power):
        self.power = np.ones((n_voters,)) * initial_power

	
