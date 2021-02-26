# Implementation of Reinforcement Learning agents that interact with the
# educational domain environment implemented in Racket.

import urllib
import requests

class State:
    def __init__(self, facts, goals, value):
        self.facts = tuple(facts)
        self.goals = tuple(goals)
        self.value = value

    def __hash__(self):
        return hash(self.facts)

    def __str__(self):
        return 'State({})'.format(self.facts[-1])

    def __repr__(self):
        return str(self)

class Action:
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __str__(self):
        return 'Action({})'.format(self.action)

    def __repr__(self):
        return str(self)

class Environment:
    def __init__(self, url, default_domain=None):
        self.url = url
        self.default_domain = default_domain

    def generate_new(self, domain=None):
        domain = domain or self.default_domain
        response = requests.post(self.url + '/generate',
                                 json={'domain': domain}).json()
        return State(response['state'], response['goals'], 0.0)

    def step(self, state, domain=None):
        domain = domain or self.default_domain
        response = requests.post(self.url + '/step',
                                 json={'domain': domain,
                                       'state': state.facts,
                                       'goals': state.goals }).json()
        reward = response['success']
        return reward, [Action(state,
                               a['action'],
                               State(state.facts + (a['state'],), state.goals, 0.0),
                               0.0)
                        for a in response['actions']]

if __name__ == '__main__':
    e = Environment('http://localhost:8832')
