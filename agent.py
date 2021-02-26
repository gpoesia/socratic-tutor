# Implementation of Reinforcement Learning agents that interact with the
# educational domain environment implemented in Racket.

import urllib
import requests
import torch
from torch import nn
import pytorch_lightning as pl

from domain_learner import CharEncoding

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

class DRRN(nn.Module):
    def __init__(self, config):
        super().__init__()
        char_emb_dim = 64
        self.hidden_dim = hidden_dim = 256
        self.lstm_layers = 2

        self.state_vocab = CharEncoding({ 'embedding_dim': char_emb_dim })
        self.action_vocab = CharEncoding({ 'embedding_dim': char_emb_dim })
        self.state_encoder = nn.LSTM(64, hidden_dim, self.lstm_layers, bidirectional=True)
        self.action_encoder = nn.LSTM(64, hidden_dim, self.lstm_layers, bidirectional=True)

    def forward(self, state, actions):
        A, H = len(actions), self.hidden_dim

        state_seq , _ = self.state_vocab.embed_batch([state])
        state_seq = state_seq.transpose(0, 1)
        actions_seq , _ = self.action_vocab.embed_batch(actions)
        actions_seq = actions_seq.transpose(0, 1)

        _, (state_hn, state_cn) = self.state_encoder(state_seq)
        _, (actions_hn, actions_cn) = self.state_encoder(actions_seq)

        state_embedding = (state_hn
                           .view(self.lstm_layers, 2, 1, self.hidden_dim)[-1]
                           .permute((1, 2, 0)).reshape(1, 2*H))
        actions_embedding = (actions_hn
                             .view(self.lstm_layers, 2, A, self.hidden_dim)[-1]
                             .permute((1, 2, 0)).reshape((A, 2*H)))

        q_values = actions_embedding.matmul(state_embedding.transpose(0, 1)).squeeze(1)

        return q_values

if __name__ == '__main__':
    e = Environment('http://localhost:8832')
    st = e.generate_new('ternary-addition')
    r, acts = e.step(st, 'ternary-addition')

    m = DRRN({})
    print(list(zip(acts, m(st.facts[-1], [a.action for a in acts]))))
