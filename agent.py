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

class QFunction(nn.Module):
    """A Q-Function estimates the total expected reward of taking a certain
       action given that the agent is at a certain state. This module
       batches the computation and evaluates a set of actions given one state."""

    def forward(self, state, actions):
        raise NotImplemented()

    def rollout(self, environment, state, max_steps):
        """Greedily picks the best action according to the Q value until either
        max_steps have been made or reached a terminal state."""
        history = [state]
        success = False

        for i in range(max_steps):
            reward, actions = environment.step([history[-1]])[0]
            if reward:
                success = True
                break

            if len(actions) == 0:
                success = False
                break

            with torch.no_grad():
                q_values = self(history[-1], actions)

            _, best_action = max(list(zip(q_values, actions)),
                                 key=lambda aq: aq[0])
            history.append(best_action.next_state)

        return success, history

class SuccessRatePolicyEvaluator:
    """Evaluates the policy derived from a Q function by its success rate at solving
       problems generated by an environment."""
    def __init__(self, environment, config):
        self.environment = environment
        self.seed = config.get('seed', 0)
        self.n_problems = config.get('n_problems', 100) # How many problems to use.
        self.max_steps = config.get('max_steps', 30) # Maximum length of an episode.

    def evaluate(self, q, verbose=False):
        successes, failures = [], []

        for i in range(self.n_problems):
            problem = self.environment.generate_new(seed=(self.seed + i))
            success, _ = q.rollout(self.environment, problem, self.max_steps)
            if success:
                successes.append(problem)
            else:
                failures.append(problem)
            if verbose:
                print(i, problem, '-- success?', success)

        return {
            'success_rate': len(successes) / self.n_problems,
            'successes': successes,
            'failures': failures,
        }

class Environment:
    def __init__(self, url, default_domain=None):
        self.url = url
        self.default_domain = default_domain

    def generate_new(self, domain=None, seed=None):
        domain = domain or self.default_domain
        params = {'domain': domain}
        if seed is not None:
            params['seed'] = seed
        response = requests.post(self.url + '/generate', json=params).json()
        return State(response['state'], response['goals'], 0.0)

    def step(self, states, domain=None):
        domain = domain or self.default_domain
        response = requests.post(self.url + '/step',
                                 json={'domain': domain,
                                       'states': [s.facts for s in states],
                                       'goals': [s.goals for s in states]}).json()
        rewards = [r['success'] for r in response]
        actions = [[Action(state,
                           a['action'],
                           State(state.facts + (a['state'],), state.goals, 0.0),
                           0.0)
                    for a in r['actions']]
                   for state, r in zip(states, response)]
        return list(zip(rewards, actions))

class EnvironmentWithEvaluationProxy:
    '''Wrapper around the environment that triggers an evaluation every K calls'''
    def __init__(self, enviromnent, q_function, config={}):
        self.environment = environment
        self.q_function = q_function
        self.n_steps = 0

        self.domains = config['domains']
        self.evaluate_every = config['evaluate_every']
        self.eval_config = config['eval_config']
        self.name = config['name']
        self.output_path = config['output']

        self.results = []
        self.n_new_problems = 0
        self.cumulative_reward = 0

    def generate_new(self, domain=None, seed=None):
        self.n_new_problems += 1
        return self.environment.generate_new(domain, seed)

    def step(self, states, domain=None):
        if (self.n_steps + 1) % self.evaluate_every == 0:
            self.evaluate()

        self.n_steps += 1
        reward_and_actions = self.environment.step(state, domain)
        self.cumulative_reward += sum(rw for rw, _ in reward_and_actions)

        return results

    def evaluate(self):
        evaluator = SuccessRatePolicyEvaluator(self.environment, self.eval_config)
        results = evaluator.evaluate()
        results['n_steps'] = self.n_steps
        results['problems_seen'] = self.n_new_problems
        results['name'] = self.name

        with open(self.output_path, 'w') as f:
            f.write(results)

class DRRN(QFunction):
    def __init__(self, config):
        super().__init__()

        char_emb_dim = config.get('char_emb_dim', 32)
        self.hidden_dim = hidden_dim = config.get('hidden_dim', 32)
        self.lstm_layers = config.get('lstm_layers', 1)

        self.state_vocab = CharEncoding({ 'embedding_dim': char_emb_dim })
        self.action_vocab = CharEncoding({ 'embedding_dim': char_emb_dim })
        self.state_encoder = nn.LSTM(char_emb_dim, hidden_dim,
                                     self.lstm_layers, bidirectional=True)
        self.action_encoder = nn.LSTM(char_emb_dim, hidden_dim,
                                      self.lstm_layers, bidirectional=True)

    def forward(self, state, actions):
        state = state.facts[-1]
        actions = [a.action for a in actions]
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

class RandomQFunction(QFunction):
    def __init__(self):
        super().__init__()

    def forward(self, state, actions):
        return torch.rand(len(actions))

if __name__ == '__main__':
    e = Environment('http://localhost:9898', 'equations')
    m = DRRN({})
    rq = RandomQFunction()
    evaluator = SuccessRatePolicyEvaluator(e, {})
