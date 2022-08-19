'Connects to either the Racket or Rust environments.'

import argparse
import requests
import json
import pickle
import random
import time
import torch
import numpy as np

import abs_util
from steps import Step, Solution
from abstractions import Rule, Axiom, ABS_TYPES

try:
    import commoncore
    COMMONCORE_AVAILABLE = True
except ModuleNotFoundError:
    print('Rust backend not loaded - use the Racket server.')
    COMMONCORE_AVAILABLE = False


class State:
    'Represents a state, which is equivalent to a problem in our domains.'
    def __init__(self, facts: list[str], goals: list[str], value: float, parent_action: 'Action' = None):
        self.facts = tuple(facts)
        self.goals = tuple(goals)
        self.value = value
        self.parent_action = parent_action

    def __hash__(self):
        return hash(self.facts[-1])

    def __str__(self):
        if self.parent_action:
            return 'State({} | {})'.format(self.facts[-1],
                                           self.parent_action.action)
        return 'State({})'.format(self.facts[-1])

    def __repr__(self):
        return str(self)

    def __eq__(self, rhs):
        return isinstance(rhs, State) and self.facts[-1] == rhs.facts[-1]


class Action:
    'Represents an action, with the pair of states that it connects.'
    def __init__(self, state, action, next_state, reward, value=0.0):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.value = value

    def __str__(self):
        return 'Action({})'.format(self.action)

    def __repr__(self):
        return str(self)


def random_initial_seed():
    return random.randint(10**7, 10**8)


class Environment:
    'Generic environment back-end'
    def generate_new(self, domain: str, seed: int = None) -> State:
        raise NotImplementedError()

    def step(self, states: list[State], domain: str = None) -> list[tuple[bool, list[Action]]]:
        raise NotImplementedError()

    def train(self):
        pass

    def test(self):
        pass

    @staticmethod
    def from_config(config: dict):
        'Returns the appropriate environment given the experiment configuration options.'
        if config.get('environment_backend') == 'Rust':
            env = RustEnvironment(config.get('domain'), config.get('abstractions'))
        else:
            env = RacketEnvironment(config['environment_url'], config.get('domain'))

        if config.get('multitask_train_domains'):
            env = MultiTaskEnvironment(env, config['multitask_train_domains'])

        return env


class RacketEnvironment(Environment):
    'Environment wrapper that sends HTTP requests to the Racket Web server.'
    def __init__(self, url, default_domain=None):
        self.url = url
        self.default_domain = default_domain
        self.next_seed = random_initial_seed()

    def generate_new(self, domain=None, seed=None):
        domain = domain or self.default_domain
        params = {'domain': domain}

        if seed is not None:
            params['seed'] = seed
        else:
            params['seed'] = self.next_seed
            self.next_seed += 1

        response = requests.post(self.url + '/generate', json=params).json()
        return State(response['state'], response['goals'], 0.0)

    def step(self, states, domain=None):
        domain = domain or self.default_domain
        try:
            response = requests.post(self.url + '/step',
                                     json={'domain': domain,
                                           'states': [s.facts for s in states],
                                           'goals': [s.goals for s in states]}).json()
        except Exception as e:
            print('Error in stepping', states, ':', e)
            print('Will try to continue silently...')
            return [(0, [])] * len(states)

        rewards = [int(r['success']) for r in response]
        actions = [[Action(state,
                           a['action'],
                           State(state.facts + (a['state'],), state.goals, 0.0),
                           0.0)
                    for a in r['actions']]
                   for state, r in zip(states, response)]

        for i, (s, sa) in enumerate(zip(states, actions)):
            s.value = rewards[i]
            for a in sa:
                a.next_state.parent_action = a

        return list(zip(rewards, actions))


class RustEnvironment(Environment):
    'Faster environment that calls into the compiled library.'
    def __init__(self, default_domain=None, abs_config=None):
        if not COMMONCORE_AVAILABLE:
            raise RuntimeError('Could not load commoncore.so')
        self.default_domain = default_domain
        self.next_seed = random_initial_seed()
        self.rules = None
        if abs_config is not None:
            self.AbsType = ABS_TYPES[abs_config['abs_type']]
            if abs_config.get('abs_ax') is not None:
                self.rules = abs_config['abs_ax']
                if isinstance(self.rules[0], str):
                    assert all(isinstance(rule_str, str) for rule_str in self.rules)
                    self.rules = [Rule.from_string(rule_str, self.AbsType) for rule_str in self.rules]
            elif abs_config.get('path') is not None:
                if abs_config['path'][-4:] == '.pkl':
                    with open(abs_config['path'], 'rb') as f:
                        self.rules = pickle.load(f)
                elif abs_config['path'][-5:] == '.json':
                    with open(abs_config['path'], 'r') as f:
                        rules = json.load(f)['axioms']
                    assert all(isinstance(rule_str, str) for rule_str in rules)
                    self.rules = [Rule.from_string(rule_str, self.AbsType) for rule_str in rules]
                else:
                    raise Exception("Invalid file extension")
            assert all(isinstance(rule, (Axiom, self.AbsType)) for rule in self.rules)
            self.rule_trie = abs_util.make_rule_trie(self.rules)

    def generate_new(self, domain=None, seed=None):
        domain = domain or self.default_domain
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        problem = commoncore.generate(domain, seed)
        return State([problem], [''], 0.0)


    def apply_abs_helper(self, state, domain, top_level=True, prev_ax=None, cur_steps=None):
        """
        Generator generating all ways to apply axioms/abstractions to state
        """
        if prev_ax is None:
            prev_ax = self.rule_trie
        if cur_steps is None:
            cur_steps = Solution([state], [])

        # abstraction completed
        if prev_ax.is_term:
            yield (state, cur_steps.display_compressed(prev_ax.value), "ABSTRACTION")

        # continuing abstractions
        next_ax = prev_ax.children
        if next_ax:
            if top_level:
                actions = commoncore.step(domain, [state])[0]
            else:
                actions = []
                for abs_elt in next_ax:
                    axiom = self.AbsType.get_ax_from_abs_elt(abs_elt)
                    new_actions = commoncore.apply(domain, [state], axiom.name)[0]
                    actions.append(new_actions)
            if actions:
                next_steps = []
                if top_level:
                    for next_state, formal_desc, _ in actions:
                        next_step = Step.from_string(formal_desc, self.AbsType)
                        try_abs_elt = self.AbsType.get_abs_elt(next_step, cur_steps)
                        if try_abs_elt in next_ax:
                            next_steps.append((next_state, next_step, try_abs_elt))
                else:
                    for action_list, abs_elt in zip(actions, next_ax):
                        if action_list:
                            for next_state, formal_desc, _ in action_list:
                                next_step = Step.from_string(formal_desc, self.AbsType)
                                try_abs_elt = self.AbsType.get_abs_elt(next_step, cur_steps)
                                if try_abs_elt == abs_elt:
                                    next_steps.append((next_state, next_step, try_abs_elt))
                if next_steps:
                    for i in range(len(next_steps) - 1 + prev_ax.is_term):
                        next_state, next_step, abs_elt = next_steps[i]
                        new_steps = Solution(cur_steps.states + [next_state], cur_steps.actions + [next_step])
                        yield from self.apply_abs_helper(next_state, domain, False, next_ax[abs_elt], new_steps)
                    if not prev_ax.is_term:
                        next_state, next_step, abs_elt = next_steps[-1]
                        cur_steps.states.append(next_state)
                        cur_steps.actions.append(next_step)
                        yield from self.apply_abs_helper(next_state, domain, False, next_ax[abs_elt], cur_steps)


    def apply_abs(self, state, domain):
        """
        List of possible next states and actions starting at state, where abstractions are allowed
        """
        # not goal state
        if commoncore.step(domain, [state])[0] is not None:
            return list(self.apply_abs_helper(state, domain))


    def step(self, states, domain=None, debug=False):
        domain = domain or self.default_domain

        try:
            # list of [(next_state (as str), formal_desc, human_desc) for each possible next state] for each current state
            if self.rules is None:
                next_states = commoncore.step(domain, [s.facts[-1] for s in states])
            else:
                next_states = [self.apply_abs(s.facts[-1], domain) for s in states]
        except:
            print('Error stepping', states, 'in', domain)
            raise

        # reward is 1 if there's no next state
        rewards = [int(ns is None) for ns in next_states]
        if debug: print("LENGTH:", len(rewards))
        # list of [Action object for each possible action] for each current state
        actions = [[Action(state,
                           formal_desc,
                           State(state.facts + (next_state,), state.goals, 0.0),
                           0.0)
                    for (next_state, formal_desc, human_desc) in (actions or [])]
                   for state, actions in zip(states, next_states)]

        # update s.value for each state based on reward; also update parent action for next states
        for i, (s, sa) in enumerate(zip(states, actions)):
            s.value = rewards[i]
            for a in sa:
                a.next_state.parent_action = a

        return list(zip(rewards, actions)) # actions[i][1] will be [] (i.e. no possible next actions) if states[i] is goal state


class MultiTaskEnvironment(Environment):
    'An environment that mixes domains during training time.'
    def __init__(self, base_environment, domains):
        self.domains = domains
        self.base_environment = base_environment
        self.default_domain = base_environment.default_domain
        print('Default domain:', self.default_domain)
        # In step(), instead of a default domain, this environment uses
        # the last one used during generation.
        self.last_domain = None
        self.train()

    def train(self):
        self.randomize_domain = True

    def test(self):
        self.randomize_domain = False

    def generate_new(self, domain=None, seed=None):
        if self.randomize_domain:
            if domain is None:
                domain = random.choice(self.domains)
                self.last_domain = domain
        else:
            domain = self.default_domain

        return self.base_environment.generate_new(domain, seed)

    def step(self, states, domain=None):
        if self.randomize_domain:
            if domain is None:
                domain = self.last_domain
        else:
            domain = self.default_domain

        return self.base_environment.step(states, domain or
                                          (self.randomize_domain and self.last_domain))


def interact(environment, scoring_model_path):
    if scoring_model_path:
        device = torch.device('cpu')
        model = torch.load(scoring_model_path, map_location=device)
        model.to(device)
    else:
        model = None

    print('Enter a problem, or empty to generate a random one:')
    problem = input('>>> ')

    if not problem:
        state = environment.generate_new(seed=random.randint(0, 10**6))
    else:
        state = State([problem], ['x = ?'], 0)

    def softmax(s):
      s = s.detach().numpy()
      return np.exp(s) / np.exp(s).sum()

    while True:
        print('State:', state)
        reward, actions = environment.step([state])[0]

        if reward:
            print('Solved!')
            break

        if model is not None:
            q = softmax(model(actions))

        for i, s in enumerate(actions):
            if model:
              print(f'{i}.\t{s.next_state.facts[-1]}\t| {s.action} {q[i]:.3f}')
            else:
              print(f'{i}.\t{s.next_state.facts[-1]}\t| {s.action}')


        choice = input('Choose next state: ')
        state = actions[int(choice)].next_state


def test(environment, scoring_model_path):
    device = torch.device('cpu')
    model = torch.load(scoring_model_path, map_location=device)
    model.to(device)

    print('Enter a problem, or empty to generate a random one:')
    problem = input('>>> ')

    if not problem:
        state = environment.generate_new(seed=random.randint(0, 10**6))
    else:
        state = State([problem], ['x = ?'], 0)

    print('State:', state)
    success, history = model.rollout(environment, state, 30, 20, debug=True)

    print('Success' if success else 'Failed')
    if success:
        print('Solution:',
              ' =>\n'.join(map(lambda s: f'{s.facts[-1]} | {s.parent_action and s.parent_action.action}',
                  model.recover_solutions(history)[0])))


def evaluate(environment, model_path, n_problems=30):
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model.to(device)
    successes = 0

    for i in range(n_problems):
        state = environment.generate_new(seed=i)
        success, history = model.rollout(environment, state, 30, 1, debug=False)
        print(f'[{i}/{n_problems}]: solved?', success)
        successes += int(success)

    print(f'{successes}/{n_problems}')


def benchmark(environment):
    before = time.time()

    for i in range(10):
        problem = env.generate_new(seed=i)
        for i in range(30):
            r, actions = env.step([problem])[0]
            if r or not actions:
                break
            problem = actions[0].next_state

    after = time.time()
    print(after - before)


def generate(environment):
    for i in range(20):
        p = environment.generate_new(seed=i)
        print(p.facts[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Interact directly with the environment.")
    parser.add_argument('--rust', help='Use the Rust back-end', action='store_true')
    parser.add_argument('--racket-url', type=str,
                        help='Use the Racket backend at the provided URL.')
    parser.add_argument('--interact', help='Solve problems interactively', action='store_true')
    parser.add_argument('--test', help='Test a model on a problem.', action='store_true')
    parser.add_argument('--evaluate', help='Test a model on a problem.', action='store_true')
    parser.add_argument('--abstract', help='Include abstractions.', type=str)
    parser.add_argument('--q-function', help='Show model-generated scores (pass a path to the model).', type=str)
    parser.add_argument('--generate', help='Prints a list of 20 problems', action='store_true')
    parser.add_argument('--benchmark', help='Run a small benchmark of the environment', action='store_true')
    parser.add_argument('--domain', type=str,
                        help='What domain to use.', default='equations-ct')

    opt = parser.parse_args()

    if opt.rust:
        assert COMMONCORE_AVAILABLE, "Could not find commoncore.so"
        env: Environment = RustEnvironment(opt.domain, {"path": opt.abstract, "tree_idx": True})
    else:
        assert opt.racket_url, 'Need a URL to use the Racket environment: either pass --racket-url or --rust'
        env = RacketEnvironment(opt.racket_url, opt.domain)

    if opt.benchmark:
        benchmark(env)
    elif opt.interact:
        interact(env, opt.q_function)
    elif opt.test:
        test(env, opt.q_function)
    elif opt.evaluate:
        evaluate(env, opt.q_function)
    elif opt.generate:
        generate(env)
