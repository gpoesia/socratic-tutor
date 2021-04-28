'Connects to either the Racket or Rust environments.'

import argparse
import requests
import random
import time

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


class Environment:
    'Generic environment back-end'
    def generate_new(self, domain: str, seed: int = None) -> State:
        raise NotImplementedError()

    def step(self, states: list[State], domain: str = None) -> list[tuple[bool, list[Action]]]:
        raise NotImplementedError()

    @staticmethod
    def from_config(config: dict):
        'Returns the appropriate environment given the experiment configuration options.'
        if config.get('environment_backend') == 'Rust':
            return RustEnvironment(config.get('domain'))
        else:
            return RacketEnvironment(config['environment_url'], config.get('domain'))


class RacketEnvironment(Environment):
    'Environment wrapper that sends HTTP requests to the Racket Web server.'
    def __init__(self, url, default_domain=None):
        self.url = url
        self.default_domain = default_domain
        self.next_seed = 0

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
    def __init__(self, default_domain=None):
        if not COMMONCORE_AVAILABLE:
            raise RuntimeError('Could not load commoncore.so')
        self.default_domain = default_domain
        self.next_seed = 0

    def generate_new(self, domain=None, seed=None):
        domain = domain or self.default_domain
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        problem = commoncore.generate(domain, seed)
        return State([problem], [''], 0.0)

    def step(self, states, domain=None):
        domain = domain or self.default_domain

        try:
            next_states = commoncore.step(domain, [s.facts[-1] for s in states])
        except:
            print('Error stepping', states)
            raise

        rewards = [int(ns is None) for ns in next_states]
        actions = [[Action(state,
                           formal_desc,
                           State(state.facts + (next_state,), state.goals, 0.0),
                           0.0)
                    for (next_state, formal_desc, human_desc) in (actions or [])]
                   for state, actions in zip(states, next_states)]

        for i, (s, sa) in enumerate(zip(states, actions)):
            s.value = rewards[i]
            for a in sa:
                a.next_state.parent_action = a

        return list(zip(rewards, actions))


def interact(environment):
    print('Enter a problem, or empty to generate a random one:')
    problem = input('>>> ')

    if not problem:
        state = environment.generate_new(seed=random.randint(0, 10**6))
    else:
        state = State([problem], ['x = ?'], 0)

    while True:
        print('State:', state)
        reward, actions = environment.step([state])[0]

        if reward:
            print('Solved!')
            break

        for i, s in enumerate(actions):
            print(f'{i}.\t{s.next_state.facts[-1]}')

        choice = input('Choose next state: ')
        state = actions[int(choice)].next_state


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Interact directly with the environment.")
    parser.add_argument('--rust', help='Use the Rust back-end', action='store_true')
    parser.add_argument('--racket-url', type=str,
                        help='Use the Racket backend at the provided URL.')
    parser.add_argument('--interact', help='Solve problems interactively', action='store_true')
    parser.add_argument('--benchmark', help='Run a small benchmark of the environment', action='store_true')
    parser.add_argument('--domain', type=str,
                        help='What domain to use.', default='equations-ct')

    opt = parser.parse_args()

    if opt.rust:
        assert COMMONCORE_AVAILABLE, "Could not find commoncore.so"
        env: Environment = RustEnvironment(opt.domain)
    else:
        assert opt.racket_url, 'Need a URL to use the Racket environment: either pass --racket-url or --rust'
        env = RacketEnvironment(opt.racket_url, opt.domain)

    if opt.benchmark:
        benchmark(env)
    elif opt.interact:
        interact(env)
