# Builds a curriculum leveraging problem embeddings.

import argparse
import json
import torch
import random
import pickle
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from flask import Flask, request

from agent import Environment, QFunction, DRRN

def find_all_solutions(env, problems, q_fn, max_steps):
    problems_with_solution = []

    for p in tqdm(problems):
        success, history = q_fn.rollout(env, p, max_steps)
        if success:
            solutions = q_fn.recover_solutions(history)
            assert len(solutions) > 0
            problems_with_solution.append((p, solutions[0]))

    return problems_with_solution

def build_curriculum(config, device):
    domain = config['domain']
    radius = config['radius']

    env = Environment(config['environment_url'], domain)
    q_fn = torch.load('models/drrn-ternary-q-m3.pt', map_location=device)

    q_fn.device = device
    q_fn.state_vocab.device = device
    q_fn.action_vocab.device = device

    print('Fetching problems...')
    problems = [env.generate_new(domain, seed=(i + config.get('seed', 0)))
                                 for i in tqdm(range(config['n_problems']))]

    print('Finding solutions...')
    problems_with_solution = find_all_solutions(env, problems, q_fn, config.get('max_steps', 30))

    problems, solutions = zip(*problems_with_solution)

    print(f'Found solutions for {len(problems_with_solution)} problems.')

    with torch.no_grad():
        embeddings = q_fn.embed_states(problems)

    print('Starting with embeddings matrix', embeddings.shape)

    if config.get('tsne'):
        X = TSNE().fit_transform(embeddings)
    if config.get('normalize'):
        X /= X.sum(axis=1).reshape(-1, 1)

    d = sklearn.metrics.pairwise_distances(X)

    # Pick smallest problem to be the initial.
    initial = min(range(len(problems)), key=lambda i: len(problems[i].facts[-1]))
    curriculum = [initial]

    print('Starting from', problems[initial])

    while True:
        elligible = []

        for i in range(len(X)):
            min_d = np.inf

            for j in curriculum:
                min_d = min(min_d, d[i][j])

            if min_d > radius:
                elligible.append((i, min_d))

        if not len(elligible):
            break

        elligible.sort(key=lambda p: p[1])
        curriculum.append(elligible[0][0])
        print('Added', problems[elligible[0][0]], '(min_d =', elligible[0][1], ')')

    print('Picked curriculum:')
    for i, s in enumerate(curriculum):
        print(i, problems[s].facts)

    data = {
        "embeddings": X,
        "pairwise_distances": d,
        "problems": problems,
        "solutions": solutions,
        "config": config,
        "static_curriculum": curriculum,
    }

    with open(config['output'], 'wb') as f:
        pickle.dump(data, f)

    print('Saved', config['output'])

def random_curriculum_next(data, student_history):
    problems = data['problems']
    seen_problems = set(r['problem'] for r in student_history)
    not_seen = set(range(len(problems))) - seen_problems
    return random.choice(list(not_seen))

def static_curriculum_next(data, student_history):
    curriculum = data['static_curriculum']
    return curriculum[len(student_history)]

def serve_curriculum(config):
    data = pickle.load(open(config['output'], 'rb'))
    problems = data['problems']
    solutions = data['solutions']

    port = config.get('port', 9191)

    print('Serving curriculum on port', port)

    app = Flask(__name__)
    @app.route('/next', methods=['POST'])
    def next():
        params = request.get_json()
        curriculum_algorithm = params.get('curriculum', 'random')
        student_history = params.get('student_history', [])

        if curriculum_algorithm == 'random':
            next_problem = random_curriculum_next(data, student_history)
        elif curriculum_algorithm == 'static':
            next_problem = static_curriculum_next(data, student_history)
        else:
            raise NotImplemented()

        return json.dumps({
            'id': next_problem,
            'problem': problems[next_problem].facts[-1],
            'solution': [
                { 'state': s.facts[-1], 'action': s.parent_action.action if s.parent_action else None }
                for s in solutions[next_problem]
            ],
        })

    app.run('127.0.0.1', port)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Build curricula using problem embeddings.")
    parser.add_argument('--build', help='Pre-compute a curriculum.', action='store_true')
    parser.add_argument('--serve', help='Start up the curriculum server.', action='store_true')
    parser.add_argument('--config', help='Path to config file.', required=True)
    parser.add_argument('--gpu', type=int, default=None, help='Which GPU to use.')

    opt = parser.parse_args()
    config = json.load(open(opt.config))

    if opt.build:
        build_curriculum(config, torch.device('cpu') if not opt.gpu else torch.device(opt.gpu))
    elif opt.serve:
        serve_curriculum(config)
