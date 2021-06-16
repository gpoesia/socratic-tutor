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

from q_function import QFunction, DRRN, StateRNNValueFn
from environment import Environment


def l2_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))

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

    env = Environment.from_config(config)
    q_fn = torch.load(config['q_function'], map_location=device)
    q_fn.to(device)

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

    X = TSNE().fit_transform(embeddings) if config.get('tsne') else embeddings

    if config.get('normalize'):
        X /= X.sum(axis=1).reshape(-1, 1)

    X = X.cpu()

    d = sklearn.metrics.pairwise_distances(X)

    # Pick smallest problem to be the initial.
    initial = min(range(len(problems)), key=lambda i: len(problems[i].facts[-1]))
    curriculum = [initial]
    curriculum_len = [initial]

    print('Starting from', problems[initial])

    # Make static length-based curriculum.
    while True:
        elligible = []

        for i in range(len(X)):
            if len(problems[i].facts[-1]) > len(problems[curriculum_len[-1]].facts[-1]):
               elligible.append((i, len(problems[i].facts[-1])))

        if not elligible:
           break

        elligible.sort(key=lambda p: p[1])
        curriculum_len.append(elligible[0][0])

    print('Picked static-len curriculum:')
    for i, s in enumerate(curriculum_len):
        print(i, problems[s].facts)

    # Make static representation-based curriculum.
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

    print('Picked static-repr curriculum:')
    for i, s in enumerate(curriculum):
        print(i, problems[s].facts)

    data = {
        "embeddings": X,
        "pairwise_distances": d,
        "problems": problems,
        "solutions": solutions,
        "config": config,
        "static_curriculum": {
            "static-repr": curriculum,
            "static-len": curriculum_len,
        },
    }

    with open(config['output'], 'wb') as f:
        pickle.dump(data, f)

    print('Saved', config['output'])

def random_curriculum_next(data, student_history):
    problems = data['problems']
    seen_problems = set(r['problem'] for r in student_history)
    not_seen = set(range(len(problems))) - seen_problems
    return random.choice(list(not_seen)) if len(not_seen) else None

def static_curriculum_next(data, curriculum_algorithm, student_history):
    curriculum = data['static_curriculum'][curriculum_algorithm]
    return (curriculum[len(student_history)]
            if len(student_history) < min(len(curriculum), data['config']['curriculum_size'])
            else None)

def dynamic_curriculum_next(data, student_history):
    '''Returns the uncovered exercise that is closest to the last exercise.'''

    if len(student_history) == 0:
        return data['static_curriculum'][0]

    last = student_history[-1]['id']
    problems = data['problems']
    radius = data['config']['radius']
    d = data['pairwise_distances']
    solved_exercises = [e['id'] for e in student_history if e['correct']]
    seen_exercises = set(e['id'] for e in student_history)
    elligible = []

    for i in range(len(problems)):
        if i in seen_exercises:
            continue

        min_d = np.inf

        for p in solved_exercises:
            min_d = min(min_d, d[p][i])

        if min_d > radius:
            elligible.append((i, d[i][last]))

        if not len(elligible):
            break

    if not len(elligible):
        # All exercises have either been seen or covered.
        return None

    # Choose the closest to the last exercise among the unseen and uncovered exercises.
    return min(elligible, key=lambda p: p[1])[0]

def sample_post_test(data, seed, n_problems):
    problems = data['problems']
    solutions = data['solutions']
    candidates = list(range(len(problems)))
    random.seed(seed)
    random.shuffle(candidates)
    return [{ 'id': p_id,
              'problem': problems[p_id].facts[-1],
              'solution': solutions[p_id][-1].facts[-1] }
            for p_id in candidates[:n_problems]]

def serve_curriculum(config):
    data = pickle.load(open(config['output'], 'rb'))
    problems = data['problems']
    solutions = data['solutions']
    size = config['curriculum_size']
    post_test_seed = config['post_test_seed']
    post_test_n_problems = config['post_test_n_problems']

    post_test = sample_post_test(data, post_test_seed, post_test_n_problems)

    port = config.get('port', 9191)

    print('Serving curriculum on port', port)

    done = json.dumps({ 'id': None, 'problem': None, 'solution': None, 'done': True })

    app = Flask(__name__)
    @app.route('/next', methods=['POST'])
    def next():
        params = request.get_json()
        curriculum_algorithm = params.get('curriculum', 'random')
        student_history = params.get('student_history', [])

        print('Fetching next from', curriculum_algorithm, 'curriculum.')
        print('History:', student_history)

        if len(student_history) >= size:
            return done

        if curriculum_algorithm == 'random':
            next_problem = random_curriculum_next(data, student_history)
        elif curriculum_algorithm.startswith('static-'):
            next_problem = static_curriculum_next(data, curriculum_algorithm, student_history)
        elif curriculum_algorithm == 'dynamic':
            next_problem = dynamic_curriculum_next(data, student_history)
        else:
            raise NotImplemented()

        if next_problem:
            return json.dumps({
                'done': False,
                'id': next_problem,
                'problem': problems[next_problem].facts[-1],
                'solution': [
                    { 'state': s.facts[-1], 'action': s.parent_action.action if s.parent_action else None }
                    for s in solutions[next_problem]
                ],
            })
        else:
            return done

    @app.route('/post-test', methods=['GET'])
    def get_post_test():
        return json.dumps(post_test)

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
        build_curriculum(config, torch.device('cpu') if opt.gpu is None else torch.device(opt.gpu))
    elif opt.serve:
        serve_curriculum(config)
