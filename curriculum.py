# Builds a curriculum leveraging problem embeddings.

import argparse
import json
import torch
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np

from agent import Environment, QFunction, DRRN

def build_curriculum(config, device):
    domain = config['domain']
    radius = config['radius']

    env = Environment(config['environment_url'], domain)
    q_fn = torch.load('models/drrn-ternary-q-m2.pt', map_location=device)
    q_fn.device = device
    q_fn.state_vocab.device = device
    q_fn.action_vocab.device = device

    print('Fetching problems...')
    problems = [env.generate_new(domain, seed=(i + config.get('seed', 0)))
                                 for i in tqdm(range(config['n_problems']))]

    with torch.no_grad():
        embeddings = q_fn.embed_states(problems)

    print('Starting with embeddings matrix', embeddings.shape)

    X = embeddings.numpy() # TSNE().fit_transform(embeddings.numpy())
    if config.get('normalize'):
        X /= X.sum(axis=1).reshape(-1, 1)

    d = sklearn.metrics.pairwise_distances(X)

    initial = 68
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Build curricula using problem embeddings.")
    parser.add_argument('--config', help='Path to config file.', required=True)
    parser.add_argument('--gpu', type=int, default=None, help='Which GPU to use.')

    opt = parser.parse_args()
    build_curriculum(json.load(open(opt.config)),
                     torch.device('cpu') if not opt.gpu else torch.device(opt.gpu))
