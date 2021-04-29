import datetime
import pickle
import traceback
import hashlib
import os
import util
from environment import Environment
from q_function import InverseLength, RandomQFunction, StateRNNValueFn
from evaluation import SuccessRatePolicyEvaluator
import torch
import wandb
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


config_example = {
    "checkpoint_path":"simple-vf-SimpleVF-equations-ct0-ck400.pt",
    "domain": "equations",
    "environment_url": "http://localhost:9876",
}


def plot_solution_embeddings(embeddings, annotations: list = None, plt_name = None):
    'PCA plot for a list of embeddings'
    pca = PCA(n_components=2)
    data = pca.fit_transform(embeddings).transpose()
    x, y = data[0], data[1]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(x, y, c='g')
    if annotations:
        for i, l in enumerate(annotations):
            if not l: l = ""
            ax.annotate(str(i)+": "+l, (x[i], y[i]))
    else:
        for i in range(len(x)):
            ax.annotate(i, (x[i], y[i]))
    if plt_name:
        plt.savefig("embedding_plots/"+ plt_name+".png")

def understand_embedding(config:dict, device, num_plots = 10):
    'Solve #num_plots randomly-generated problems'
    'and plot states embeddings from the solution path'

    try:
        checkpoint_path = config['checkpoint_path']
        q = torch.load(checkpoint_path, map_location=device)
        q.to(device)
        q.device = device
        env = Environment.from_config(config)
        seed = config.get('seed', 100)
        max_steps = config.get('max_steps', 20)
        beam_size = config.get('beam_size', 5)
        debug = config.get('debug', False)
        i = 0
        total_success = 0
        while True:
            problem = env.generate_new(seed=(seed + i))
            i+=1
            success, history = q.rollout(env, problem,max_steps, beam_size, debug)
            if success:
                solutions = q.recover_solutions(history)[0]
                solution_actions = [s.parent_action for s in solutions]
                solution_actions_str = [action.action if action else "" for action in solution_actions]
                state_str = [state.facts[-1] for state in solutions]
                embeddings = q.embed_states(solutions).detach().numpy()
                plot_solution_embeddings(embeddings, annotations = state_str, plt_name="seed" + str(seed)+"-"+ str(i))
                total_success+=1
                if total_success >= num_plots:
                    break
    except FileNotFoundError:
        print('Checkpoint', i, 'does not exist -- stopping.')


if __name__ == '__main__':
    understand_embedding(config_example, torch.device('cpu'))

