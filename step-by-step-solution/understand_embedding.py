import datetime
import pickle
import traceback
import hashlib
import os
import util
import argparse
from environment import Environment, State
from q_function import InverseLength, RandomQFunction, StateRNNValueFn
from evaluation import SuccessRatePolicyEvaluator
import torch
import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

config_example = {
    "checkpoint_path":"simple-vf-SimpleVF-equations-ct0-ck400.pt",
    "domain": "equations-ct",
    "environment_url": "http://localhost:9876",
}

## Command to visualize step by step solution:
## python understand_embedding.py --visualize --visualize_file_path trimmed_solutions.pickle --visualize_idx 20

## Command to visualize stats for different separation distance
## python understand_embedding.py --experiment --experiment_file_path equations-ct-embeddings.pkl


def plot_embeddings(embeddings, annotations: list = None, plt_name = None):
    'PCA plot for a list of embeddings (from a single problem)'
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

def save_solutions_and_embeddings(solutions: list[list[State]], embeddings, file_path: str, backup_path:str):
    'Append a list of solutions and embeddings to a file. '

    if not os.path.isfile(file_path):
        with open(file_path,'wb') as file:
            pickle.dump({"solutions":[], "embeddings":[]}, file)
    if not os.path.isfile(backup_path):
        with open(backup_path,'wb') as file:
            pickle.dump({"solutions":[], "embeddings":[]}, file)

    sol_emb: dict
    #save current data in backup first
    with open(file_path, 'rb') as file:
        with open(backup_path, 'wb') as back_up:
            sol_emb = pickle.load(file)
            pickle.dump(sol_emb, back_up, protocol=pickle.HIGHEST_PROTOCOL)
    #save updated data in file
    with open(file_path, 'wb') as file:
        sol_emb["solutions"].extend(solutions)
        sol_emb["embeddings"].extend(embeddings)
        pickle.dump(sol_emb, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_solutions_and_embeddings(file_path):
    with open(file_path, 'rb') as file:
        sol_emb = pickle.load(file)
        return sol_emb["solutions"], sol_emb["embeddings"]

def generate_solutions_embeddings(config:dict, device, num_problems = 100, save_embeddings: bool= True, file_path:str = None, backup_path:str = None, generate_plots:bool = False):
    '''Successfully solve #num_problems randomly generated problems and'''
    '''save solution trajectory (list of states) and embeddings to file_path'''
    try:

        print("Loading model from checkpint...")
        checkpoint_path = config['checkpoint_path']
        q = torch.load(checkpoint_path, map_location=device)
        q.to(device)
        q.device = device
        env = Environment.from_config(config)
        seed = config.get('seed', 0)
        max_steps = config.get('max_steps', 30)
        beam_size = config.get('beam_size', 5)
        debug = config.get('debug', False)
        i = 0
        total_success = 0
        solutions, embeddings = [], []

        print("Solving randam problems using seed ", seed, " beam_size ", beam_size, " max_steps ", max_steps, "...")
        while total_success < num_problems:
            problem = env.generate_new(seed=(seed + i))
            i+=1
            success, history = q.rollout(env, problem,max_steps, beam_size, debug)
            if success:
                print("Solved ", total_success, " problems so far.")
                total_success+=1

                solution = q.recover_solutions(history)[0]
                solutions.append(solution)
                embedding = q.embed_states(solution).detach().numpy()
                embeddings.append(embedding)

                if generate_plots:
                    'plot embeddings'
                    state_str = [state.facts[-1] for state in solution]
                    plot_embeddings(embedding, annotations = state_str, plt_name="equations-ct-"+str(seed) + "-"+str(i))
                if save_embeddings:
                    'Save on every iteration to avoid lose due to crashes'
                    save_solutions_and_embeddings([solution], [embedding], file_path, backup_path)
        return solutions, embeddings
    except FileNotFoundError:
        print('Checkpoint', i, 'does not exist -- stopping.')


def plot_embeddings_from_file(config:dict, device, file_path:str, num_plots = 10):
    'Plot states embeddings from a solution file'
    solutions, embeddings = load_solutions_and_embeddings(file_path)
    for idx, sol_embed in enumerate(zip(solutions, embeddings))[:num_plots]:
        solution, embedding = sol_embed
        # solution_actions = [s.parent_action for s in solutions]
        # solution_actions_str = [action.action if action else "" for action in solution_actions]
        state_str = [state.facts[-1] for state in solutions]
        plot_solution_embeddings(embedding, annotations = state_str, plt_name=file_path.split(".")[0]+"-"+ str(idx))

def trim_solutions(solutions: list[list[State]], embeddings, min_separation_dist: float):
    '''Trim step-by-step solutions by grouping nearby states'''
    trimmed_solutions: list[list[State]] = []
    ignored_actions = {}
    for solution, embedding in zip(solutions, embeddings):
        trimmed_solution = [solution[0]] #always add the last step
        trimmed_embedding = [embedding[0]]
        solution_actions = [s.parent_action for s in solution]
        solution_actions_str = [action.action.split(' ')[0] if action else "" for action in solution_actions]

        for i in range(1, len(embedding)-1):
            'calculate the distance between the current state and the last state in the trimmed solution'
            dist = np.linalg.norm(embedding[i] - trimmed_embedding[-1])
            if dist>= min_separation_dist:
                trimmed_solution.append(solution[i])
                trimmed_embedding.append(embedding[i])
            else:
                ignored_actions[solution_actions_str[i]] = ignored_actions.get(solution_actions_str[i], 0) + 1
        trimmed_solution.append(solution[-1]) #always add the last step
        trimmed_solutions.append(trimmed_solution)
    print("frequency of ignored actions", sorted(ignored_actions.items(), key =
             lambda kv:(kv[1], kv[0])))
    return trimmed_solutions


###just trying out an idea here, not used anywhere
def trim_solutions_pairwise(solutions: list[list[State]], embeddings, min_separation_dist: float):
    '''Trim step-by-step solutions by grouping nearby states'''
    trimmed_solutions: list[list[State]] = []
    ignored_actions = {}
    for solution, embedding in zip(solutions, embeddings):
        trimmed_solution = [solution[0]] #always add the last step
        trimmed_embedding = [embedding[0]]
        solution_actions = [s.parent_action for s in solution]
        solution_actions_str = [action.action.split(' ')[0] if action else "" for action in solution_actions]

        for i in range(1, len(embedding)-1):
            'calculate the distance between the current state and the last state in the fall solution'
            dist = np.linalg.norm(embedding[i] - embedding[i-1])
            if dist>= min_separation_dist:
                trimmed_solution.append(solution[i])
                trimmed_embedding.append(embedding[i])
            else:
                ignored_actions[solution_actions_str[i]] = ignored_actions.get(solution_actions_str[i], 0) + 1
        trimmed_solution.append(solution[-1]) #always add the last step
        trimmed_solutions.append(trimmed_solution)
    print("frequency of ignored actions", sorted(ignored_actions.items(), key =
             lambda kv:(kv[1], kv[0])))
    return trimmed_solutions
def print_step_by_step_solution(solution):
    state_str = [state.facts[-1] for state in solution]
    for idx, line in enumerate(state_str):
        print(idx, ":           ", line)

def try_diff_separation_dist(file_name, output_file_name:str = None):
    solutions, embeddings = load_solutions_and_embeddings(file_name)
    try_dist = [0, 1, 2, 3, 4, 5,6, 7, 8,9, 10, 11, 12, 13, 14, 15]
    trimmed_solutions_dict = {}
    for min_separation_dist in try_dist:
        trimmed_solutions = trim_solutions(solutions, embeddings, min_separation_dist)
        sol_lens = [len(solution) for solution in trimmed_solutions]
        print("min_separation_dist = ", min_separation_dist)
        print("average solution length: ", sum(sol_lens)/(len(sol_lens)+0.000001))
        print("max solution length: ", max(sol_lens))
        print("min solution length: ", min(sol_lens))
        trimmed_solutions_dict[min_separation_dist] = trimmed_solutions
    if output_file_name:
        with open(output_file_name, 'wb') as file:
            pickle.dump(trimmed_solutions_dict, file)

def visualize(file_name, problem_idx):
    '''Print step-by-step solutions for a particular problem specified by problem_idx'''
    with open(file_name, 'rb') as file:
        trimmed_solutions = pickle.load(file)
        for key in trimmed_solutions:
            print("-------------- separation distance: ", key, "--------------")
            print_step_by_step_solution(trimmed_solutions[key][problem_idx])

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Shorten step by step solutions")
    parser.add_argument('--generate', help='Generate embeddings from solving random problems', action='store_true', default = False)
    parser.add_argument('--num_examples', type = int, help='How many examples to generate', default = 100)
    parser.add_argument('--save_file_name', type=str,
                        help='Save generated embedding to this file path (backup)')
    parser.add_argument('--backup_file_name', type=str,
                        help='Save generated embedding to this file path')

    parser.add_argument('--experiment', help='Try different minimum separation distance', action='store_true', default=False)
    parser.add_argument('--experiment_file_path', type=str, help='Apply different minimum separation distance on solutions in this file')
    parser.add_argument('--experiment_save_file_path', type=str, help='Save shortened solutions in this file')

    parser.add_argument('--visualize', help='Visualize step-by-step solutions from file', action='store_true', default=False)
    parser.add_argument('--visualize_file_path', type=str, help='Visualize solutions in this file')
    parser.add_argument('--visualize_idx', type=int, help='Specify the problem idx', default = 12)

    opt = parser.parse_args()
    if opt.generate:
        generate_solutions_embeddings(config_example, torch.device("cpu"), num_examples = opt.num_examples, file_path= opt.save_file_name, backup_path=opt.backup_file_name)
    elif opt.experiment:
        try_diff_separation_dist(opt.experiment_file_path, output_file_name = opt.experiment_save_file_path)
    elif opt.visualize:
        visualize(opt.visualize_file_path, opt.visualize_idx)
