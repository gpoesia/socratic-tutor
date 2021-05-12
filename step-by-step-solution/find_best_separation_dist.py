import json
import argparse
from understand_embedding import *


config = {
    "checkpoint_path":"nce-equations-ct.pt",
    "domain": "equations-ct",
    "environment_url": "http://localhost:9876",
    "environment_backend":"Rust"
}



def find_human_sol_avg_length(jsonpath):
    ''' human solution length is 1 step less than machine solution because
        machine solution contains the question as the first step'''
    data = json.load(open(jsonpath))
    total = 0
    count = 0
    for obj in data:
        total += sum([len(s) for s in obj["solutions"]])
        count += len(obj["solutions"])
    print("human average solution length is", total / count)
    return total / count  # 3.2


def find_best_separation_dist(pkl_file_path, avg_len):
    print("The average solution length aiming for is", avg_len)
    solutions, embeddings = load_solutions_and_embeddings(pkl_file_path)
    best_dist, best_avg_sol_len = try_best_separation_dist(solutions, embeddings, avg_len, floor=0, cap=15, step=0.01)
    print("best separation distance found is", best_dist)
    print("the average solution length with separation distance=", best_dist, "is", best_avg_sol_len)
    print("---------------visualize some solutions----------------")
    trimmed_solutions = trim_solutions(solutions, embeddings, best_dist)
    for solution in trimmed_solutions[:10]:
        print("-"*20)
        print_step_by_step_solution(solution)
    return best_dist  # 3.35


def try_best_separation_dist(solutions, embeddings, avg_len, floor=0, cap=15, step=0.01):
    '''loop through each candidate separation distance and return the best one'''
    best_avg_sol_len = 0
    best_dist = 0
    while floor < cap:
        trimmed_solutions = trim_solutions(solutions, embeddings, floor)
        avg_sol_lens = sum([len(solution) for solution in trimmed_solutions])/len(trimmed_solutions)
        if (abs(avg_sol_lens - avg_len) < abs(best_avg_sol_len - avg_len)):
            best_avg_sol_len = avg_sol_lens
            best_dist = floor
        floor += step
    return best_dist, best_avg_sol_len


def get_problems(jsonpath):
    data = json.load(open(jsonpath))
    return [obj["problem"] for obj in data]


def solve_problems(config: dict, device, problems: list[str], file_path: str):
    '''Use Racket enviroment to solve provided problems and'''
    '''save solution trajectory (list of states) and embeddings to file_path'''
    try:
        print("Loading model from checkpoint...")
        checkpoint_path = config['checkpoint_path']
        q = torch.load(checkpoint_path, map_location=device)
        q.to(device)
        q.device = device
        env = Environment.from_config(config)
        max_steps = config.get('max_steps', 30)
        beam_size = config.get('beam_size', 5)
        solutions, embeddings, succ_problems = [], [], []
        print("Solving", len(problems), "problems using", " beam_size ", beam_size, " max_steps ", max_steps, "...")
        for problem in problems:
            state = State([problem], [''], 0)
            success, history = q.rollout(env, state, max_steps, beam_size)
            if success:
                solution = q.recover_solutions(history)[0]
                solutions.append(solution)
                embedding = q.embed_states(solution).detach().numpy()
                embeddings.append(embedding)
                succ_problems.append(problem)
                print("success")
            else:
                print("no success")
        with open(file_path, 'wb') as file:
            pickle.dump({"solutions": solutions, "embeddings": embeddings, "problems": succ_problems}, file)
        return solutions,embeddings,succ_problems
    except FileNotFoundError:
        print("Checkpoint does not exist")


def save_machine_and_human_sol_to_json(problems, trimmed_solutions, human_json_file, output_path):
    print("Saving output json file to", output_path, "...")
    human = json.load(open(human_json_file))
    human_dict = {}
    res = []
    for obj in human:
        p = obj["problem"]
        sol_1 = obj["solutions"][0]
        sol_2 = obj["solutions"][1]
        human_dict[p] = [{ "id": "human1", "steps": sol_1}, {"id": "human2","steps": sol_2}]
    for q, sol in zip(problems, trimmed_solutions):
        res.append({
        "question": q,
        "solutions": human_dict[q] + [{"id": "nce", "steps": [state.facts[-1] for state in sol[1:]]}]
        })
    with open(output_path, 'w') as outfile:
        json.dump(res, outfile)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Prepare for Turning test")
    parser.add_argument('--human_avg', help='Find human average', action='store_true', default=False)
    parser.add_argument('--json_file', type=str, help='Json file path (human solutions)',
                        default="../data/normalized_human_solutions.json")
    parser.add_argument('--find_dist', help='Find best separation distance tuned for the provided average solution length',
                        action='store_true', default=False)
    parser.add_argument('--opt_sol_len', help='Optimal solution length to aim for', type=float, default=4.2)
    parser.add_argument('--pkl_file', type=str, help='Pickle file path (Embeddings)', default="nce_embeddings_rust.pkl")
    opt = parser.parse_args()

    if opt.human_avg:
        find_human_sol_avg_length(opt.json_file)
    elif opt.find_dist:
        find_best_separation_dist(opt.pkl_file, opt.opt_sol_len)
    else:
        problems = get_problems(opt.json_file)
        solutions, embeddings, problems= solve_problems(config, torch.device("cpu"), problems, "machine_solutions.pickle")
        # solutions, embeddings = load_solutions_and_embeddings("nce_embeddings_rust.pkl")
        # for solution in solutions[:10]:
        #     print("-"*20)
        #     print_step_by_step_solution(solution)

        trimmed_solutions = trim_solutions(solutions, embeddings, 3.8799999999999613)
        save_machine_and_human_sol_to_json(problems, trimmed_solutions, opt.json_file, "human_and_machine_sols.json")
