# Dataset wrapper needed to use VIBO on the Cogntive Tutor logs.

import re
import collections
import torch
import numpy as np
import argparse
import json
import subprocess
from environment import RustEnvironment
from tqdm import tqdm


def extract_problem(p, canonicalize_problems=False):
    equality = re.sub('(\\.[0-9]+)', '', p)
#    equality = p # [p.index(' ')+1:]

    if canonicalize_problems:
        # Make problems that differ only in numeric values equal
        # by assigning sequential values to the numbers.
        idx = 0
        l = re.split('[0-9.]+', equality)
        s = l[0]
        for i, t in enumerate(l[1:]):
            s += str(i+1) + t
        return s

    return equality

def parse_cognitive_tutor_log(path, canonicalize_problems=False):
    dataset = collections.defaultdict(list)

    col = {}

    with open(path) as f:
        for i, l in enumerate(f.readlines()):
            l = l.split('\t')
            if i == 0:
                col = {c:i for i, c in enumerate(l)}
            else:
                student = l[col['Anon Student Id']]
                problem = l[col['Step Name']]

                dataset[student, problem].append({ 'timestamp': l[col['Time']],
                                                   'outcome': l[col['Outcome']] })

    rows = []
    for k, v in dataset.items():
        all_outcomes = set([o['outcome'] for o in v])
        first_timestamp = min([o['timestamp'] for o in v])
        correct = True
        if len(all_outcomes) > 1 or 'OK' not in all_outcomes:
            correct = False
        rows.append({ 'student': k[0],
                      'problem': extract_problem(k[1], canonicalize_problems),
                      'timestamp': first_timestamp,
                      'correct': correct })

    return rows

class CognitiveTutorDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()

        with open(path) as f:
            observations = json.load(f)

        all_problems = list(set([row['problem'] for row in observations]))
        problem_id = dict(zip(all_problems, range(len(all_problems))))
        observations.sort(key=lambda row: row['timestamp'])
        data_by_student = collections.defaultdict(list)
        data_by_problem = collections.defaultdict(list)

        for row in observations:
            data_by_student[row['student']].append((problem_id[row['problem']],
                                                    int(row['correct'])))
            data_by_problem[row['problem']].append((row['student'],
                                                    int(row['correct'])))

        self.observations = observations
        self.obs_by_student = data_by_student
        self.obs_by_problem = data_by_problem
        self.student_ids = list(data_by_student.keys())
        self.max_observations = max(len(s_obs) for s_obs in data_by_student.values())
        self.n_students = len(data_by_student)
        self.n_problems = len(all_problems)

        self.problems = all_problems
        self.response = np.zeros((self.n_students, self.max_observations), dtype=int) - 1
        self.problem_id = np.zeros((self.n_students, self.max_observations), dtype=int)
        self.response_mask = np.zeros((self.n_students, self.max_observations), dtype=bool)

        for i, s_obs in enumerate(data_by_student.values()):
            for j, (problem, correct) in enumerate(s_obs):
                self.response[i][j] = float(correct)
                self.problem_id[i][j] = problem
                self.response_mask[i][j] = True

    def __len__(self):
        return self.response.shape[0]

    def __getitem__(self, index):
        return index, self.response[index], self.problem_id[index], self.response_mask[index]


def generate_solutions_dataset(agent, domain, output, device):
    env = RustEnvironment(domain)
    device = torch.device(device)
    q_fn = torch.load(agent, map_location=device)
    q_fn.to(device)

    dataset = []

    SIZE = 10000

    for i in tqdm(range(SIZE)):
        p = env.generate_new()
        try:
            success, history = q_fn.rollout(env, p, 30, 1)
        except:
            continue
        if success:
            dataset.append({
                'problem': p.facts[-1],
                'solution': [{'state': s.facts[-1],
                              'action': 'assumption' if s.parent_action is None else s.parent_action.action}
                             for s in q_fn.recover_solutions(history)[0]]
                })

    with open(output, 'w') as f:
        json.dump(dataset, f)
        print('Wrote', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset utils')
    parser.add_argument('--path', help='Path to the Cognitive Tutor log')
    parser.add_argument('--canonicalize', help='Whether to canonicalize problems.',
                        action='store_true')
    parser.add_argument('--reformat', help='Reformat the problems using our parser.',
                        action='store_true')
    parser.add_argument('--generate', help='Generate a dataset of solutions using a pre-trained agent.',
                        action='store_true')
    parser.add_argument('--agent', help='Path to the pre-trained agent.', required=False)
    parser.add_argument('--domain', help='What domain to generate problems in.', required=False)
    parser.add_argument('--output', help='Path to the output.')
    parser.add_argument('--device', help='Torch device to use.', default='cpu')

    opt = parser.parse_args()

    if opt.reformat:
        ds = parse_cognitive_tutor_log(opt.path, opt.canonicalize)
        # Call Racket util to reformat problems as we do.
        problems = set()
        for entry in ds:
            problems.add(entry['problem'])
        problems = list(problems)
        with open('input.txt', 'w') as f:
            f.write('\n'.join(problems))
        print('Making terms canonical...')
        p = subprocess.run(['racket', '-tm', 'canonicalize-terms.rkt'],
                           capture_output=True)
        canonical_terms = dict(zip(problems, p.stdout.decode('utf8').split('\n')[:-1]))
        print('Done. example:', repr(problems[0]), '==>', repr(canonical_terms[problems[0]]))
        for entry in ds:
            entry['problem'] = canonical_terms[entry['problem']]
        with open(opt.output, 'w') as f:
            json.dump(ds, f)
        print('Wrote', opt.output)
    elif opt.generate:
        generate_solutions_dataset(opt.agent, opt.domain, opt.output, opt.device)
