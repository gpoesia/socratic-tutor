import json
import argparse
import collections

def extract_problem(p):
    return p[p.index(' ')+1:]

def parse_cognitive_tutor_log(config):
    dataset = collections.defaultdict(list)

    col = {}

    with open(config['path']) as f:
        for i, l in enumerate(f.readlines()):
            l = l.split('\t')
            if i == 0:
                col = {c:i for i, c in enumerate(l)}
            else:
                student = l[col['Anon Student Id']]
                problem = l[col['Problem Name']]

                dataset[student, problem].append({ 'timestamp': l[col['Time']],
                                                   'outcome': l[col['Outcome']] })

    rows = []
    for k, v in dataset.items():
        all_outcomes = set([o['outcome'] for o in v])
        first_timestamp = min([o['timestamp'] for o in v])
        correct = True
        if len(all_outcomes) > 0 or 'OK' not in all_outcomes:
            correct = False
        rows.append({ 'student': k[0],
                      'problem': extract_problem(k[1]),
                      'timestamp': first_timestamp,
                      'correct': correct })

    return rows

def run_experiments(config):
    cogntive_tutor_log = parse_cognitive_tutor_log(config['cognitive_tutor_log'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file')

    opt = parser.parse_args()

    config = json.load(open(opt.config))

    run_experiments(config)
