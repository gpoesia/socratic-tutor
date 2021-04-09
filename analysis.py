import matplotlib
from matplotlib import pyplot as plt
import json
import pymongo
import numpy as np
import collections
from scipy.stats import norm
import argparse
import pickle
from agent import State, Action
import dateutil

def load_data(config):
    client = pymongo.MongoClient()
    sessions = client['tutor'].usersessions.find({})

    sessions = [s for s in sessions
                if s.get('endTimestamp') is not None and
                   s.get('survey', {}).get('experience', '') != 'Test']

    with open(config['testProblems']) as f:
        c = json.load(f)
    testProblems = { str(p['id']) : p for p in c['testProblems'] }

    return { 'sessions': sessions, 'testProblems': testProblems }

def test_score(testAnswers, problems):
    c = []
    for a in testAnswers:
        c.append(a['answer'] == problems[a['id']]['solution'])
    return np.mean(c)

def session_length(session, data):
    return (session['endTimestamp'] - session['beginTimestamp']).total_seconds() / 60

def pretest_score(session, data):
    return test_score(session['preTestResponses'], data['testProblems'])

def posttest_score(session, data):
    return test_score(session['postTestResponses'], data['testProblems'])

def correctness_exercise_phase(session, data):
    correct = sum([1 for e in session['exerciseResponses'] if e['response'] == 0])
    return correct / len(session['exerciseResponses'])

def compute_statistics(scores):
    return { 'raw': np.array(scores),
             'mean': np.mean(scores),
             'max': np.max(scores),
             'min': np.min(scores) }

def aggregate_session_statistic(f, data):
    scores = []

    for s in data['sessions']:
        scores.append(f(s, data))

    return compute_statistics(scores)

def exercise_correctness(k, responses, data):
    correct = sum([1 for e in responses if e['response'] == 0])
    return correct / len(responses)

def number_of_occurrences(k, responses, data):
    return len(responses)

def aggregate_exercise_statistic(f, data):
    responses_by_exercise = collections.defaultdict(list)

    for i, s in enumerate(data['sessions']):
        for r in s['exerciseResponses']:
            responses_by_exercise[r['id']].append({ **r, 'session': s })

    scores = []

    for k, v in responses_by_exercise.items():
        scores.append(f(k, v, data))

    return compute_statistics(scores)

def ith_question(i, question, response, state):
    return True, i // 2, state

def ith_question_with_op(op, n=1):
    def criterion(i, question, response, state):
        if question.count(op) < n:
            return False, None, None
        return True, i // 2, state
    return criterion

def bernoulli_ci(values):
    p = np.mean(values)
    return (p, norm.ppf(0.975) * np.sqrt(p * (1 - p) / len(values)), len(values))

def analyze_student_success_rate(dataset, criterion):
    results = collections.defaultdict(list)

    for st in dataset.obs_by_student.values():
        student_state = {}
        i = 0
        for q, r in st:
            use, key, student_state = criterion(i, dataset.problems[q], r, student_state)
            if use:
                i += 1
                results[key].append(r)

    return { k:bernoulli_ci(v)  for k, v in results.items() }

def question_difficulty(q, r):
    return True, q

def question_length(q, r):
    return True, len(q)

def analyze_question_difficulty(dataset, criterion):
    results = collections.defaultdict(list)

    for k, v in dataset.obs_by_problem.items():
        for st, r in v:
            use, key = criterion(k, r)
            if use:
                results[key].append(r)

    return { k:bernoulli_ci(v)  for k, v in results.items() }

def compare_learning_algorithms(config):
    print('Comparing learning algorithms...')
    results = config['results']
    output = config['output']

    data_points = collections.defaultdict(list)

    for path in results:
        with open(path, 'rb') as f:
            r = pickle.load(f)

        for p in r:
            algorithm, domain = p['name'], p['domain']
            data_points[algorithm, domain].append(p)

    algorithms = list(set(k[0] for k in data_points.keys()))
    domains = list(set(k[1] for k in data_points.keys()))

    success_rate = {}

    for a in algorithms:
        for d in domains:
            if len(data_points[a, d]):
                success_rate[a, d] = '{:.3f}'.format(max(map(lambda r: r['success_rate'],
                                                             data_points[a, d])))
            else:
                success_rate[a, d] = 'N/A'

    with open(output, 'w') as f:
        f.write(f'\\begin{{tabular}}{{| l | {" c" * len(domains)} |}}\n')
        f.write('\\hline')

        headers = ['Algorithm'] + domains
        f.write(' & '.join('\\textbf{{{}}}'.format(c) for c in headers))
        f.write('\\\\ \\hline\n')

        for a in algorithms:
            line = [a]

            for d in domains:
                line.append(success_rate[a, d])

            f.write(' & '.join(line))
            f.write('\\\\\n')

        f.write('\\end{tabular}\n')

    print('Wrote', output)

def compare_agents(config):
    if config.get('compare_learning_algorithms'):
        compare_learning_algorithms(config['compare_learning_algorithms'])

def analyze_user_study(config):
    dump = config['db_dump']

    db = json.load(open(dump))
    i = 0

    for row in db:
        if not row.get('endTimestamp'):
            continue

        i += 1
        time_taken = (dateutil.parser.parse(row['endTimestamp']['$date']) -
                      dateutil.parser.parse(row['beginTimestamp']['$date']))

        n_exercises = len(row['exerciseResponses'])
        n_post_test_questions = len(row['postTestResponses'])
        exercise_score = sum(1 for e in row['exerciseResponses'] if e['correct'])
        post_test_score = sum(1 for e in row['postTestResponses'] if e['correct'])

        print(f'Participant #{i}:')
        print('Curriculum:', row['curriculum'])
        print('Time elapsed:', time_taken)
        print('Survey:', row['survey'])
        print(f'Exercise phase: {exercise_score}/{n_exercises} correct')
        print(f'Post test: {post_test_score}/{n_post_test_questions} correct')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze experiments & user study results')
    parser.add_argument('--config', required=True, help='Path to config file.')
    parser.add_argument('--user-study', help='Analyze data from the tutoring user study.',
                        action='store_true')
    parser.add_argument('--agent-comparison', help='Compare different learning agents.',
                        action='store_true')
    opt = parser.parse_args()

    config = json.load(open(opt.config))

    if opt.user_study:
        analyze_user_study(config)
    elif opt.agent_comparison:
        compare_agents(config)
