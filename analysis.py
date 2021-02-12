import matplotlib
from matplotlib import pyplot as plt
import json
import pymongo
import numpy as np
import collections

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
