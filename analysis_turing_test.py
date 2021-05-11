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

TuringTestJson = json.load(open("turing-test/pages/turing_test.json"))["experiment"]

def load_data():
    client = pymongo.MongoClient()
    sessions = client['tutor'].usersessions.find({})

    sessions = [s for s in sessions
                if s.get('endTimestamp') is not None and
                   s.get('survey', {}).get('experience', '') != 'Test'
                   and s.get('type') == 'turing']

    return sessions
def getResponses(sessions):
    responses: list[dict] = []
    for session in sessions:
        response = remove_duplicate_responses(session["exerciseResponses"])
        responses.append(response)
    return responses

def remove_duplicate_responses(exerciseResponses):
    '''for each question, extract the lastest response'''
    qa = {}

    for item in exerciseResponses:
        if item == {}: continue
        q = item["question"]
        a = item["answer"]
        timestamp  = item["timestamp"]
        if (q not in qa) or (q in qa and timestamp> qa[q]["timestamp"]):
            qa[q] = {"answer":a, "timestamp": timestamp}
    return qa

def session_length(session, data):
    return (session['endTimestamp'] - session['beginTimestamp']).total_seconds() / 60

def count_results(responses):
    count = {}
    for response in responses:
        for question in response:
            answer = response[question]["answer"]
            for x in answer:
                count[x] = count.get(x, 0) +1
    return count


if __name__ == '__main__':
    res = load_data()
    res = getResponses(res)
    res = count_results(res)
    print(res)
