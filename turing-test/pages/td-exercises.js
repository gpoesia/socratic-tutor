import React, { useState, useEffect } from "react";
import dynamic from 'next/dynamic';
import useStore from '../lib/state';
import { useRouter } from 'next/router';
import { apiRequest } from '../lib/api';
import { extractTernaryDigits, ternaryDigitsEqual } from '../lib/ternary';
import _ from 'lodash';

import LinearProgress from '@material-ui/core/LinearProgress';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';

import { Button } from '@material-ui/core';

const T = dynamic(() => import('../lib/components/ternary-string.js'),
                  { ssr: false });

const SuzzleInstructions = dynamic(() => import('../lib/components/suzzle-instructions.js'),
                                   { ssr: false });
import CircularProgress from '@material-ui/core/CircularProgress';
import Alert from '@material-ui/lab/Alert';

const SuzzleEditor = dynamic(() => import('../lib/components/suzzle-editor.js'),
                             { ssr: false });

// This page has 4 states:
// 1- (start state) when there's no exercise, it requests one, and renders "loading".
//    It might either receive an exercise (goes to 2) or not (end, redirects to /td-post-test).
const LOAD_STATE = 'load';
// 2- When there is an exercise, it waits for user input, until the user submits.
const USER_INPUT_STATE = 'user-input';
// 3- If the user got it right, it shows a success message for a few seconds, and goes back to 1.
const SUCCESS_STATE = 'show-success';
// 4- If the user got it wrong, it shows the solution step-by-step, then goes back to 1.
const SHOW_SOLUTION_STATE = 'show-solution';

const formatAction = (action) => {
  if (!action) {
    return "Initial suzzle:";
  }

  const digits = extractTernaryDigits(action);
  if (action.startsWith("swap")) {
    return <span>Swap <T digits={digits[0]} /> and <T digits={digits[1]} /> to get:</span>;
  } else if (action.startsWith("del")) {
    return <span>Erase <T digits={digits[0]} /> to get:</span>;
  } else {
    return <span>Combine <T digits={digits[0]} /> and <T digits={digits[1]} /> to get:</span>;
  }
}

const TernaryExercises = () => {
  const state = useRouter();
  const sessionId = useStore(state => state.id);

  const router = useRouter();
  const [pageState, setPageState] = useState(LOAD_STATE);
  const [steps, setSteps] = useState([[]]);
  const [problem, setProblem] = useState(null);
  const [shownSolutionSteps, setShownSolutionSteps] = useState(0);

  useEffect(async () => {
    if (pageState === LOAD_STATE) {
      const p = await apiRequest('curriculum', { id: sessionId });
      if (p.done) {
        router.push('/td-post-test');
      } else {
        setProblem(p);
        setPageState(USER_INPUT_STATE);
      }
    } else if (pageState === SUCCESS_STATE) {
      setTimeout(
        () => {
          setSteps([[]]);
          setPageState(LOAD_STATE);
        },
        1000);
    } else if (pageState === SHOW_SOLUTION_STATE) {
      const finished = shownSolutionSteps === problem.solution.length;

      setTimeout(() => {
        if (finished) {
          setShownSolutionSteps(0);
          setSteps([[]]);
          setPageState(LOAD_STATE);
        } else {
          setShownSolutionSteps(shownSolutionSteps + 1);
        }
      }, finished ? 3000 : 3000);

      window.scroll({ top: window.scrollMaxY, left: 0, behavior: 'smooth' });
    }
  });

  let body = null;
  let footer = null;

  if (pageState === LOAD_STATE) {
    body = <CircularProgress />;
  } else {
    body = <SuzzleEditor
             problem={problem.problem}
             steps={steps}
             setSteps={setSteps}
             onSubmit={async () => {
               const correct = ternaryDigitsEqual(
                 steps[steps.length - 1].join(''),
                 problem.solution[problem.solution.length - 1].state
               );
               await apiRequest('save-answers',
                                {
                                  'id': sessionId,
                                  'stage': 'exercises',
                                  'answers': [{
                                    'id': problem.id,
                                    'problem': problem.problem,
                                    'answer': steps[steps.length - 1],
                                    'correct': correct,
                                  }],
                                });
               if (correct) {
                 setPageState(SUCCESS_STATE);
               } else {
                 setPageState(SHOW_SOLUTION_STATE);
               }
             }}
           />;
  }

  if (pageState === SUCCESS_STATE) {
    footer = <Alert severity="success">Great, you got it correct!</Alert>;
  } else if (pageState === SHOW_SOLUTION_STATE) {
    footer = (
      <div>
        <Alert severity="error">Oh, you didn't get this one right! Here is one possible solution.</Alert>
        <List component="nav">
          {
            problem.solution.slice(0, shownSolutionSteps).map((step, i) => (
              <React.Fragment key={i}>
                <ListItem key={i}>
                  <ListItemIcon>
                    {i + 1}.
                  </ListItemIcon>
                  <ListItemText primary={formatAction(problem.solution[i].action)} />
                </ListItem>
                <ListItem>
                  <ListItemText
                    inset
                    primary={<T digits={problem.solution[i].state} />}
                  />
                </ListItem>
              </React.Fragment>
            ))
          }
        </List>
        {
          shownSolutionSteps == problem.solution.length
            ? <span>This is the final answer!</span>
            : null
        }
      </div>
    );
  }

  return (
    <div className="content">
      <div className="challenge-root">
        <div className="challenge-content">
          <h1>The Suzzle Challenge</h1>
          { body }
          { footer }
        </div>
        <div className="challenge-instructions">
          <SuzzleInstructions />
        </div>
      </div>
    </div>
  );
};

export default TernaryExercises;
